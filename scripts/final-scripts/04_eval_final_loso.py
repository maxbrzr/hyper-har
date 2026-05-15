from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from common import (
    ROOT,
    SharedConfig,
    SubjectFold,
    build_or_load_loso_folds,
    prepare_cfg,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
from whar_datasets import (
    Loader,
    PostProcessingPipeline,
    PreProcessingPipeline,
    WHARDatasetID,
)

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hyper_har.backbone.cbn_attention_tinierhar import PointwiseCBNAttentionTinierHAR
from hyper_har.backbone.film_tinierhar import FiLMTinierHAR
from hyper_har.backbone.tinierhar import TinierHAR
from hyper_har.config import DEFAULT_CONFIG


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SET_ENCODER_ATTENTION_MODULE = _load_module_from_path(
    "final_phase4_attention_set_encoder",
    SRC / "hyper_har" / "set-encoder" / "attention.py",
)
SET_ENCODER_SIMPLE_MODULE = _load_module_from_path(
    "final_phase4_simple_set_encoder",
    SRC / "hyper_har" / "set-encoder" / "simple.py",
)
AttentionSetEncoder = SET_ENCODER_ATTENTION_MODULE.AttentionSetEncoder
PrototypicalSetEncoder = SET_ENCODER_SIMPLE_MODULE.PrototypicalSetEncoder

MODULATOR_TYPES = {"film", "pointwise_cbn_attention"}


@dataclass(frozen=True)
class Config:
    dataset_id: str = WHARDatasetID.WEAR.value
    datasets_dir: str = str(ROOT / "datasets")
    selected_activities: list[str] | None = None
    window_overlap: float = 0.5
    subjects_per_group: int = 6
    base_train_subjects: int = 14
    meta_train_subjects: int = 6
    val_subjects: int = 3
    test_subjects: int = 1
    seed: int = 0

    film_stage_name: str = (
        "03_pointwise_cbn_attention_meta_guarded__block-start-2"
        "__attn-score-delta__attn-bound-0p5__gamma-0p2__beta-0p5"
    )
    k_values: tuple[int, ...] = (1, 4, 8, 16, 32)
    episodes_per_k: int = 100
    min_query_per_class: int = 1
    query_batch_size: int = 512
    require_all_k_activities: bool = True
    skip_missing_folds: bool = False

    output_root: str = str(ROOT / "artifacts" / "final_pipeline")
    eval_stage_name: str = "04_final_loso_eval"
    max_folds: int | None = None
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


RUN_CONFIG = Config()


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: Any, default: float) -> float:
    return float(default if value is None else value)


def _as_int(value: Any, default: int) -> int:
    return int(default if value is None else value)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_folds(
    manifest_path: Path,
    session_df: pd.DataFrame,
    window_df: pd.DataFrame,
    shared_cfg: SharedConfig,
) -> list[SubjectFold]:
    if manifest_path.exists():
        payload = _load_json(manifest_path)
        return [SubjectFold(**row) for row in payload["folds"]]
    return build_or_load_loso_folds(session_df, window_df, shared_cfg, manifest_path)


def _load_base_model(
    ckpt_path: Path,
    num_channels: int,
    num_classes: int,
    window_size: int,
    device: str,
) -> TinierHAR:
    model = TinierHAR(
        num_channels=num_channels,
        num_classes=num_classes,
        window_size=window_size,
        backbone_config=DEFAULT_CONFIG.backbone,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _build_set_encoder(
    cfg: dict[str, Any],
    base_backbone: TinierHAR,
    num_classes: int,
) -> torch.nn.Module:
    se_cfg = replace(DEFAULT_CONFIG.set_encoder, include_global_context=False)
    encoder = str(cfg.get("encoder", "attention"))
    train_mode = str(cfg.get("set_encoder_backbone_train_mode", "freeze_all"))
    force_bn_eval = _as_bool(cfg.get("force_conv_bn_eval"), True)
    if encoder == "attention":
        return AttentionSetEncoder(
            backbone=base_backbone,
            num_classes=num_classes,
            backbone_train_mode=train_mode,
            force_conv_bn_eval=force_bn_eval,
            set_encoder_config=se_cfg,
        )
    return PrototypicalSetEncoder(
        backbone=base_backbone,
        num_classes=num_classes,
        backbone_train_mode=train_mode,
        force_conv_bn_eval=force_bn_eval,
        set_encoder_config=se_cfg,
    )


def _build_conditioned_model(
    cfg: dict[str, Any],
    base_model: TinierHAR,
    subject_embedding_dim: int,
) -> torch.nn.Module:
    modulator_type = str(cfg.get("modulator_type", "film")).strip().lower()
    if modulator_type == "film":
        return FiLMTinierHAR(
            base_model=base_model,
            subject_embedding_dim=subject_embedding_dim,
            film_hidden_dim=_as_int(cfg.get("film_hidden_dim"), 128),
            film_dropout=_as_float(cfg.get("film_dropout"), 0.0),
            film_use_explosion_guard=_as_bool(
                cfg.get("film_use_explosion_guard"), False
            ),
            film_gamma_bound=_as_float(cfg.get("film_gamma_bound"), 0.5),
            film_beta_bound=_as_float(cfg.get("film_beta_bound"), 1.0),
            film_enable_conv1=_as_bool(cfg.get("film_enable_conv1"), False),
            film_modulation_mode=str(cfg.get("film_modulation_mode", "static")),
            film_condition_gru_h0=_as_bool(cfg.get("film_condition_gru_h0"), False),
        )
    if modulator_type == "pointwise_cbn_attention":
        return PointwiseCBNAttentionTinierHAR(
            base_model=base_model,
            subject_embedding_dim=subject_embedding_dim,
            modulator_hidden_dim=_as_int(cfg.get("film_hidden_dim"), 128),
            modulator_dropout=_as_float(cfg.get("film_dropout"), 0.0),
            use_tanh_gating=_as_bool(cfg.get("film_use_explosion_guard"), True),
            gamma_bound=_as_float(cfg.get("film_gamma_bound"), 0.5),
            beta_bound=_as_float(cfg.get("film_beta_bound"), 1.0),
            enable_pointwise_bn=_as_bool(
                cfg.get("modulator_enable_pointwise_bn"), True
            ),
            enable_attention_query=_as_bool(
                cfg.get("modulator_enable_attention_query"), True
            ),
            pointwise_block_start=_as_int(
                cfg.get("modulator_pointwise_block_start"), 0
            ),
            attention_adapter_type=str(
                cfg.get("modulator_attention_adapter_type", "feature_film")
            ),
            attention_score_bound=_as_float(
                cfg.get("modulator_attention_score_bound"), 1.0
            ),
        )
    raise ValueError(
        f"modulator_type must be one of {sorted(MODULATOR_TYPES)}, "
        f"got {modulator_type!r}."
    )


def _indices_for_subject(
    session_df: pd.DataFrame,
    window_df: pd.DataFrame,
    subject_id: int,
) -> list[int]:
    meta = window_df[["session_id"]].copy()
    meta["window_index"] = meta.index.astype(int)
    session_meta = session_df[
        ["session_id", "subject_id"]
    ].drop_duplicates("session_id")
    merged = meta.merge(session_meta, on="session_id", how="left")
    out = merged.loc[
        merged["subject_id"].astype("Int64") == int(subject_id), "window_index"
    ]
    return sorted(int(x) for x in out.tolist())


def _build_subject_activity_index(
    loader: Loader,
    indices: Sequence[int],
) -> dict[int, np.ndarray]:
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    subset["window_index"] = subset.index.astype(int)
    session_meta = loader.session_df[
        ["session_id", "activity_id"]
    ].drop_duplicates("session_id")
    merged = subset.merge(session_meta, on="session_id", how="left")
    if merged["activity_id"].isna().any():
        raise ValueError("Missing activity metadata while building test episodes.")
    grouped = merged.groupby("activity_id")["window_index"]
    return {
        int(activity_id): np.asarray(group.tolist(), dtype=np.int64)
        for activity_id, group in grouped
    }


def _sample_window_array(loader: Loader, index: int) -> np.ndarray:
    sample = loader.get_sample(index)
    if not sample:
        raise ValueError(f"Empty sample for window index {index}.")
    x_np = np.asarray(sample[0])
    if x_np.ndim == 2:
        return x_np
    if x_np.ndim == 3 and x_np.shape[0] == 1:
        return x_np[0]
    raise ValueError(f"Expected sample with shape (window, sensors), got {x_np.shape}.")


def _eligible_activities(
    indices_by_activity: dict[int, np.ndarray],
    k: int,
    min_query_per_class: int,
) -> list[int]:
    return [
        int(activity_id)
        for activity_id, indices in sorted(indices_by_activity.items())
        if len(indices) >= int(k) + int(min_query_per_class)
    ]


def _sample_test_episode(
    loader: Loader,
    indices_by_activity: dict[int, np.ndarray],
    activity_ids: Sequence[int],
    k: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    support_x: list[np.ndarray] = []
    support_y: list[int] = []
    query_x: list[np.ndarray] = []
    query_y: list[int] = []

    for activity_id in activity_ids:
        candidates = np.asarray(indices_by_activity[int(activity_id)], dtype=np.int64)
        perm = rng.permutation(candidates)
        support_indices = perm[: int(k)]
        query_indices = perm[int(k) :]
        for idx in support_indices.tolist():
            support_x.append(_sample_window_array(loader, int(idx)))
            support_y.append(int(activity_id))
        for idx in query_indices.tolist():
            query_x.append(_sample_window_array(loader, int(idx)))
            query_y.append(int(activity_id))

    support_perm = rng.permutation(len(support_x))
    query_perm = rng.permutation(len(query_x))
    x_support = torch.from_numpy(
        np.stack([support_x[i] for i in support_perm], axis=0)
    ).float()
    y_support = torch.from_numpy(
        np.asarray([support_y[i] for i in support_perm], dtype=np.int64)
    ).long()
    x_query = torch.from_numpy(
        np.stack([query_x[i] for i in query_perm], axis=0)
    ).float()
    y_query = torch.from_numpy(
        np.asarray([query_y[i] for i in query_perm], dtype=np.int64)
    ).long()

    return (
        x_support.unsqueeze(0).unsqueeze(2),
        y_support.unsqueeze(0),
        x_query.unsqueeze(0).unsqueeze(2),
        y_query.unsqueeze(0),
    )


@torch.no_grad()
def _predict_film(
    film_model: torch.nn.Module,
    set_encoder: torch.nn.Module,
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    x_query: torch.Tensor,
    device: torch.device,
    query_batch_size: int,
) -> torch.Tensor:
    c_subject = set_encoder(x_support.to(device), y_support.to(device))
    preds: list[torch.Tensor] = []
    flat_query = x_query.squeeze(0)
    for start in range(0, flat_query.size(0), int(query_batch_size)):
        chunk = flat_query[start : start + int(query_batch_size)].unsqueeze(0).to(device)
        logits = film_model.forward_episode(chunk, c_subject)
        preds.append(logits.squeeze(0).argmax(dim=-1).cpu())
    return torch.cat(preds, dim=0)


@torch.no_grad()
def _predict_base(
    base_model: TinierHAR,
    x_query: torch.Tensor,
    device: torch.device,
    query_batch_size: int,
) -> torch.Tensor:
    preds: list[torch.Tensor] = []
    flat_query = x_query.squeeze(0)
    for start in range(0, flat_query.size(0), int(query_batch_size)):
        chunk = flat_query[start : start + int(query_batch_size)].to(device)
        logits = base_model(chunk)
        preds.append(logits.argmax(dim=-1).cpu())
    return torch.cat(preds, dim=0)


def _mean_ci(values: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(max(1, arr.size)))
    return mean, std, ci95


def _plot_k_curve(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(
        summary_df["k"],
        summary_df["film_macro_f1_mean"],
        yerr=summary_df["film_macro_f1_ci95"],
        marker="o",
        linewidth=2,
        capsize=4,
        label="Subject-modulated TinierHAR",
    )
    ax.errorbar(
        summary_df["k"],
        summary_df["base_macro_f1_mean"],
        yerr=summary_df["base_macro_f1_ci95"],
        marker="s",
        linewidth=2,
        capsize=4,
        label="Base TinierHAR",
    )
    ax.set_xlabel("K shots per class")
    ax.set_ylabel("Macro F1")
    ax.set_title("Final LOSO Monte Carlo Evaluation")
    ax.set_xticks(summary_df["k"].tolist())
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)
    output_root = Path(config.output_root)
    film_stage_dir = output_root / config.film_stage_name
    film_summary_path = film_stage_dir / "summary.json"
    film_summary = _load_json(film_summary_path)
    film_train_config = dict(film_summary.get("config", {}))

    set_summary_path = output_root / "02_set_encoder_supcon" / "summary.json"
    set_config = dict(_load_json(set_summary_path).get("config", {}))

    dataset_id = WHARDatasetID(config.dataset_id)
    cfg = prepare_cfg(
        dataset_id=dataset_id,
        datasets_dir=Path(config.datasets_dir),
        selected_activities=config.selected_activities,
        window_overlap=config.window_overlap,
    )
    pre = PreProcessingPipeline(cfg)
    _raw_df, session_df, window_df = pre.run()
    shared_cfg = SharedConfig(
        dataset_id=config.dataset_id,
        datasets_dir=config.datasets_dir,
        selected_activities=config.selected_activities,
        window_overlap=config.window_overlap,
        subjects_per_group=config.subjects_per_group,
        base_train_subjects=config.base_train_subjects,
        meta_train_subjects=config.meta_train_subjects,
        val_subjects=config.val_subjects,
        test_subjects=config.test_subjects,
        seed=config.seed,
    )
    manifest_path = Path(
        film_summary.get(
            "splits_manifest_path",
            output_root / "shared_splits" / "loso_subject_folds.json",
        )
    )
    folds = _load_folds(manifest_path, session_df, window_df, shared_cfg)
    if config.max_folds is not None:
        folds = folds[: int(config.max_folds)]

    eval_dir = output_root / config.eval_stage_name / config.film_stage_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    pretrain_root = output_root / "01_pretrain_base"
    set_encoder_root = output_root / "02_set_encoder_supcon"

    all_trial_rows: list[dict[str, Any]] = []
    subject_k_rows: list[dict[str, Any]] = []
    skipped_folds: list[str] = []

    for fold in folds:
        if len(fold.test_subject_ids) != 1:
            raise ValueError(f"Expected one test subject in {fold.fold_id}.")
        test_subject_id = int(fold.test_subject_ids[0])
        base_ckpt = pretrain_root / fold.fold_id / "best_base_model.pt"
        set_ckpt = set_encoder_root / fold.fold_id / "best_set_encoder_supcon.pt"
        film_ckpt = film_stage_dir / fold.fold_id / "best_film_tinierhar.pt"
        missing = [p for p in (base_ckpt, set_ckpt, film_ckpt) if not p.exists()]
        if missing:
            if config.skip_missing_folds:
                skipped_folds.append(fold.fold_id)
                print(f"[{fold.fold_id}] skipping missing artifacts: {missing}")
                continue
            raise FileNotFoundError(f"[{fold.fold_id}] missing artifacts: {missing}")

        subject_indices = _indices_for_subject(session_df, window_df, test_subject_id)
        if not subject_indices:
            raise ValueError(f"No windows found for subject {test_subject_id}.")
        post = PostProcessingPipeline(cfg, pre, window_df, subject_indices)
        samples = post.run()
        loader = Loader(session_df, window_df, post.samples_dir, samples)
        x_np = np.asarray(loader.get_sample(subject_indices[0])[0])
        if x_np.ndim == 3 and x_np.shape[0] == 1:
            x_np = x_np[0]
        window_size = int(x_np.shape[0])
        num_channels = int(cfg.num_of_channels)
        num_classes = int(cfg.num_of_activities)

        base_model = _load_base_model(
            base_ckpt, num_channels, num_classes, window_size, config.device
        ).to(device)
        film_base_model = _load_base_model(
            base_ckpt, num_channels, num_classes, window_size, config.device
        )
        se_backbone = TinierHAR(
            num_channels=num_channels,
            num_classes=num_classes,
            window_size=window_size,
            backbone_config=DEFAULT_CONFIG.backbone,
        )
        set_encoder = _build_set_encoder(set_config, se_backbone, num_classes)
        se_payload = torch.load(set_ckpt, map_location=config.device, weights_only=False)
        set_encoder.load_state_dict(se_payload["set_encoder"])
        set_encoder.eval().to(device)
        for param in set_encoder.parameters():
            param.requires_grad = False

        film_model = _build_conditioned_model(
            film_train_config,
            film_base_model,
            int(getattr(set_encoder, "output_dim")),
        )
        film_payload = torch.load(film_ckpt, map_location=config.device, weights_only=False)
        film_model.load_state_dict(film_payload["film_model"])
        film_model.eval().to(device)

        indices_by_activity = _build_subject_activity_index(loader, subject_indices)
        max_k = max(int(k) for k in config.k_values)
        common_activity_ids = _eligible_activities(
            indices_by_activity, max_k, config.min_query_per_class
        )
        if config.require_all_k_activities and not common_activity_ids:
            raise ValueError(
                f"{fold.fold_id} subject {test_subject_id} has no activities with "
                f">= {max_k + config.min_query_per_class} windows."
            )

        for k in config.k_values:
            k = int(k)
            activity_ids = (
                common_activity_ids
                if config.require_all_k_activities
                else _eligible_activities(
                    indices_by_activity, k, config.min_query_per_class
                )
            )
            if not activity_ids:
                if config.skip_missing_folds:
                    continue
                raise ValueError(
                    f"{fold.fold_id} subject {test_subject_id} has no eligible "
                    f"activities for k={k}."
                )

            film_scores: list[float] = []
            base_scores: list[float] = []
            improvements: list[float] = []
            film_accs: list[float] = []
            base_accs: list[float] = []
            iterator = tqdm(
                range(int(config.episodes_per_k)),
                desc=f"{fold.fold_id} subject={test_subject_id} k={k}",
                leave=False,
            )
            for trial in iterator:
                rng = np.random.default_rng(
                    int(config.seed)
                    + 1_000_000 * int(test_subject_id)
                    + 10_000 * k
                    + int(trial)
                )
                x_support, y_support, x_query, y_query = _sample_test_episode(
                    loader, indices_by_activity, activity_ids, k, rng
                )
                y_true = y_query.reshape(-1).cpu().numpy()
                film_pred = _predict_film(
                    film_model,
                    set_encoder,
                    x_support,
                    y_support,
                    x_query,
                    device,
                    config.query_batch_size,
                ).numpy()
                base_pred = _predict_base(
                    base_model,
                    x_query,
                    device,
                    config.query_batch_size,
                ).numpy()
                film_f1 = float(
                    f1_score(
                        y_true,
                        film_pred,
                        labels=list(activity_ids),
                        average="macro",
                        zero_division=0,
                    )
                )
                base_f1 = float(
                    f1_score(
                        y_true,
                        base_pred,
                        labels=list(activity_ids),
                        average="macro",
                        zero_division=0,
                    )
                )
                film_acc = float(accuracy_score(y_true, film_pred))
                base_acc = float(accuracy_score(y_true, base_pred))
                improvement = float(film_f1 - base_f1)
                film_scores.append(film_f1)
                base_scores.append(base_f1)
                improvements.append(improvement)
                film_accs.append(film_acc)
                base_accs.append(base_acc)
                all_trial_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "test_subject_id": test_subject_id,
                        "k": k,
                        "trial": int(trial),
                        "num_activities": len(activity_ids),
                        "num_support": int(y_support.numel()),
                        "num_query": int(y_query.numel()),
                        "film_macro_f1": film_f1,
                        "base_macro_f1": base_f1,
                        "macro_f1_improvement": improvement,
                        "film_accuracy": film_acc,
                        "base_accuracy": base_acc,
                    }
                )

            film_mean, film_std, film_ci95 = _mean_ci(film_scores)
            base_mean, base_std, base_ci95 = _mean_ci(base_scores)
            imp_mean, imp_std, imp_ci95 = _mean_ci(improvements)
            film_acc_mean, film_acc_std, film_acc_ci95 = _mean_ci(film_accs)
            base_acc_mean, base_acc_std, base_acc_ci95 = _mean_ci(base_accs)
            subject_k_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "test_subject_id": test_subject_id,
                    "k": k,
                    "episodes": int(config.episodes_per_k),
                    "activity_ids": [int(a) for a in activity_ids],
                    "film_macro_f1_mean": film_mean,
                    "film_macro_f1_std": film_std,
                    "film_macro_f1_ci95": film_ci95,
                    "base_macro_f1_mean": base_mean,
                    "base_macro_f1_std": base_std,
                    "base_macro_f1_ci95": base_ci95,
                    "macro_f1_improvement_mean": imp_mean,
                    "macro_f1_improvement_std": imp_std,
                    "macro_f1_improvement_ci95": imp_ci95,
                    "film_accuracy_mean": film_acc_mean,
                    "film_accuracy_std": film_acc_std,
                    "film_accuracy_ci95": film_acc_ci95,
                    "base_accuracy_mean": base_acc_mean,
                    "base_accuracy_std": base_acc_std,
                    "base_accuracy_ci95": base_acc_ci95,
                }
            )

    trial_df = pd.DataFrame(all_trial_rows)
    subject_k_df = pd.DataFrame(subject_k_rows)
    if subject_k_df.empty:
        raise RuntimeError("No evaluation rows were produced.")

    overall_rows: list[dict[str, Any]] = []
    for k, group in subject_k_df.groupby("k", sort=True):
        row: dict[str, Any] = {"k": int(k), "num_subjects": int(group.shape[0])}
        for metric in (
            "film_macro_f1_mean",
            "base_macro_f1_mean",
            "macro_f1_improvement_mean",
            "film_accuracy_mean",
            "base_accuracy_mean",
        ):
            mean, std, ci95 = _mean_ci(group[metric].astype(float).tolist())
            row[metric] = mean
            row[metric.replace("_mean", "_subject_std")] = std
            row[metric.replace("_mean", "_subject_ci95")] = ci95
        overall_rows.append(row)
    overall_df = pd.DataFrame(overall_rows).sort_values("k")

    trial_csv = eval_dir / "trial_results.csv"
    subject_csv = eval_dir / "subject_by_k_results.csv"
    overall_csv = eval_dir / "overall_by_k_results.csv"
    trial_df.to_csv(trial_csv, index=False)
    subject_k_df.to_csv(subject_csv, index=False)
    overall_df.to_csv(overall_csv, index=False)
    _plot_k_curve(overall_df, eval_dir / "k_shot_curve.png")

    summary = {
        "config": asdict(config),
        "film_stage_dir": str(film_stage_dir),
        "splits_manifest_path": str(manifest_path),
        "skipped_folds": skipped_folds,
        "trial_results_csv": str(trial_csv),
        "subject_by_k_results_csv": str(subject_csv),
        "overall_by_k_results_csv": str(overall_csv),
        "plot_path": str(eval_dir / "k_shot_curve.png"),
        "overall_by_k": overall_df.to_dict(orient="records"),
        "num_trial_rows": int(trial_df.shape[0]),
        "num_subject_k_rows": int(subject_k_df.shape[0]),
    }
    (eval_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

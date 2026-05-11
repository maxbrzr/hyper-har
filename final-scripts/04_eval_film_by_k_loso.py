from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from common import (
    DEFAULT_TRAIN_MAX_K_PER_CLASS,
    DEFAULT_TRAIN_MIN_K_PER_CLASS,
    ROOT,
    SharedConfig,
    build_or_load_loso_folds,
    k_choices_from_range,
    prepare_cfg,
    set_seed,
)
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from whar_datasets import (
    Loader,
    PostProcessingPipeline,
    PreProcessingPipeline,
    WHARDatasetID,
)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hyper_har.backbone.film_tinierhar import FiLMTinierHAR
from hyper_har.backbone.tinierhar import TinierHAR
from hyper_har.config import DEFAULT_CONFIG
from hyper_har.training.conditioned_meta_trainer import (
    ConditionedMetaTrainerConfig,
    SubjectConditionedMetaTrainer,
)


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SET_ENCODER_ATTENTION_MODULE = _load_module_from_path(
    "final_phase4_film_eval_attention_set_encoder",
    SRC / "hyper_har" / "set-encoder" / "attention.py",
)
SET_ENCODER_SIMPLE_MODULE = _load_module_from_path(
    "final_phase4_film_eval_simple_set_encoder",
    SRC / "hyper_har" / "set-encoder" / "simple.py",
)
AttentionSetEncoder = SET_ENCODER_ATTENTION_MODULE.AttentionSetEncoder
PrototypicalSetEncoder = SET_ENCODER_SIMPLE_MODULE.PrototypicalSetEncoder


@dataclass(frozen=True)
class SubjectKMetrics:
    fold_id: str
    subject_id: int
    k: int
    episodes: int
    query_per_class: int
    activity_count: int
    base_loss: float
    film_loss: float
    base_macro_f1: float
    film_macro_f1: float
    macro_f1_improvement: float
    episode_base_macro_f1_mean: float
    episode_base_macro_f1_std: float
    episode_film_macro_f1_mean: float
    episode_film_macro_f1_std: float
    episode_improvement_mean: float
    episode_improvement_std: float
    episode_improvement_sem: float
    episode_improvement_ci95: float


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)}.")
    return payload


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    return default if value is None else int(value)


def _as_float(value: Any, default: float) -> float:
    return default if value is None else float(value)


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def _resolve_stage_dir(output_root: Path, film_stage: str | None) -> Path:
    if film_stage is not None:
        as_path = Path(film_stage)
        if as_path.exists() and as_path.is_dir():
            return as_path
        candidate = output_root / film_stage
        if candidate.exists() and candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Could not resolve FiLM stage dir: {film_stage}")

    default_stage = output_root / "03_film_meta"
    guarded_stage = output_root / "03_film_meta_guarded"
    if default_stage.exists():
        return default_stage
    if guarded_stage.exists():
        return guarded_stage
    raise FileNotFoundError(
        f"Could not find {default_stage} or {guarded_stage}. "
        "Pass --film-stage to select a trained FiLM artifact directory."
    )


def _build_set_encoder(
    encoder: str,
    backbone: TinierHAR,
    num_classes: int,
    backbone_train_mode: str,
    force_conv_bn_eval: bool,
) -> torch.nn.Module:
    se_cfg = DEFAULT_CONFIG.set_encoder
    if hasattr(se_cfg, "__dataclass_fields__"):
        from dataclasses import replace

        se_cfg = replace(se_cfg, include_global_context=False)

    encoder_norm = encoder.strip().lower()
    if encoder_norm == "attention":
        return AttentionSetEncoder(
            backbone=backbone,
            num_classes=num_classes,
            backbone_train_mode=backbone_train_mode,
            force_conv_bn_eval=force_conv_bn_eval,
            set_encoder_config=se_cfg,
        )
    if encoder_norm in {"prototypical", "simple"}:
        return PrototypicalSetEncoder(
            backbone=backbone,
            num_classes=num_classes,
            backbone_train_mode=backbone_train_mode,
            force_conv_bn_eval=force_conv_bn_eval,
            set_encoder_config=se_cfg,
        )
    raise ValueError(f"Unsupported encoder: {encoder!r}")


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


def _indices_for_subject(
    session_df: Any,
    window_df: Any,
    subject_id: int,
) -> list[int]:
    session_ids = set(
        session_df.loc[
            session_df["subject_id"].astype(int) == int(subject_id),
            "session_id",
        ].tolist()
    )
    indices = window_df.index[window_df["session_id"].isin(session_ids)].tolist()
    return [int(idx) for idx in indices]


def _choose_activity_ids(
    loader: Loader,
    indices: Sequence[int],
    needed_per_subject_activity: int,
) -> list[int]:
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    subset["window_index"] = subset.index.astype(int)
    session_meta = loader.session_df[
        ["session_id", "subject_id", "activity_id"]
    ].drop_duplicates("session_id")
    merged = subset.merge(session_meta, on="session_id", how="left")
    if merged["subject_id"].isna().any() or merged["activity_id"].isna().any():
        raise ValueError("Missing subject/activity metadata while choosing activities.")

    counts = (
        merged.groupby(["subject_id", "activity_id"])["window_index"]
        .count()
        .reset_index(name="count")
    )
    subject_ids = sorted(int(x) for x in counts["subject_id"].unique().tolist())
    if len(subject_ids) != 1:
        raise ValueError(f"Expected one subject, found {subject_ids}.")
    selected = [
        int(row.activity_id)
        for row in counts.itertuples(index=False)
        if int(row.count) >= int(needed_per_subject_activity)
    ]
    if not selected:
        raise ValueError(
            "No activity ids satisfy per-subject episodic requirements: "
            f"needed_per_subject_activity={needed_per_subject_activity}"
        )
    return sorted(selected)


def _mean_std_sem(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean_v = float(arr.mean())
    std_v = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sem_v = float(std_v / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean_v, std_v, sem_v, float(1.96 * sem_v)


@torch.no_grad()
def _evaluate_subject_for_k(
    trainer: SubjectConditionedMetaTrainer,
    fold_id: str,
    subject_id: int,
    k: int,
    episodes: int,
    query_per_class: int,
) -> SubjectKMetrics:
    trainer.baseline_model.eval()
    trainer.set_encoder.eval()
    trainer.conditioned_model.eval()

    base_losses: list[float] = []
    film_losses: list[float] = []
    all_targets: list[torch.Tensor] = []
    base_preds_all: list[torch.Tensor] = []
    film_preds_all: list[torch.Tensor] = []
    episode_base_f1s: list[float] = []
    episode_film_f1s: list[float] = []
    episode_improvements: list[float] = []

    progress = tqdm(
        range(int(episodes)),
        desc=f"{fold_id} subject {subject_id} K={k}",
        leave=False,
    )
    for _ in progress:
        x_support, y_support, x_query, y_query, _subjects = trainer._sample_episode(
            support_per_class=int(k)
        )
        x_support = x_support.to(trainer.device)
        y_support = y_support.to(trainer.device)
        x_query = x_query.to(trainer.device)
        y_query = y_query.to(trainer.device)

        targets_flat = y_query.reshape(-1)
        base_logits = trainer.baseline_model(
            x_query.reshape(-1, *x_query.shape[2:])
        ).reshape(x_query.size(0), x_query.size(1), -1)
        base_logits_flat = base_logits.reshape(-1, base_logits.size(-1))
        base_loss = F.cross_entropy(
            base_logits_flat, targets_flat, weight=trainer.class_weights
        )

        c_subject = trainer.set_encoder(x_support, y_support)
        film_logits = trainer.conditioned_model.forward_episode(x_query, c_subject)
        film_logits_flat = film_logits.reshape(-1, film_logits.size(-1))
        film_loss = F.cross_entropy(
            film_logits_flat, targets_flat, weight=trainer.class_weights
        )

        targets_cpu = targets_flat.cpu()
        base_preds = base_logits.argmax(dim=-1).reshape(-1).cpu()
        film_preds = film_logits.argmax(dim=-1).reshape(-1).cpu()
        episode_base_f1 = f1_score(
            targets_cpu.numpy(),
            base_preds.numpy(),
            average="macro",
            zero_division=0,
        )
        episode_film_f1 = f1_score(
            targets_cpu.numpy(),
            film_preds.numpy(),
            average="macro",
            zero_division=0,
        )

        base_losses.append(float(base_loss.item()))
        film_losses.append(float(film_loss.item()))
        all_targets.append(targets_cpu)
        base_preds_all.append(base_preds)
        film_preds_all.append(film_preds)
        episode_base_f1s.append(float(episode_base_f1))
        episode_film_f1s.append(float(episode_film_f1))
        episode_improvements.append(float(episode_film_f1 - episode_base_f1))

    targets_t = torch.cat(all_targets)
    base_preds_t = torch.cat(base_preds_all)
    film_preds_t = torch.cat(film_preds_all)
    base_macro_f1 = f1_score(
        targets_t.numpy(),
        base_preds_t.numpy(),
        average="macro",
        zero_division=0,
    )
    film_macro_f1 = f1_score(
        targets_t.numpy(),
        film_preds_t.numpy(),
        average="macro",
        zero_division=0,
    )
    base_ep_mean, base_ep_std, _, _ = _mean_std_sem(episode_base_f1s)
    film_ep_mean, film_ep_std, _, _ = _mean_std_sem(episode_film_f1s)
    imp_ep_mean, imp_ep_std, imp_ep_sem, imp_ep_ci95 = _mean_std_sem(
        episode_improvements
    )

    return SubjectKMetrics(
        fold_id=fold_id,
        subject_id=int(subject_id),
        k=int(k),
        episodes=int(episodes),
        query_per_class=int(query_per_class),
        activity_count=len(trainer.activity_ids),
        base_loss=sum(base_losses) / max(1, len(base_losses)),
        film_loss=sum(film_losses) / max(1, len(film_losses)),
        base_macro_f1=float(base_macro_f1),
        film_macro_f1=float(film_macro_f1),
        macro_f1_improvement=float(film_macro_f1 - base_macro_f1),
        episode_base_macro_f1_mean=base_ep_mean,
        episode_base_macro_f1_std=base_ep_std,
        episode_film_macro_f1_mean=film_ep_mean,
        episode_film_macro_f1_std=film_ep_std,
        episode_improvement_mean=imp_ep_mean,
        episode_improvement_std=imp_ep_std,
        episode_improvement_sem=imp_ep_sem,
        episode_improvement_ci95=imp_ep_ci95,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize_by_k(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for k in sorted({int(row["k"]) for row in rows}):
        rows_k = [row for row in rows if int(row["k"]) == k]
        improvements = [float(row["macro_f1_improvement"]) for row in rows_k]
        mean_imp, std_imp, sem_imp, ci95_imp = _mean_std_sem(improvements)
        summary.append(
            {
                "k": int(k),
                "num_subjects": len(rows_k),
                "mean_base_macro_f1": float(
                    np.mean([float(row["base_macro_f1"]) for row in rows_k])
                ),
                "mean_film_macro_f1": float(
                    np.mean([float(row["film_macro_f1"]) for row in rows_k])
                ),
                "mean_macro_f1_improvement": mean_imp,
                "std_macro_f1_improvement": std_imp,
                "sem_macro_f1_improvement": sem_imp,
                "ci95_macro_f1_improvement": ci95_imp,
                "mean_episode_improvement": float(
                    np.mean([float(row["episode_improvement_mean"]) for row in rows_k])
                ),
                "mean_episode_improvement_sem": float(
                    np.mean([float(row["episode_improvement_sem"]) for row in rows_k])
                ),
            }
        )
    return summary


def _plot_k_improvement_lines(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))

    subject_ids = sorted({int(row["subject_id"]) for row in rows})
    for subject_id in subject_ids:
        subject_rows = sorted(
            [row for row in rows if int(row["subject_id"]) == subject_id],
            key=lambda row: int(row["k"]),
        )
        ax.plot(
            [int(row["k"]) for row in subject_rows],
            [float(row["macro_f1_improvement"]) for row in subject_rows],
            marker="o",
            linewidth=1.2,
            alpha=0.45,
            label=f"subject {subject_id}",
        )

    summary_sorted = sorted(summary_rows, key=lambda row: int(row["k"]))
    k_values = [int(row["k"]) for row in summary_sorted]
    means = [float(row["mean_macro_f1_improvement"]) for row in summary_sorted]
    ci95 = [float(row["ci95_macro_f1_improvement"]) for row in summary_sorted]
    ax.plot(
        k_values,
        means,
        marker="o",
        linewidth=3.0,
        color="black",
        label="mean",
    )
    lower = np.asarray(means) - np.asarray(ci95)
    upper = np.asarray(means) + np.asarray(ci95)
    ax.fill_between(k_values, lower, upper, color="black", alpha=0.12, label="mean 95% CI")

    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.set_xlabel("Support windows per class (K)")
    ax.set_ylabel("Macro F1 improvement (FiLM - base)")
    ax.set_title("FiLM Test-Subject Improvement by Support K")
    ax.grid(alpha=0.25)
    ax.set_xticks(k_values)
    if len(subject_ids) <= 12:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained final-pipeline FiLM stage on each test subject "
            "individually across K support windows per class, then plot K vs improvement."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "final_pipeline",
        help="Final pipeline artifact root.",
    )
    parser.add_argument(
        "--film-stage",
        type=str,
        default=None,
        help=(
            "FiLM stage directory or name. Defaults to 03_film_meta if present, "
            "otherwise 03_film_meta_guarded."
        ),
    )
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument(
        "--k-values",
        type=str,
        default=None,
        help="Comma-separated K values. Defaults to the FiLM config eval K range.",
    )
    parser.add_argument(
        "--query-per-class",
        type=int,
        default=None,
        help="Query windows per class. Defaults to FiLM config eval_query_per_class.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Optional comma-separated subject ids to evaluate.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stage_dir = _resolve_stage_dir(args.output_root, args.film_stage)
    summary_path = stage_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing FiLM stage summary: {summary_path}")
    train_summary = _load_json(summary_path)
    train_config = dict(train_summary.get("config", {}))

    seed = args.seed if args.seed is not None else _as_int(train_config.get("seed"), 0)
    set_seed(seed)
    device = args.device or _as_str_device(train_config.get("device"))
    output_dir = args.output_dir or (stage_dir / "evaluation_by_k")
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    k_values = _parse_int_list(args.k_values)
    if k_values is None:
        k_values = list(
            k_choices_from_range(
                _as_int(train_config.get("eval_min_k_per_class"), DEFAULT_TRAIN_MIN_K_PER_CLASS),
                _as_int(train_config.get("eval_max_k_per_class"), DEFAULT_TRAIN_MAX_K_PER_CLASS),
            )
        )
    query_per_class = (
        int(args.query_per_class)
        if args.query_per_class is not None
        else _as_int(train_config.get("eval_query_per_class"), 16)
    )
    selected_subjects = _parse_int_list(args.subjects)
    selected_subject_set = set(selected_subjects) if selected_subjects else None

    dataset_id = WHARDatasetID(str(train_config.get("dataset_id", WHARDatasetID.WEAR.value)))
    cfg = prepare_cfg(
        dataset_id=dataset_id,
        datasets_dir=Path(str(train_config.get("datasets_dir", ROOT / "datasets"))),
        selected_activities=train_config.get("selected_activities"),
        window_overlap=_as_float(train_config.get("window_overlap"), 0.0),
    )
    pre = PreProcessingPipeline(cfg)
    _raw_df, session_df, window_df = pre.run()
    shared_cfg = SharedConfig(
        dataset_id=str(train_config.get("dataset_id", WHARDatasetID.WEAR.value)),
        datasets_dir=str(train_config.get("datasets_dir", ROOT / "datasets")),
        selected_activities=train_config.get("selected_activities"),
        window_overlap=_as_float(train_config.get("window_overlap"), 0.0),
        subjects_per_group=_as_int(train_config.get("subjects_per_group"), 6),
        seed=_as_int(train_config.get("seed"), 0),
    )
    manifest_path = args.output_root / "shared_splits" / "group4_subject_folds.json"
    folds = build_or_load_loso_folds(session_df, window_df, shared_cfg, manifest_path)
    max_folds = train_config.get("max_folds")
    if max_folds is not None:
        folds = folds[: int(max_folds)]

    encoder = str(train_config.get("encoder", "attention"))
    set_encoder_mode = str(train_config.get("set_encoder_backbone_train_mode", "freeze_all"))
    force_conv_bn_eval = _as_bool(train_config.get("force_conv_bn_eval"), True)
    film_hidden_dim = _as_int(train_config.get("film_hidden_dim"), 128)
    film_dropout = _as_float(train_config.get("film_dropout"), 0.0)
    film_use_explosion_guard = _as_bool(
        train_config.get("film_use_explosion_guard"), False
    )
    film_gamma_bound = _as_float(train_config.get("film_gamma_bound"), 0.5)
    film_beta_bound = _as_float(train_config.get("film_beta_bound"), 1.0)

    pretrain_root = args.output_root / "01_pretrain_base"
    set_encoder_root = args.output_root / "02_set_encoder_supcon"
    max_k = max(k_values)
    all_rows: list[dict[str, Any]] = []

    for fold_idx, fold in enumerate(folds):
        fold_ckpt = stage_dir / fold.fold_id / "best_film_tinierhar.pt"
        base_ckpt = pretrain_root / fold.fold_id / "best_base_model.pt"
        set_ckpt = set_encoder_root / fold.fold_id / "best_set_encoder_supcon.pt"
        if not fold_ckpt.exists():
            print(f"Skipping {fold.fold_id}: missing {fold_ckpt}")
            continue
        if not base_ckpt.exists():
            raise FileNotFoundError(f"Missing base checkpoint: {base_ckpt}")
        if not set_ckpt.exists():
            raise FileNotFoundError(f"Missing set encoder checkpoint: {set_ckpt}")

        for subject_id in fold.test_subject_ids:
            subject_id = int(subject_id)
            if selected_subject_set is not None and subject_id not in selected_subject_set:
                continue

            subject_indices = _indices_for_subject(session_df, window_df, subject_id)
            if not subject_indices:
                print(f"Skipping {fold.fold_id} subject {subject_id}: no windows found")
                continue

            post = PostProcessingPipeline(cfg, pre, window_df, subject_indices)
            samples = post.run()
            loader = Loader(session_df, window_df, post.samples_dir, samples)
            x_np = np.asarray(loader.get_sample(subject_indices[0])[0])
            if x_np.ndim == 3 and x_np.shape[0] == 1:
                x_np = x_np[0]
            window_size = int(x_np.shape[0])
            num_channels = int(cfg.num_of_channels)
            num_classes = int(cfg.num_of_activities)

            baseline_model = _load_base_model(
                base_ckpt, num_channels, num_classes, window_size, device
            )
            film_base_model = _load_base_model(
                base_ckpt, num_channels, num_classes, window_size, device
            )
            se_backbone = TinierHAR(
                num_channels=num_channels,
                num_classes=num_classes,
                window_size=window_size,
                backbone_config=DEFAULT_CONFIG.backbone,
            )
            set_encoder = _build_set_encoder(
                encoder=encoder,
                backbone=se_backbone,
                num_classes=num_classes,
                backbone_train_mode=set_encoder_mode,
                force_conv_bn_eval=force_conv_bn_eval,
            )
            se_payload = torch.load(set_ckpt, map_location=device, weights_only=False)
            set_encoder.load_state_dict(se_payload["set_encoder"])
            for param in set_encoder.parameters():
                param.requires_grad = False
            set_encoder.eval()

            film_model = FiLMTinierHAR(
                base_model=film_base_model,
                subject_embedding_dim=int(getattr(set_encoder, "output_dim")),
                film_hidden_dim=film_hidden_dim,
                film_dropout=film_dropout,
                film_use_explosion_guard=film_use_explosion_guard,
                film_gamma_bound=film_gamma_bound,
                film_beta_bound=film_beta_bound,
            )
            film_payload = torch.load(fold_ckpt, map_location=device, weights_only=False)
            film_model.load_state_dict(film_payload["film_model"])
            film_model.eval()

            for k in k_values:
                needed = int(k) + int(query_per_class)
                try:
                    activity_ids = _choose_activity_ids(
                        loader,
                        subject_indices,
                        needed_per_subject_activity=needed,
                    )
                except ValueError as exc:
                    print(
                        f"Skipping {fold.fold_id} subject {subject_id} K={k}: {exc}"
                    )
                    continue

                eval_cfg = ConditionedMetaTrainerConfig(
                    batch_subjects=1,
                    support_per_class=int(k),
                    support_per_class_choices=None,
                    query_per_class=int(query_per_class),
                    seed=50_000 + (fold_idx * 1_000) + (subject_id * 10) + int(k),
                    device=device,
                )
                trainer = SubjectConditionedMetaTrainer(
                    conditioned_model=film_model,
                    set_encoder=set_encoder,
                    baseline_model=baseline_model,
                    loader=loader,
                    num_classes=num_classes,
                    config=eval_cfg,
                    optimizer=torch.optim.AdamW(
                        [p for p in film_model.parameters() if p.requires_grad],
                        lr=1e-4,
                    ),
                    class_weights=None,
                    indices=subject_indices,
                    activity_ids=activity_ids,
                    freeze_set_encoder=True,
                )
                metrics = _evaluate_subject_for_k(
                    trainer=trainer,
                    fold_id=fold.fold_id,
                    subject_id=subject_id,
                    k=int(k),
                    episodes=int(args.episodes),
                    query_per_class=int(query_per_class),
                )
                row = asdict(metrics)
                row["film_checkpoint"] = str(fold_ckpt)
                row["base_checkpoint"] = str(base_ckpt)
                row["set_encoder_checkpoint"] = str(set_ckpt)
                all_rows.append(row)
                print(
                    f"{fold.fold_id} subject={subject_id} K={k}: "
                    f"base={metrics.base_macro_f1:.4f} "
                    f"film={metrics.film_macro_f1:.4f} "
                    f"improvement={metrics.macro_f1_improvement:+.4f}"
                )

    if not all_rows:
        raise RuntimeError("No evaluation rows produced.")

    summary_rows = _summarize_by_k(all_rows)
    _write_csv(output_dir / "per_subject_by_k.csv", all_rows)
    _write_csv(output_dir / "summary_by_k.csv", summary_rows)
    _plot_k_improvement_lines(
        all_rows,
        summary_rows,
        plot_dir / "k_improvement_lineplot.png",
    )

    payload = {
        "film_stage_dir": str(stage_dir),
        "output_dir": str(output_dir),
        "episodes": int(args.episodes),
        "k_values": [int(k) for k in k_values],
        "query_per_class": int(query_per_class),
        "device": device,
        "train_config": train_config,
        "summary_by_k": summary_rows,
        "per_subject_by_k": all_rows,
    }
    (output_dir / "evaluation_by_k.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("\n=== FiLM by-K evaluation finished ===")
    print(f"Saved per-subject CSV: {output_dir / 'per_subject_by_k.csv'}")
    print(f"Saved K summary CSV: {output_dir / 'summary_by_k.csv'}")
    print(f"Saved JSON: {output_dir / 'evaluation_by_k.json'}")
    print(f"Saved lineplot: {plot_dir / 'k_improvement_lineplot.png'}")


def _as_str_device(value: Any) -> str:
    if value is None:
        return (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    return str(value)


if __name__ == "__main__":
    main()

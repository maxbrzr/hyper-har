from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
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
    build_loader,
    build_or_load_loso_folds,
    prepare_cfg,
    sample_window_array,
    set_seed,
)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm.auto import tqdm
from whar_datasets import PreProcessingPipeline, WHARDatasetID

TRAIN_SCRIPT = Path(__file__).resolve().parent / "02_train_blackbox_meta_loso.py"


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRAIN_MODULE = _load_module_from_path("meta_blackbox_train_module", TRAIN_SCRIPT)


@dataclass(frozen=True)
class Config:
    dataset_id: str = WHARDatasetID.WEAR.value
    datasets_dir: str = str(ROOT / "datasets")
    selected_activities: list[str] | None = None
    window_overlap: float = 0.0
    val_subjects: int = 6
    test_subjects: int = 1
    seed: int = 0

    k_values: tuple[int, ...] = (1, 4, 8, 16, 32)
    episodes_per_k: int = 100
    min_query_per_class: int = 1
    query_batch_size: int = 512
    require_all_k_activities: bool = True
    skip_missing_folds: bool = False

    output_root: str = str(ROOT / "artifacts" / "blackbox_meta_loso")
    meta_stage_name: str = "02_blackbox_meta"
    eval_stage_name: str = "03_blackbox_meta_eval"
    max_folds: int | None = None
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


RUN_CONFIG = Config()


def _indices_for_subject(session_df: pd.DataFrame, window_df: pd.DataFrame, subject_id: int) -> list[int]:
    meta = window_df[["session_id"]].copy()
    meta["window_index"] = meta.index.astype(int)
    session_meta = session_df[["session_id", "subject_id"]].drop_duplicates("session_id")
    merged = meta.merge(session_meta, on="session_id", how="left")
    out = merged.loc[
        merged["subject_id"].astype("Int64") == int(subject_id),
        "window_index",
    ]
    return sorted(int(x) for x in out.tolist())


def _build_subject_activity_index(loader: Any, indices: Sequence[int]) -> dict[int, np.ndarray]:
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    subset["window_index"] = subset.index.astype(int)
    session_meta = loader.session_df[["session_id", "activity_id"]].drop_duplicates("session_id")
    merged = subset.merge(session_meta, on="session_id", how="left")
    if merged["activity_id"].isna().any():
        raise ValueError("Missing activity metadata while building evaluation episodes.")
    grouped = merged.groupby("activity_id")["window_index"]
    return {
        int(activity_id): np.asarray(group.tolist(), dtype=np.int64)
        for activity_id, group in grouped
    }


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
    loader: Any,
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
            support_x.append(sample_window_array(loader, int(idx)))
            support_y.append(int(activity_id))
        for idx in query_indices.tolist():
            query_x.append(sample_window_array(loader, int(idx)))
            query_y.append(int(activity_id))
    support_perm = rng.permutation(len(support_x))
    query_perm = rng.permutation(len(query_x))
    x_support = torch.from_numpy(np.stack([support_x[i] for i in support_perm], axis=0)).float()
    y_support = torch.from_numpy(np.asarray([support_y[i] for i in support_perm], dtype=np.int64)).long()
    x_query = torch.from_numpy(np.stack([query_x[i] for i in query_perm], axis=0)).float()
    y_query = torch.from_numpy(np.asarray([query_y[i] for i in query_perm], dtype=np.int64)).long()
    return (
        x_support.unsqueeze(0).unsqueeze(2),
        y_support.unsqueeze(0),
        x_query.unsqueeze(0).unsqueeze(2),
        y_query.unsqueeze(0),
    )


@torch.no_grad()
def _predict_conditioned(
    model: torch.nn.Module,
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
        logits = model.forward_episode(chunk, c_subject)
        preds.append(logits.squeeze(0).argmax(dim=-1).cpu())
    return torch.cat(preds, dim=0)


@torch.no_grad()
def _predict_unconditioned(
    model: torch.nn.Module,
    x_query: torch.Tensor,
    device: torch.device,
    query_batch_size: int,
) -> torch.Tensor:
    preds: list[torch.Tensor] = []
    flat_query = x_query.squeeze(0)
    for start in range(0, flat_query.size(0), int(query_batch_size)):
        chunk = flat_query[start : start + int(query_batch_size)].unsqueeze(0).to(device)
        logits = model.forward_unconditioned_episode(chunk)
        preds.append(logits.squeeze(0).argmax(dim=-1).cpu())
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
        summary_df["conditioned_macro_f1_mean"],
        yerr=summary_df["conditioned_macro_f1_subject_ci95"],
        marker="o",
        linewidth=2,
        capsize=4,
        label="Conditioned TinierHAR",
    )
    ax.errorbar(
        summary_df["k"],
        summary_df["unconditioned_macro_f1_mean"],
        yerr=summary_df["unconditioned_macro_f1_subject_ci95"],
        marker="s",
        linewidth=2,
        capsize=4,
        label="Same base, no support conditioning",
    )
    ax.set_xlabel("K shots per class")
    ax.set_ylabel("Macro F1")
    ax.set_title("Black-Box Meta-Learning LOSO Evaluation")
    ax.set_xticks(summary_df["k"].tolist())
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_confusion_matrix(
    matrix: np.ndarray,
    labels: Sequence[int],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=[str(x) for x in labels],
        yticklabels=[str(x) for x in labels],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)
    output_root = Path(config.output_root)
    meta_stage_dir = output_root / config.meta_stage_name
    meta_summary_path = meta_stage_dir / "summary.json"
    if not meta_summary_path.exists():
        raise FileNotFoundError(f"Run black-box meta-training first: {meta_summary_path}")
    meta_summary = json.loads(meta_summary_path.read_text(encoding="utf-8"))
    train_config = dict(meta_summary.get("config", {}))

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
        val_subjects=config.val_subjects,
        test_subjects=config.test_subjects,
        seed=config.seed,
    )
    manifest_path = Path(
        meta_summary.get(
            "splits_manifest_path",
            output_root / "shared_splits" / "loso_subject_folds.json",
        )
    )
    folds = build_or_load_loso_folds(session_df, window_df, shared_cfg, manifest_path)
    if config.max_folds is not None:
        folds = folds[: int(config.max_folds)]

    eval_dir = output_root / config.eval_stage_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    all_trial_rows: list[dict[str, Any]] = []
    subject_k_rows: list[dict[str, Any]] = []
    confusion_accumulator: dict[int, dict[str, np.ndarray]] = {}
    skipped_folds: list[str] = []

    for fold in folds:
        if len(fold.test_subject_ids) != 1:
            raise ValueError(f"Expected one test subject in {fold.fold_id}.")
        test_subject_id = int(fold.test_subject_ids[0])
        ckpt_path = meta_stage_dir / fold.fold_id / "best_blackbox_meta.pt"
        if not ckpt_path.exists():
            if config.skip_missing_folds:
                skipped_folds.append(fold.fold_id)
                print(f"[{fold.fold_id}] skipping missing checkpoint: {ckpt_path}")
                continue
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        subject_indices = _indices_for_subject(session_df, window_df, test_subject_id)
        if not subject_indices:
            raise ValueError(f"No windows found for subject {test_subject_id}.")
        loader = build_loader(cfg, session_df, pre, window_df, subject_indices)
        window_size = sample_window_array(loader, subject_indices[0]).shape[0]
        num_channels = int(cfg.num_of_channels)
        num_classes = int(cfg.num_of_activities)

        checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)
        set_encoder_config = dict(checkpoint.get("set_encoder_config", {}))
        set_encoder = TRAIN_MODULE._build_set_encoder(
            set_encoder_config,
            num_channels,
            num_classes,
            int(window_size),
        )
        set_encoder.load_state_dict(checkpoint["set_encoder"])
        set_encoder.to(device).eval()
        for param in set_encoder.parameters():
            param.requires_grad = False

        base_model = TRAIN_MODULE.TinierHAR(
            num_channels=num_channels,
            num_classes=num_classes,
            window_size=int(window_size),
            backbone_config=TRAIN_MODULE.DEFAULT_CONFIG.backbone,
        )
        model = TRAIN_MODULE.CoTrainedCBNAttentionTinierHAR(
            base_model=base_model,
            subject_embedding_dim=int(getattr(set_encoder, "output_dim")),
            modulator_hidden_dim=int(train_config.get("modulator_hidden_dim", 192)),
            modulator_dropout=float(train_config.get("modulator_dropout", 0.1)),
            modulator_use_tanh_gating=bool(train_config.get("modulator_use_tanh_gating", False)),
            modulator_gamma_bound=float(train_config.get("modulator_gamma_bound", 0.5)),
            modulator_beta_bound=float(train_config.get("modulator_beta_bound", 1.0)),
            pointwise_block_start=int(train_config.get("pointwise_block_start", 0)),
            pre_classifier_dropout=float(train_config.get("pre_classifier_dropout", 0.3)),
        )
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()

        indices_by_activity = _build_subject_activity_index(loader, subject_indices)
        max_k = max(int(k) for k in config.k_values)
        common_activity_ids = _eligible_activities(
            indices_by_activity,
            max_k,
            config.min_query_per_class,
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
                else _eligible_activities(indices_by_activity, k, config.min_query_per_class)
            )
            if not activity_ids:
                if config.skip_missing_folds:
                    continue
                raise ValueError(f"{fold.fold_id} has no eligible activities for k={k}.")

            cond_f1s: list[float] = []
            uncond_f1s: list[float] = []
            weighted_f1s: list[float] = []
            gains: list[float] = []
            cond_accs: list[float] = []
            uncond_accs: list[float] = []
            confusion_accumulator.setdefault(
                k,
                {
                    "conditioned": np.zeros((num_classes, num_classes), dtype=np.int64),
                    "unconditioned": np.zeros((num_classes, num_classes), dtype=np.int64),
                },
            )
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
                    loader,
                    indices_by_activity,
                    activity_ids,
                    k,
                    rng,
                )
                y_true = y_query.reshape(-1).cpu().numpy()
                cond_pred = _predict_conditioned(
                    model,
                    set_encoder,
                    x_support,
                    y_support,
                    x_query,
                    device,
                    config.query_batch_size,
                ).numpy()
                uncond_pred = _predict_unconditioned(
                    model,
                    x_query,
                    device,
                    config.query_batch_size,
                ).numpy()
                cond_f1 = float(
                    f1_score(y_true, cond_pred, labels=list(activity_ids), average="macro", zero_division=0)
                )
                uncond_f1 = float(
                    f1_score(y_true, uncond_pred, labels=list(activity_ids), average="macro", zero_division=0)
                )
                weighted_f1 = float(
                    f1_score(y_true, cond_pred, labels=list(activity_ids), average="weighted", zero_division=0)
                )
                cond_acc = float(accuracy_score(y_true, cond_pred))
                uncond_acc = float(accuracy_score(y_true, uncond_pred))
                gain = float(cond_f1 - uncond_f1)
                cond_f1s.append(cond_f1)
                uncond_f1s.append(uncond_f1)
                weighted_f1s.append(weighted_f1)
                gains.append(gain)
                cond_accs.append(cond_acc)
                uncond_accs.append(uncond_acc)
                confusion_accumulator[k]["conditioned"] += confusion_matrix(
                    y_true,
                    cond_pred,
                    labels=list(range(num_classes)),
                )
                confusion_accumulator[k]["unconditioned"] += confusion_matrix(
                    y_true,
                    uncond_pred,
                    labels=list(range(num_classes)),
                )
                all_trial_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "test_subject_id": test_subject_id,
                        "k": k,
                        "trial": int(trial),
                        "activity_ids": json.dumps([int(aid) for aid in activity_ids]),
                        "num_activities": int(len(activity_ids)),
                        "num_support": int(y_support.numel()),
                        "num_query": int(y_query.numel()),
                        "conditioned_macro_f1": cond_f1,
                        "unconditioned_macro_f1": uncond_f1,
                        "macro_f1_improvement": gain,
                        "conditioned_weighted_f1": weighted_f1,
                        "conditioned_accuracy": cond_acc,
                        "unconditioned_accuracy": uncond_acc,
                    }
                )

            cond_mean, cond_std, cond_ci95 = _mean_ci(cond_f1s)
            uncond_mean, uncond_std, uncond_ci95 = _mean_ci(uncond_f1s)
            gain_mean, gain_std, gain_ci95 = _mean_ci(gains)
            weighted_mean, weighted_std, weighted_ci95 = _mean_ci(weighted_f1s)
            cond_acc_mean, cond_acc_std, cond_acc_ci95 = _mean_ci(cond_accs)
            uncond_acc_mean, uncond_acc_std, uncond_acc_ci95 = _mean_ci(uncond_accs)
            subject_k_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "test_subject_id": test_subject_id,
                    "k": k,
                    "episodes": int(config.episodes_per_k),
                    "activity_ids": [int(aid) for aid in activity_ids],
                    "conditioned_macro_f1_mean": cond_mean,
                    "conditioned_macro_f1_std": cond_std,
                    "conditioned_macro_f1_ci95": cond_ci95,
                    "unconditioned_macro_f1_mean": uncond_mean,
                    "unconditioned_macro_f1_std": uncond_std,
                    "unconditioned_macro_f1_ci95": uncond_ci95,
                    "macro_f1_improvement_mean": gain_mean,
                    "macro_f1_improvement_std": gain_std,
                    "macro_f1_improvement_ci95": gain_ci95,
                    "conditioned_weighted_f1_mean": weighted_mean,
                    "conditioned_weighted_f1_std": weighted_std,
                    "conditioned_weighted_f1_ci95": weighted_ci95,
                    "conditioned_accuracy_mean": cond_acc_mean,
                    "conditioned_accuracy_std": cond_acc_std,
                    "conditioned_accuracy_ci95": cond_acc_ci95,
                    "unconditioned_accuracy_mean": uncond_acc_mean,
                    "unconditioned_accuracy_std": uncond_acc_std,
                    "unconditioned_accuracy_ci95": uncond_acc_ci95,
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
            "conditioned_macro_f1_mean",
            "unconditioned_macro_f1_mean",
            "macro_f1_improvement_mean",
            "conditioned_weighted_f1_mean",
            "conditioned_accuracy_mean",
            "unconditioned_accuracy_mean",
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
    plot_path = eval_dir / "k_shot_curve.png"
    _plot_k_curve(overall_df, plot_path)

    confusion_paths: list[str] = []
    for k, matrices in sorted(confusion_accumulator.items()):
        labels = list(range(int(cfg.num_of_activities)))
        cond_path = eval_dir / f"confusion_conditioned_k{k}.png"
        uncond_path = eval_dir / f"confusion_unconditioned_k{k}.png"
        _save_confusion_matrix(matrices["conditioned"], labels, cond_path, f"Conditioned Confusion Matrix (K={k})")
        _save_confusion_matrix(matrices["unconditioned"], labels, uncond_path, f"Unconditioned Confusion Matrix (K={k})")
        confusion_paths.extend([str(cond_path), str(uncond_path)])

    summary = {
        "config": asdict(config),
        "meta_stage_dir": str(meta_stage_dir),
        "splits_manifest_path": str(manifest_path),
        "skipped_folds": skipped_folds,
        "trial_results_csv": str(trial_csv),
        "subject_by_k_results_csv": str(subject_csv),
        "overall_by_k_results_csv": str(overall_csv),
        "plot_path": str(plot_path),
        "confusion_matrix_paths": confusion_paths,
        "overall_by_k": overall_df.to_dict(orient="records"),
        "num_trial_rows": int(trial_df.shape[0]),
        "num_subject_k_rows": int(subject_k_df.shape[0]),
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

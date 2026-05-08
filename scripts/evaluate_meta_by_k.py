from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import inspect
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from whar_datasets import (
    Loader,
    PostProcessingPipeline,
    PreProcessingPipeline,
    WHARDatasetID,
    get_dataset_cfg,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hyper_har.backbone.tinierhar import TinierHAR
from hyper_har.config import DEFAULT_CONFIG
from hyper_har.hypernet.hypernet import HyperNet
from hyper_har.splitting import MetaLOSOSplitter

DEFAULT_ARTIFACTS = ROOT / "artifacts"
DEFAULT_META_ROOT = DEFAULT_ARTIFACTS / "meta_loso_cv"
DEFAULT_BACKBONE_DIR = DEFAULT_ARTIFACTS / "loso_cv"
DEFAULT_K_VALUES = (2, 4, 8, 12, 16, 20)
DEFAULT_EPISODES = 64


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SET_ENCODER_SIMPLE_MODULE = _load_module_from_path(
    "hyper_har_eval_set_encoder_simple", SRC / "hyper_har" / "set-encoder" / "simple.py"
)
SET_ENCODER_ATTENTION_MODULE = _load_module_from_path(
    "hyper_har_eval_set_encoder_attention",
    SRC / "hyper_har" / "set-encoder" / "attention.py",
)
META_TRAINER_MODULE = _load_module_from_path(
    "hyper_har_eval_meta_trainer",
    SRC / "hyper_har" / "training" / "meta-trainer.py",
)

PrototypicalSetEncoder = SET_ENCODER_SIMPLE_MODULE.PrototypicalSetEncoder
AttentionSetEncoder = SET_ENCODER_ATTENTION_MODULE.AttentionSetEncoder
MetaTrainerConfig = META_TRAINER_MODULE.MetaTrainerConfig
SetToLoRAMetaTrainer = META_TRAINER_MODULE.SetToLoRAMetaTrainer


@dataclass
class SubjectKMetrics:
    subject_id: int
    test_subject_id: int
    split_index: int
    k: int
    episodes: int
    query_per_class: int
    activity_count: int
    base_loss: float
    adapted_loss: float
    base_macro_f1: float
    adapted_macro_f1: float
    macro_f1_improvement: float
    episode_base_macro_f1_mean: float
    episode_base_macro_f1_std: float
    episode_adapted_macro_f1_mean: float
    episode_adapted_macro_f1_std: float
    episode_improvement_mean: float
    episode_improvement_std: float
    episode_improvement_sem: float
    episode_improvement_ci95: float
    lora_relative_delta_norm: float


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)}.")
    return payload


def _discover_latest_meta_variant(meta_root: Path) -> Path:
    candidates: list[tuple[float, Path]] = []
    for variant_dir in meta_root.iterdir():
        if not variant_dir.is_dir():
            continue
        summary_path = variant_dir / "summary.json"
        run_config_path = variant_dir / "run_config.json"
        subject_metric_files = list(variant_dir.glob("subject_*/meta_metrics.json"))
        if summary_path.exists():
            candidates.append((summary_path.stat().st_mtime, variant_dir))
        elif run_config_path.exists():
            candidates.append((run_config_path.stat().st_mtime, variant_dir))
        elif subject_metric_files:
            candidates.append(
                (max(p.stat().st_mtime for p in subject_metric_files), variant_dir)
            )
    if not candidates:
        raise FileNotFoundError(f"No meta variant artifacts found in {meta_root}")
    return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]


def _resolve_meta_variant(meta_root: Path, meta_variant: str | None) -> Path:
    if meta_variant is None:
        return _discover_latest_meta_variant(meta_root)

    as_path = Path(meta_variant)
    if as_path.exists() and as_path.is_dir():
        return as_path

    candidate = meta_root / meta_variant
    if candidate.exists() and candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve meta variant. Tried '{as_path}' and '{candidate}'."
    )


def _resolve_meta_subfolder(
    artifacts_root: Path,
    meta_root: Path,
    meta_subfolder: str | None,
    meta_variant: str | None,
) -> Path:
    if meta_subfolder is None:
        return _resolve_meta_variant(meta_root, meta_variant)

    as_path = Path(meta_subfolder)
    if as_path.exists() and as_path.is_dir():
        return as_path

    artifacts_candidate = artifacts_root / meta_subfolder
    if artifacts_candidate.exists() and artifacts_candidate.is_dir():
        return artifacts_candidate

    meta_root_candidate = meta_root / meta_subfolder
    if meta_root_candidate.exists() and meta_root_candidate.is_dir():
        return meta_root_candidate

    raise FileNotFoundError(
        "Could not resolve --meta-subfolder. "
        f"Tried '{as_path}', '{artifacts_candidate}', and '{meta_root_candidate}'."
    )


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise ValueError("Expected at least one integer.")
    return result


def _infer_window_size(loader: Loader, indices: Sequence[int]) -> int:
    if len(indices) == 0:
        raise ValueError("Cannot infer window size from an empty index set.")
    sample = loader.get_sample(int(indices[0]))
    if not sample:
        raise ValueError("Could not infer window size: empty sample.")
    x = np.asarray(sample[0])
    if x.ndim == 2:
        return int(x.shape[0])
    if x.ndim == 3 and x.shape[0] == 1:
        return int(x.shape[1])
    raise ValueError(f"Unexpected sample shape for window inference: {tuple(x.shape)}")


def _infer_subject_id(loader: Loader, indices: Sequence[int], fallback: int) -> int:
    if len(indices) == 0:
        return fallback
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    session_meta = (
        loader.session_df[["session_id", "subject_id"]]
        .drop_duplicates("session_id")
        .set_index("session_id")
    )
    merged = subset.join(session_meta, on="session_id", how="left")
    subjects = merged["subject_id"].dropna().astype(int).unique().tolist()
    return int(subjects[0]) if len(subjects) == 1 else fallback


def _activity_support_by_subject(
    loader: Loader, indices: Sequence[int]
) -> tuple[dict[int, dict[int, int]], list[int]]:
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    subset["window_index"] = subset.index.astype(int)

    session_meta = loader.session_df[
        ["session_id", "subject_id", "activity_id"]
    ].drop_duplicates("session_id")
    merged = subset.merge(session_meta, on="session_id", how="left")
    if merged["subject_id"].isna().any() or merged["activity_id"].isna().any():
        raise ValueError("Missing subject/activity metadata.")

    counts = (
        merged.groupby(["subject_id", "activity_id"])
        .size()
        .rename("count")
        .reset_index()
    )
    activity_ids = sorted(int(x) for x in counts["activity_id"].unique().tolist())
    support: dict[int, dict[int, int]] = {}
    for row in counts.itertuples(index=False):
        support.setdefault(int(row.subject_id), {})[int(row.activity_id)] = int(
            row.count
        )
    return support, activity_ids


def _choose_activity_ids(
    loader: Loader,
    indices: Sequence[int],
    needed_per_subject_activity: int,
    min_subjects: int,
) -> list[int]:
    support, activities = _activity_support_by_subject(loader, indices)
    candidate = activities.copy()
    while candidate:
        eligible_subjects = [
            sid
            for sid, per_activity in support.items()
            if all(
                per_activity.get(aid, 0) >= needed_per_subject_activity
                for aid in candidate
            )
        ]
        if len(eligible_subjects) >= min_subjects:
            return candidate

        support_counts = {
            aid: sum(
                1
                for per_activity in support.values()
                if per_activity.get(aid, 0) >= needed_per_subject_activity
            )
            for aid in candidate
        }
        drop_aid = min(candidate, key=lambda aid: (support_counts[aid], aid))
        candidate = [aid for aid in candidate if aid != drop_aid]

    raise ValueError("No activity set has enough support for episodic evaluation.")


def _fetch_class_weights(
    loader: object, split: object, num_classes: int
) -> torch.Tensor | None:
    get_weights = getattr(loader, "get_class_weights", None)
    if get_weights is None:
        return None

    weights_obj = None
    try:
        n_params = len(inspect.signature(get_weights).parameters)
    except (TypeError, ValueError):
        n_params = -1

    if n_params in (0, -1):
        try:
            weights_obj = get_weights()
        except TypeError:
            pass

    if weights_obj is None and n_params in (1, -1):
        train_indices = getattr(split, "train_indices", None)
        if train_indices is not None:
            try:
                weights_obj = get_weights(train_indices)
            except TypeError:
                pass

    if weights_obj is None:
        return None

    if isinstance(weights_obj, dict):
        weights = torch.ones(num_classes, dtype=torch.float32)
        for class_id in range(num_classes):
            raw_w = float(weights_obj.get(class_id, 1.0))
            weights[class_id] = 0.0 if raw_w < 0.0 else raw_w
        return weights

    weights = torch.as_tensor(weights_obj, dtype=torch.float32).view(-1)
    if weights.numel() != num_classes:
        raise ValueError(
            f"Class weights length mismatch: expected {num_classes}, got {weights.numel()}."
        )
    return torch.where(weights < 0.0, torch.zeros_like(weights), weights)


def _build_set_encoder(
    kind: str,
    backbone: TinierHAR,
    num_classes: int,
    backbone_train_mode: str,
    force_conv_bn_eval: bool,
) -> torch.nn.Module:
    set_encoder_backbone = (
        backbone if backbone_train_mode == "freeze_all" else copy.deepcopy(backbone)
    )
    kind_norm = kind.strip().lower()
    if kind_norm == "prototypical":
        return PrototypicalSetEncoder(
            backbone=set_encoder_backbone,
            num_classes=num_classes,
            backbone_train_mode=backbone_train_mode,
            force_conv_bn_eval=force_conv_bn_eval,
            set_encoder_config=DEFAULT_CONFIG.set_encoder,
        )
    if kind_norm == "attention":
        return AttentionSetEncoder(
            backbone=set_encoder_backbone,
            num_classes=num_classes,
            backbone_train_mode=backbone_train_mode,
            force_conv_bn_eval=force_conv_bn_eval,
            set_encoder_config=DEFAULT_CONFIG.set_encoder,
        )
    raise ValueError(f"Unsupported set encoder kind: {kind!r}")


def _forward_adapted(
    trainer: SetToLoRAMetaTrainer,
    lora_weights: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    x_query: torch.Tensor,
    use_vmap: bool,
) -> torch.Tensor:
    batched_params = trainer._build_batched_params(x_query.size(0), lora_weights)
    if use_vmap:
        try:
            return trainer._forward_queries_vmap(batched_params, x_query)
        except RuntimeError as exc:
            if "aten::gru.input" not in str(exc):
                raise
    return trainer._forward_queries_loop(batched_params, x_query)


def _mean_std_sem(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean_v = float(arr.mean())
    std_v = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sem_v = float(std_v / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean_v, std_v, sem_v, float(1.96 * sem_v)


def _evaluate_subject_for_k(
    trainer: SetToLoRAMetaTrainer,
    subject_id: int,
    test_subject_id: int,
    split_index: int,
    k: int,
    episodes: int,
    query_per_class: int,
    use_vmap: bool,
) -> SubjectKMetrics:
    trainer.base_model.eval()
    trainer.set_encoder.eval()
    trainer.hypernet.eval()

    base_losses: list[float] = []
    adapted_losses: list[float] = []
    all_targets: list[torch.Tensor] = []
    base_preds_all: list[torch.Tensor] = []
    adapted_preds_all: list[torch.Tensor] = []
    episode_base_f1s: list[float] = []
    episode_adapted_f1s: list[float] = []
    episode_improvements: list[float] = []
    delta_norms: list[float] = []
    base_params = dict(trainer.base_model.named_parameters())

    with torch.no_grad():
        progress = tqdm(
            range(episodes),
            desc=f"Eval subject {subject_id} K={k}",
            leave=False,
        )
        for _ in progress:
            x_support, y_support, x_query, y_query, _subjects = trainer._sample_episode(
                support_per_class=k
            )
            x_support = x_support.to(trainer.device)
            y_support = y_support.to(trainer.device)
            x_query = x_query.to(trainer.device)
            y_query = y_query.to(trainer.device)

            targets_flat = y_query.reshape(-1)
            base_logits = trainer.base_model(
                x_query.reshape(-1, *x_query.shape[2:])
            ).reshape(x_query.size(0), x_query.size(1), -1)
            base_logits_flat = base_logits.reshape(-1, base_logits.size(-1))
            base_loss = F.cross_entropy(
                base_logits_flat, targets_flat, weight=trainer.class_weights
            )

            c_subject = trainer.set_encoder(x_support, y_support)
            lora_weights = trainer.hypernet(c_subject)
            for adapter_name, param_name in trainer.target_param_names.items():
                A, B = lora_weights[adapter_name]
                delta = trainer._compute_lora_delta(A, B) * trainer.lora_scale
                base_norm = base_params[param_name].detach().norm().clamp_min(1e-12)
                rel_norm = delta.flatten(1).norm(dim=1) / base_norm
                delta_norms.append(float(rel_norm.mean().item()))

            adapted_logits = _forward_adapted(
                trainer, lora_weights, x_query, use_vmap=use_vmap
            )
            adapted_logits_flat = adapted_logits.reshape(-1, adapted_logits.size(-1))
            adapted_loss = F.cross_entropy(
                adapted_logits_flat, targets_flat, weight=trainer.class_weights
            )

            base_preds = base_logits.argmax(dim=-1).reshape(-1).cpu()
            adapted_preds = adapted_logits.argmax(dim=-1).reshape(-1).cpu()
            targets_cpu = targets_flat.cpu()

            episode_base_f1 = f1_score(
                targets_cpu.numpy(),
                base_preds.numpy(),
                average="macro",
                zero_division=0,
            )
            episode_adapted_f1 = f1_score(
                targets_cpu.numpy(),
                adapted_preds.numpy(),
                average="macro",
                zero_division=0,
            )

            base_losses.append(float(base_loss.item()))
            adapted_losses.append(float(adapted_loss.item()))
            all_targets.append(targets_cpu)
            base_preds_all.append(base_preds)
            adapted_preds_all.append(adapted_preds)
            episode_base_f1s.append(float(episode_base_f1))
            episode_adapted_f1s.append(float(episode_adapted_f1))
            episode_improvements.append(float(episode_adapted_f1 - episode_base_f1))

    targets_t = torch.cat(all_targets)
    base_preds_t = torch.cat(base_preds_all)
    adapted_preds_t = torch.cat(adapted_preds_all)
    base_macro_f1 = f1_score(
        targets_t.numpy(), base_preds_t.numpy(), average="macro", zero_division=0
    )
    adapted_macro_f1 = f1_score(
        targets_t.numpy(), adapted_preds_t.numpy(), average="macro", zero_division=0
    )
    base_ep_mean, base_ep_std, _, _ = _mean_std_sem(episode_base_f1s)
    adapted_ep_mean, adapted_ep_std, _, _ = _mean_std_sem(episode_adapted_f1s)
    imp_ep_mean, imp_ep_std, imp_ep_sem, imp_ep_ci95 = _mean_std_sem(
        episode_improvements
    )

    return SubjectKMetrics(
        subject_id=subject_id,
        test_subject_id=test_subject_id,
        split_index=split_index,
        k=k,
        episodes=episodes,
        query_per_class=query_per_class,
        activity_count=len(trainer.activity_ids),
        base_loss=sum(base_losses) / max(1, len(base_losses)),
        adapted_loss=sum(adapted_losses) / max(1, len(adapted_losses)),
        base_macro_f1=float(base_macro_f1),
        adapted_macro_f1=float(adapted_macro_f1),
        macro_f1_improvement=float(adapted_macro_f1 - base_macro_f1),
        episode_base_macro_f1_mean=base_ep_mean,
        episode_base_macro_f1_std=base_ep_std,
        episode_adapted_macro_f1_mean=adapted_ep_mean,
        episode_adapted_macro_f1_std=adapted_ep_std,
        episode_improvement_mean=imp_ep_mean,
        episode_improvement_std=imp_ep_std,
        episode_improvement_sem=imp_ep_sem,
        episode_improvement_ci95=imp_ep_ci95,
        lora_relative_delta_norm=sum(delta_norms) / max(1, len(delta_norms)),
    )


def _plot_k_bar(rows: list[dict[str, Any]], k: int, output_path: Path) -> None:
    rows_k = sorted(
        [r for r in rows if int(r["k"]) == k], key=lambda r: int(r["subject_id"])
    )
    if not rows_k:
        return
    subject_ids = [int(r["subject_id"]) for r in rows_k]
    improvements = [float(r["macro_f1_improvement"]) for r in rows_k]
    mean_improvement = float(np.mean(improvements))
    mean_x = max(subject_ids) + 1 if subject_ids else 0
    all_x = subject_ids + [mean_x]
    all_y = improvements + [mean_improvement]
    colors = ["#2a9d8f" if value >= 0 else "#d62828" for value in improvements]
    colors.append("#264653")

    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(all_x, all_y, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Same-Query Macro F1 Improvement")
    ax.set_title(f"Meta Adaptation Improvement by Subject (K={k})")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticks(all_x)
    ax.set_xticklabels([str(s) for s in subject_ids] + ["mean"])

    y_min = min(all_y) if all_y else 0.0
    y_max = max(all_y) if all_y else 0.0
    pad = 0.03 * max(1e-9, y_max - y_min)
    for bar, row in zip(bars[:-1], rows_k):
        value = float(row["macro_f1_improvement"])
        base_f1 = float(row["base_macro_f1"])
        adapted_f1 = float(row["adapted_macro_f1"])
        label = f"{value:+.4f}\n{base_f1:.4f}->{adapted_f1:.4f}"
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()
        ax.text(
            x,
            y + pad if value >= 0 else y - pad,
            label,
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
            rotation=90,
        )

    mean_bar = bars[-1]
    mean_xpos = mean_bar.get_x() + mean_bar.get_width() / 2.0
    mean_y = mean_bar.get_height()
    ax.text(
        mean_xpos,
        mean_y + pad if mean_improvement >= 0 else mean_y - pad,
        f"mean\n{mean_improvement:+.4f}",
        ha="center",
        va="bottom" if mean_improvement >= 0 else "top",
        fontsize=9,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_k_summary(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return
    rows = sorted(summary_rows, key=lambda r: int(r["k"]))
    k_values = [int(r["k"]) for r in rows]
    means = [float(r["mean_macro_f1_improvement"]) for r in rows]
    sems = [float(r["sem_macro_f1_improvement"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        [str(k) for k in k_values], means, yerr=sems, capsize=5, color="#457b9d"
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Support Windows per Class (K)")
    ax.set_ylabel("Mean Same-Query Macro F1 Improvement")
    ax.set_title("Mean Adaptation Improvement by K")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:+.4f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize_by_k(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    k_values = sorted({int(row["k"]) for row in rows})
    for k in k_values:
        rows_k = [row for row in rows if int(row["k"]) == k]
        improvements = [float(row["macro_f1_improvement"]) for row in rows_k]
        mean_imp, std_imp, sem_imp, ci95_imp = _mean_std_sem(improvements)
        summary.append(
            {
                "k": k,
                "num_subjects": len(rows_k),
                "mean_base_macro_f1": float(
                    np.mean([float(row["base_macro_f1"]) for row in rows_k])
                ),
                "mean_adapted_macro_f1": float(
                    np.mean([float(row["adapted_macro_f1"]) for row in rows_k])
                ),
                "mean_macro_f1_improvement": mean_imp,
                "std_macro_f1_improvement": std_imp,
                "sem_macro_f1_improvement": sem_imp,
                "ci95_macro_f1_improvement": ci95_imp,
                "mean_episode_improvement_sem": float(
                    np.mean([float(row["episode_improvement_sem"]) for row in rows_k])
                ),
                "mean_lora_relative_delta_norm": float(
                    np.mean([float(row["lora_relative_delta_norm"]) for row in rows_k])
                ),
            }
        )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load trained Set-to-LoRA meta checkpoints and run extensive same-query "
            "base/adapted episodic evaluation for each requested K."
        )
    )
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--meta-root", type=Path, default=DEFAULT_META_ROOT)
    parser.add_argument("--meta-variant", type=str, default=None)
    parser.add_argument("--meta-subfolder", type=str, default=None)
    parser.add_argument("--backbone-dir", type=Path, default=DEFAULT_BACKBONE_DIR)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--k-values",
        type=str,
        default=None,
        help="Comma-separated K values. Defaults to run_config eval K values or 2,4,8,12,16,20.",
    )
    parser.add_argument(
        "--query-per-class",
        type=int,
        default=None,
        help="Evaluation query windows per class. Defaults to run_config eval_query_per_class or 16.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Optional comma-separated held-out subject ids to evaluate.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--use-vmap", action=argparse.BooleanOptionalAction, default=None
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    meta_root = args.meta_root
    backbone_dir = args.backbone_dir
    if args.artifacts_root != DEFAULT_ARTIFACTS:
        if args.meta_root == DEFAULT_META_ROOT:
            meta_root = args.artifacts_root / "meta_loso_cv"
        if args.backbone_dir == DEFAULT_BACKBONE_DIR:
            backbone_dir = args.artifacts_root / "loso_cv"

    meta_variant_dir = _resolve_meta_subfolder(
        artifacts_root=args.artifacts_root,
        meta_root=meta_root,
        meta_subfolder=args.meta_subfolder,
        meta_variant=args.meta_variant,
    )
    run_config_path = meta_variant_dir / "run_config.json"
    run_config = _load_json(run_config_path) if run_config_path.exists() else {}
    output_dir = args.output_dir or (meta_variant_dir / "evaluation_by_k")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    k_values = _parse_int_list(args.k_values)
    if k_values is None:
        k_values = [
            int(k)
            for k in run_config.get("eval_support_per_class_choices", DEFAULT_K_VALUES)
        ]
    query_per_class = (
        args.query_per_class
        if args.query_per_class is not None
        else _as_int(run_config.get("eval_query_per_class"), 16)
    )
    selected_subjects = _parse_int_list(args.subjects)
    selected_subject_set = set(selected_subjects) if selected_subjects else None

    device_str = args.device or (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    use_vmap = (
        args.use_vmap
        if args.use_vmap is not None
        else _as_bool(run_config.get("use_vmap"), True)
    )
    set_encoder_kind = str(run_config.get("set_encoder_kind", "attention"))
    set_encoder_mode = str(
        run_config.get("set_encoder_backbone_train_mode", "freeze_conv_blocks")
    )
    force_conv_bn_eval = _as_bool(run_config.get("force_conv_bn_eval"), True)
    lora_rank = _as_int(run_config.get("lora_rank"), DEFAULT_CONFIG.hypernet.lora_rank)
    lora_alpha = _as_float(
        run_config.get("lora_alpha"), DEFAULT_CONFIG.hypernet.lora_alpha
    )
    enable_conv1_adapter = _as_bool(run_config.get("enable_conv1_adapter"), False)
    enable_conv_last_adapter = _as_bool(
        run_config.get("enable_conv_last_adapter"), False
    )

    cfg = get_dataset_cfg(WHARDatasetID.WEAR, datasets_dir=str(ROOT / "datasets"))
    pre_pipeline = PreProcessingPipeline(cfg)
    _, session_df, window_df = pre_pipeline.run()
    splitter = MetaLOSOSplitter(cfg)
    folds = splitter.get_folds(session_df, window_df)

    max_k = max(k_values)
    needed_per_activity = max_k + query_per_class
    all_rows: list[dict[str, Any]] = []

    for k in k_values:
        print(f"\n### Evaluating K={k} across subjects ###")
        rows_before_k = len(all_rows)

        for split_idx, fold in enumerate(folds):
            subject_id = int(fold.test_subject_id)
            if selected_subject_set is not None and subject_id not in selected_subject_set:
                continue

            split = fold.meta_split
            subject_dir = meta_variant_dir / f"subject_{subject_id}"
            checkpoint_path = subject_dir / "best_meta_modules.pt"
            if not checkpoint_path.exists():
                print(f"Skipping subject {subject_id}: missing {checkpoint_path}")
                continue

            print(
                f"\n=== Extensive meta eval K={k} | split {split_idx + 1}/{len(folds)} | subject={subject_id} ==="
            )
            post_pipeline = PostProcessingPipeline(
                cfg, pre_pipeline, window_df, split.train_indices
            )
            samples = post_pipeline.run()
            loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)
            inferred_subject = _infer_subject_id(
                loader, split.test_indices, fallback=subject_id
            )

            num_channels = cfg.num_of_channels
            num_classes = cfg.num_of_activities
            window_size = _infer_window_size(loader, split.train_indices)
            base_model = TinierHAR(
                num_channels=num_channels,
                num_classes=num_classes,
                window_size=window_size,
                backbone_config=DEFAULT_CONFIG.backbone,
            )
            backbone_ckpt = backbone_dir / f"subject_{subject_id}" / "best_tinierhar.pt"
            if not backbone_ckpt.exists():
                raise FileNotFoundError(f"Missing pretrained backbone: {backbone_ckpt}")
            base_model.load_state_dict(torch.load(backbone_ckpt, map_location=device_str))
            base_model.eval()
            for param in base_model.parameters():
                param.requires_grad = False

            set_encoder = _build_set_encoder(
                kind=set_encoder_kind,
                backbone=base_model,
                num_classes=num_classes,
                backbone_train_mode=set_encoder_mode,
                force_conv_bn_eval=force_conv_bn_eval,
            )
            hypernet = HyperNet(
                num_channels=num_channels,
                num_classes=num_classes,
                set_encoder_output_dim=getattr(set_encoder, "output_dim", None),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                enable_conv1_adapter=enable_conv1_adapter,
                enable_conv_last_adapter=enable_conv_last_adapter,
                set_encoder_config=DEFAULT_CONFIG.set_encoder,
                backbone_config=DEFAULT_CONFIG.backbone,
                hypernet_config=DEFAULT_CONFIG.hypernet,
            )
            checkpoint = torch.load(checkpoint_path, map_location=device_str)
            set_encoder.load_state_dict(checkpoint["set_encoder"])
            hypernet.load_state_dict(checkpoint["hypernet"])

            activity_ids = _choose_activity_ids(
                loader,
                split.test_indices,
                needed_per_subject_activity=needed_per_activity,
                min_subjects=1,
            )
            eval_cfg = MetaTrainerConfig(
                batch_subjects=1,
                support_per_class=max_k,
                support_per_class_choices=k_values,
                query_per_class=query_per_class,
                use_vmap=use_vmap,
                seed=40_000 + split_idx + (1000 * int(k)),
                device=device_str,
            )
            class_weights = _fetch_class_weights(loader, split, num_classes)
            trainer = SetToLoRAMetaTrainer(
                base_model=base_model,
                set_encoder=set_encoder,
                hypernet=hypernet,
                loader=loader,
                num_classes=num_classes,
                config=eval_cfg,
                class_weights=class_weights,
                indices=split.test_indices,
                activity_ids=activity_ids,
            )

            metrics = _evaluate_subject_for_k(
                trainer=trainer,
                subject_id=inferred_subject,
                test_subject_id=subject_id,
                split_index=split_idx,
                k=int(k),
                episodes=args.episodes,
                query_per_class=query_per_class,
                use_vmap=use_vmap,
            )
            row = asdict(metrics)
            row["checkpoint_path"] = str(checkpoint_path)
            row["backbone_checkpoint"] = str(backbone_ckpt)
            all_rows.append(row)
            print(
                f"subject={inferred_subject} K={k}: "
                f"base={metrics.base_macro_f1:.4f} "
                f"adapted={metrics.adapted_macro_f1:.4f} "
                f"improvement={metrics.macro_f1_improvement:+.4f} "
                f"episode_sem={metrics.episode_improvement_sem:.4f}"
            )

        if len(all_rows) == rows_before_k:
            print(f"No rows produced for K={k}; skipping plot.")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        rows_k = [row for row in all_rows if int(row["k"]) == int(k)]
        _write_csv(output_dir / f"per_subject_k_{int(k)}.csv", rows_k)
        _plot_k_bar(rows_k, int(k), plot_dir / f"k_{int(k)}_improvement_barplot.png")

        partial_summary_rows = _summarize_by_k(all_rows)
        _write_csv(output_dir / "per_subject_by_k.csv", all_rows)
        _write_csv(output_dir / "summary_by_k.csv", partial_summary_rows)
        _plot_k_summary(
            partial_summary_rows, plot_dir / "k_summary_improvement_barplot.png"
        )
        with (output_dir / "evaluation_by_k.partial.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "meta_variant_dir": str(meta_variant_dir),
                    "output_dir": str(output_dir),
                    "episodes": args.episodes,
                    "completed_k_values": sorted({int(row["k"]) for row in all_rows}),
                    "requested_k_values": k_values,
                    "query_per_class": query_per_class,
                    "device": device_str,
                    "use_vmap": use_vmap,
                    "summary_by_k": partial_summary_rows,
                    "per_subject_by_k": all_rows,
                },
                f,
                indent=2,
            )
        print(f"Saved completed K={k} plot and partial summaries to: {output_dir}")

    if not all_rows:
        raise RuntimeError("No evaluation rows produced. Check variant/checkpoints.")

    summary_rows = _summarize_by_k(all_rows)
    _write_csv(output_dir / "per_subject_by_k.csv", all_rows)
    _write_csv(output_dir / "summary_by_k.csv", summary_rows)

    _plot_k_summary(summary_rows, plot_dir / "k_summary_improvement_barplot.png")

    payload = {
        "meta_variant_dir": str(meta_variant_dir),
        "output_dir": str(output_dir),
        "episodes": args.episodes,
        "k_values": k_values,
        "query_per_class": query_per_class,
        "device": device_str,
        "use_vmap": use_vmap,
        "run_config": run_config,
        "summary_by_k": summary_rows,
        "per_subject_by_k": all_rows,
    }
    with (output_dir / "evaluation_by_k.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Extensive evaluation finished ===")
    print(f"Saved per-subject CSV: {output_dir / 'per_subject_by_k.csv'}")
    print(f"Saved K summary CSV: {output_dir / 'summary_by_k.csv'}")
    print(f"Saved JSON: {output_dir / 'evaluation_by_k.json'}")
    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()

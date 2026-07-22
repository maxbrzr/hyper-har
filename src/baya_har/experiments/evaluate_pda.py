import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from .common import (
    DEFAULT_DATASET_ID,
    DEFAULT_DATASETS_DIR,
    DEFAULT_SEED,
    DEFAULT_SELECTED_ACTIVITIES,
    DEFAULT_SPLIT_STRATEGY,
    DEFAULT_TEST_SUBJECTS,
    DEFAULT_VAL_PERCENTAGE,
    DEFAULT_VAL_SUBJECTS,
    DEFAULT_WINDOW_OVERLAP,
    SharedConfig,
    WindowDataset,
    build_loader,
    build_or_load_loso_folds,
    class_names,
    config_fingerprint,
    indices_by_activity,
    load_ce_backbone,
    mean_std_ci,
    prepare_cfg,
    prepare_inputs,
    prototype_logits,
    reconcile_activity_config,
    resolve_output_root,
    save_confusion_matrix_plot,
    set_seed,
    split_indices_for_fold,
)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from whar_datasets import PreProcessingPipeline, WHARDatasetID

try:
    from sklearn.covariance import MinCovDet
except Exception:  # pragma: no cover - optional refinement dependency
    MinCovDet = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Config:
    dataset_id: str = DEFAULT_DATASET_ID
    datasets_dir: str = DEFAULT_DATASETS_DIR
    selected_activities: list[str] | None = DEFAULT_SELECTED_ACTIVITIES
    window_overlap: float = DEFAULT_WINDOW_OVERLAP
    val_subjects: int = DEFAULT_VAL_SUBJECTS
    test_subjects: int = DEFAULT_TEST_SUBJECTS
    seed: int = DEFAULT_SEED
    split_strategy: str = DEFAULT_SPLIT_STRATEGY
    val_percentage: float = DEFAULT_VAL_PERCENTAGE

    k_values: tuple[int, ...] = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        32,
    )
    episodes_per_k: int = 100
    min_query_per_class: int = 1
    support_query_session_disjoint: bool = False
    min_activities_per_episode: int = 2
    require_all_k_activities: bool = True
    skip_ineligible_k_values: bool = True
    batch_size: int = 256
    num_workers: int = 0
    cosine_temperature: float = 0.1
    normalize_embeddings: bool = True
    embedding_transform: str = "none"  # "none" or "signed_power"
    power_transform_exponent: float = 0.5
    confidence_weighting: bool = True
    pseudo_labeler: str = "classifier"  # "classifier" or "mcd"
    pseudo_label_temperature: float = 1.0
    center_support_query: bool = False
    mcd_min_samples: int = 8
    mcd_support_fraction: float | None = None
    mcd_regularization: float = 1e-4
    mcd_diagonal_fallback: bool = True
    skip_missing_folds: bool = False
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_root: str | None = None
    ce_stage_name: str = "classifier"
    eval_stage_name: str = "pda"
    max_folds: int | None = None
    force_rerun: bool = False


RUN_CONFIG = Config()


def _artifact_safe_name(value: str) -> str:
    return value.replace(".", "p").replace("-", "m").replace("+", "p").replace(" ", "_")


def _active_embedding_transform_name(config: Config) -> str:
    if config.embedding_transform == "none":
        return "none"
    if config.embedding_transform != "signed_power":
        raise ValueError("embedding_transform must be 'none' or 'signed_power'.")
    return f"signed_power_{float(config.power_transform_exponent):g}"


def _apply_embedding_transform(
    embeddings: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, str]:
    transform_name = _active_embedding_transform_name(config)
    if transform_name == "none":
        return embeddings, transform_name
    exponent = float(config.power_transform_exponent)
    if exponent <= 0:
        raise ValueError("power_transform_exponent must be > 0.")
    transformed = torch.sign(embeddings) * torch.abs(embeddings).pow(exponent)
    return transformed, transform_name


@torch.no_grad()
def _extract_features_logits(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    model.eval()
    embeddings: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    subjects: list[torch.Tensor] = []
    for batch in dataloader:
        x = prepare_inputs(batch["x"]).to(device).float()
        emb = model.encode(x)
        logits = model.classifier(emb)
        embeddings.append(emb.cpu())
        logits_list.append(logits.cpu())
        labels.append(batch["y"].long().view(-1).cpu())
        subjects.append(batch["subject_id"].long().view(-1).cpu())
    emb_all = torch.cat(embeddings, dim=0)
    emb_all, transform_name = _apply_embedding_transform(emb_all, config)
    if config.normalize_embeddings:
        emb_all = F.normalize(emb_all, p=2, dim=1)
    return (
        emb_all,
        torch.cat(logits_list, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(subjects, dim=0),
        transform_name,
    )


def _eligible_activities(
    by_activity: dict[int, np.ndarray],
    k: int,
    min_query_per_class: int,
) -> list[int]:
    return [
        int(activity_id)
        for activity_id, values in sorted(by_activity.items())
        if len(values) >= int(k) + int(min_query_per_class)
    ]


def _indices_by_activity_session(
    loader: Any,
    indices: Sequence[int],
) -> dict[int, dict[Any, np.ndarray]]:
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    subset["window_index"] = subset.index.astype(int)
    session_meta = loader.session_df[["session_id", "activity_id"]].drop_duplicates(
        "session_id"
    )
    merged = subset.merge(session_meta, on="session_id", how="left")
    if merged["session_id"].isna().any() or merged["activity_id"].isna().any():
        raise ValueError("Missing session/activity metadata.")
    nested: dict[int, dict[Any, np.ndarray]] = {}
    grouped = merged.groupby(["activity_id", "session_id"])["window_index"]
    for (activity_id, session_id), group in grouped:
        nested.setdefault(int(activity_id), {})[session_id] = group.to_numpy(
            dtype=np.int64
        )
    return nested


def _eligible_activities_session_disjoint(
    by_activity_session: dict[int, dict[Any, np.ndarray]],
    k: int,
    min_query_per_class: int,
) -> list[int]:
    eligible: list[int] = []
    for activity_id, by_session in sorted(by_activity_session.items()):
        sessions = list(by_session.keys())
        if len(sessions) < 2:
            continue
        for query_session_id in sessions:
            query_count = len(by_session[query_session_id])
            support_count = sum(
                len(values)
                for session_id, values in by_session.items()
                if session_id != query_session_id
            )
            if query_count >= int(min_query_per_class) and support_count >= int(k):
                eligible.append(int(activity_id))
                break
    return eligible


def _sample_episode(
    by_activity: dict[int, np.ndarray],
    activity_ids: Sequence[int],
    k: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    support_indices: list[int] = []
    query_indices: list[int] = []
    support_y: list[int] = []
    query_y: list[int] = []
    for activity_id in activity_ids:
        candidates = np.asarray(by_activity[int(activity_id)], dtype=np.int64)
        perm = rng.permutation(candidates)
        support = perm[: int(k)]
        query = perm[int(k) :]
        support_indices.extend(int(x) for x in support.tolist())
        query_indices.extend(int(x) for x in query.tolist())
        support_y.extend([int(activity_id)] * len(support))
        query_y.extend([int(activity_id)] * len(query))
    query_perm = rng.permutation(len(query_indices))
    support_perm = rng.permutation(len(support_indices))
    support_indices = [support_indices[int(i)] for i in support_perm.tolist()]
    support_y_np = np.asarray(
        [support_y[int(i)] for i in support_perm.tolist()], dtype=np.int64
    )
    query_indices = [query_indices[int(i)] for i in query_perm.tolist()]
    query_y_np = np.asarray(
        [query_y[int(i)] for i in query_perm.tolist()], dtype=np.int64
    )
    return support_indices, query_indices, support_y_np, query_y_np


def _sample_episode_session_disjoint(
    by_activity_session: dict[int, dict[Any, np.ndarray]],
    activity_ids: Sequence[int],
    k: int,
    min_query_per_class: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
    support_indices: list[int] = []
    query_indices: list[int] = []
    support_y: list[int] = []
    query_y: list[int] = []
    session_meta: dict[int, dict[str, Any]] = {}
    for activity_id in activity_ids:
        by_session = by_activity_session[int(activity_id)]
        candidate_query_sessions = []
        for query_session_id, query_candidates in by_session.items():
            support_candidates = [
                values
                for session_id, values in by_session.items()
                if session_id != query_session_id
            ]
            support_count = sum(len(values) for values in support_candidates)
            if len(query_candidates) >= int(
                min_query_per_class
            ) and support_count >= int(k):
                candidate_query_sessions.append(query_session_id)
        if not candidate_query_sessions:
            raise ValueError(
                f"Activity {activity_id} has no session-disjoint support/query split "
                f"for k={k}."
            )
        query_session_id = candidate_query_sessions[
            int(rng.integers(0, len(candidate_query_sessions)))
        ]
        query = np.asarray(by_session[query_session_id], dtype=np.int64)
        support_pool = np.concatenate(
            [
                np.asarray(values, dtype=np.int64)
                for session_id, values in by_session.items()
                if session_id != query_session_id
            ],
            axis=0,
        )
        support = rng.choice(support_pool, size=int(k), replace=False)
        support_indices.extend(int(x) for x in support.tolist())
        query_indices.extend(int(x) for x in query.tolist())
        support_y.extend([int(activity_id)] * len(support))
        query_y.extend([int(activity_id)] * len(query))
        support_session_ids = [
            session_id
            for session_id, values in by_session.items()
            if session_id != query_session_id
            and np.isin(support, np.asarray(values, dtype=np.int64)).any()
        ]
        session_meta[int(activity_id)] = {
            "query_session_id": str(query_session_id),
            "support_session_ids": [str(x) for x in support_session_ids],
        }

    query_perm = rng.permutation(len(query_indices))
    support_perm = rng.permutation(len(support_indices))
    support_indices = [support_indices[int(i)] for i in support_perm.tolist()]
    support_y_np = np.asarray(
        [support_y[int(i)] for i in support_perm.tolist()], dtype=np.int64
    )
    query_indices = [query_indices[int(i)] for i in query_perm.tolist()]
    query_y_np = np.asarray(
        [query_y[int(i)] for i in query_perm.tolist()], dtype=np.int64
    )
    return support_indices, query_indices, support_y_np, query_y_np, session_meta


def _classifier_pseudo_labels(
    logits: torch.Tensor,
    activity_ids: Sequence[int],
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if float(temperature) <= 0:
        raise ValueError("pseudo_label_temperature must be > 0.")
    activity_tensor = torch.tensor([int(x) for x in activity_ids], dtype=torch.long)
    restricted_logits = logits[:, activity_tensor] / float(temperature)
    probs = torch.softmax(restricted_logits, dim=1)
    confidence, local_pred = probs.max(dim=1)
    pseudo_y = activity_tensor[local_pred]
    return pseudo_y, confidence, probs


def _diagonal_gaussian_scores(
    embeddings: torch.Tensor,
    pseudo_y: torch.Tensor,
    activity_ids: Sequence[int],
    regularization: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    scores: list[torch.Tensor] = []
    usable_classes: list[int] = []
    fallback_count = 0
    global_var = embeddings.var(dim=0, unbiased=False).clamp_min(float(regularization))
    for activity_id in activity_ids:
        cls_emb = embeddings[pseudo_y == int(activity_id)]
        if cls_emb.numel() == 0:
            continue
        usable_classes.append(int(activity_id))
        if cls_emb.shape[0] >= 2:
            center = cls_emb.median(dim=0).values
            var = cls_emb.var(dim=0, unbiased=False).clamp_min(float(regularization))
        else:
            fallback_count += 1
            center = cls_emb.mean(dim=0)
            var = global_var
        diff = embeddings - center.view(1, -1)
        score = -0.5 * ((diff.pow(2) / var.view(1, -1)) + var.log().view(1, -1)).sum(
            dim=1
        )
        scores.append(score)
    if not scores:
        raise ValueError("MCD pseudo-label refinement has no usable pseudo-classes.")
    return torch.stack(scores, dim=1), {
        "mcd_usable_classes": usable_classes,
        "mcd_diagonal_fallback_classes": int(fallback_count),
    }


def _mcd_scores(
    embeddings: torch.Tensor,
    pseudo_y: torch.Tensor,
    activity_ids: Sequence[int],
    min_samples: int,
    support_fraction: float | None,
    regularization: float,
    diagonal_fallback: bool,
) -> tuple[torch.Tensor, list[int], dict[str, Any]]:
    feature_dim = int(embeddings.shape[1])
    scores: list[torch.Tensor] = []
    usable_classes: list[int] = []
    full_mcd_classes = 0
    fallback_classes = 0
    emb_np = embeddings.numpy()
    for activity_id in activity_ids:
        cls_emb = embeddings[pseudo_y == int(activity_id)]
        n_samples = int(cls_emb.shape[0])
        if n_samples == 0:
            continue
        can_fit_mcd = MinCovDet is not None and n_samples >= max(
            int(min_samples), feature_dim + 2
        )
        if can_fit_mcd:
            try:
                estimator = MinCovDet(
                    support_fraction=support_fraction,
                    random_state=0,
                ).fit(cls_emb.numpy())
                center = estimator.location_
                cov = estimator.covariance_
                cov = cov + float(regularization) * np.eye(cov.shape[0])
                precision = np.linalg.pinv(cov)
                diff = emb_np - center.reshape(1, -1)
                mahal = np.einsum("nd,dd,nd->n", diff, precision, diff)
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    logdet = float(
                        np.log(np.maximum(np.diag(cov), regularization)).sum()
                    )
                score_np = -0.5 * (mahal + logdet)
                scores.append(torch.from_numpy(score_np).to(dtype=embeddings.dtype))
                usable_classes.append(int(activity_id))
                full_mcd_classes += 1
                continue
            except Exception:
                pass
        if not diagonal_fallback:
            continue
        diagonal_scores, diag_info = _diagonal_gaussian_scores(
            embeddings,
            pseudo_y,
            [int(activity_id)],
            regularization,
        )
        scores.append(diagonal_scores[:, 0])
        usable_classes.append(int(activity_id))
        fallback_classes += int(diag_info["mcd_diagonal_fallback_classes"]) or 1
    if not scores:
        raise ValueError(
            "MCD pseudo-label refinement could not fit any class. "
            "Try pseudo_labeler='classifier' or enable mcd_diagonal_fallback."
        )
    diagnostics = {
        "mcd_full_covariance_classes": int(full_mcd_classes),
        "mcd_fallback_classes": int(fallback_classes),
        "mcd_available": bool(MinCovDet is not None),
    }
    return torch.stack(scores, dim=1), usable_classes, diagnostics


def _refine_pseudo_labels_with_mcd(
    embeddings: torch.Tensor,
    initial_pseudo_y: torch.Tensor,
    activity_ids: Sequence[int],
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    scores, usable_classes, diagnostics = _mcd_scores(
        embeddings,
        initial_pseudo_y,
        activity_ids,
        config.mcd_min_samples,
        config.mcd_support_fraction,
        config.mcd_regularization,
        config.mcd_diagonal_fallback,
    )
    probs = torch.softmax(scores / float(config.pseudo_label_temperature), dim=1)
    confidence, local_pred = probs.max(dim=1)
    class_tensor = torch.tensor(usable_classes, dtype=torch.long)
    pseudo_y = class_tensor[local_pred]
    diagnostics["mcd_usable_classes"] = [int(x) for x in usable_classes]
    return pseudo_y, confidence, diagnostics


def _build_pda_prototypes(
    support_emb: torch.Tensor,
    support_logits: torch.Tensor,
    activity_ids: Sequence[int],
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if support_emb.numel() == 0:
        raise ValueError("PDA needs at least one unlabeled support embedding.")
    if config.pseudo_labeler not in {"classifier", "mcd"}:
        raise ValueError("pseudo_labeler must be 'classifier' or 'mcd'.")
    initial_pseudo_y, initial_confidence, _initial_probs = _classifier_pseudo_labels(
        support_logits,
        activity_ids,
        config.pseudo_label_temperature,
    )
    pseudo_y = initial_pseudo_y
    confidence = initial_confidence
    refinement_diagnostics: dict[str, Any] = {}
    if config.pseudo_labeler == "mcd":
        pseudo_y, confidence, refinement_diagnostics = _refine_pseudo_labels_with_mcd(
            support_emb,
            initial_pseudo_y,
            activity_ids,
            config,
        )

    prototypes: list[torch.Tensor] = []
    class_labels: list[int] = []
    pseudo_counts: dict[int, int] = {}
    for activity_id in activity_ids:
        mask = pseudo_y == int(activity_id)
        count = int(mask.sum().item())
        pseudo_counts[int(activity_id)] = count
        if count == 0:
            continue
        cls_emb = support_emb[mask]
        if config.confidence_weighting:
            weights = confidence[mask].clamp_min(1e-12)
            proto = (cls_emb * weights.view(-1, 1)).sum(dim=0) / weights.sum()
        else:
            proto = cls_emb.mean(dim=0)
        prototypes.append(proto)
        class_labels.append(int(activity_id))
    if not prototypes:
        raise ValueError("PDA pseudo-labeling produced no prototypes.")
    prototype_tensor = torch.stack(prototypes, dim=0)
    if config.normalize_embeddings:
        prototype_tensor = F.normalize(prototype_tensor, p=2, dim=1)
    class_tensor = torch.tensor(class_labels, dtype=torch.long)
    missing = [int(x) for x in activity_ids if int(x) not in set(class_labels)]
    diagnostics = {
        "pseudo_labeler": config.pseudo_labeler,
        "num_prototypes": int(len(class_labels)),
        "prototype_class_labels": [int(x) for x in class_labels],
        "missing_prototype_classes": missing,
        "pseudo_label_counts": pseudo_counts,
        "pseudo_label_confidence_mean": float(confidence.mean().item()),
        "pseudo_label_confidence_min": float(confidence.min().item()),
        "pseudo_label_confidence_max": float(confidence.max().item()),
        "prototype_euclidean_norm_mean": float(
            prototype_tensor.norm(p=2, dim=1).mean().item()
        ),
        "initial_pseudo_label_confidence_mean": float(initial_confidence.mean().item()),
    }
    diagnostics.update(refinement_diagnostics)
    return prototype_tensor, class_tensor, diagnostics


def _restricted_classifier_predict(
    logits: torch.Tensor,
    activity_ids: Sequence[int],
) -> np.ndarray:
    activity_tensor = torch.tensor([int(x) for x in activity_ids], dtype=torch.long)
    local_pred = logits[:, activity_tensor].argmax(dim=1)
    return activity_tensor[local_pred].numpy()


def _pda_episode_predict(
    embeddings_by_window: dict[int, torch.Tensor],
    logits_by_window: dict[int, torch.Tensor],
    support_indices: Sequence[int],
    query_indices: Sequence[int],
    activity_ids: Sequence[int],
    config: Config,
) -> tuple[np.ndarray, dict[str, Any]]:
    support_emb = torch.stack(
        [embeddings_by_window[int(idx)] for idx in support_indices], dim=0
    )
    support_logits = torch.stack(
        [logits_by_window[int(idx)] for idx in support_indices], dim=0
    )
    query_emb = torch.stack(
        [embeddings_by_window[int(idx)] for idx in query_indices], dim=0
    )
    center_shift_norm = 0.0
    if config.center_support_query:
        support_center = support_emb.mean(dim=0)
        support_emb = support_emb - support_center
        query_emb = query_emb - support_center
        center_shift_norm = float(support_center.norm(p=2).item())
    prototypes, prototype_class_labels, diagnostics = _build_pda_prototypes(
        support_emb,
        support_logits,
        activity_ids,
        config,
    )
    logits = prototype_logits(
        query_emb,
        prototypes,
        config.cosine_temperature,
        "cosine",
    )
    local_pred = logits.argmax(dim=1)
    pred = prototype_class_labels[local_pred].numpy()
    diagnostics["center_shift_norm"] = center_shift_norm
    return pred, diagnostics


def _plot_k_curve(overall_df: pd.DataFrame, out_path: Path) -> None:
    if overall_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(
        overall_df["k"],
        overall_df["macro_f1_mean"],
        yerr=overall_df["macro_f1_subject_ci95"],
        marker="o",
        linewidth=2,
        capsize=4,
        label="PDA support prototypes",
    )
    ax.set_xlabel("K shots per activity")
    ax.set_ylabel("Macro F1")
    ax.set_title("PDA Prototype LOSO Evaluation")
    ax.set_xticks(overall_df["k"].tolist())
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)
    output_root = resolve_output_root(config.output_root, config.dataset_id)
    ce_stage_dir = output_root / config.ce_stage_name
    active_embedding_transform = _active_embedding_transform_name(config)
    eval_stage_parts = [config.eval_stage_name, "ce_backbone", "cosine"]
    if active_embedding_transform != "none":
        eval_stage_parts.append(_artifact_safe_name(active_embedding_transform))
    eval_stage_parts.append("l2norm" if config.normalize_embeddings else "raw")
    eval_stage_parts.append(config.pseudo_labeler)
    if config.confidence_weighting:
        eval_stage_parts.append("confidence_weighted")
    if config.center_support_query:
        eval_stage_parts.append("centered")
    eval_stage_name = "_".join(eval_stage_parts)
    eval_dir = output_root / eval_stage_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = WHARDatasetID(config.dataset_id)
    cfg = prepare_cfg(
        dataset_id=dataset_id,
        datasets_dir=Path(config.datasets_dir),
        selected_activities=config.selected_activities,
        window_overlap=config.window_overlap,
    )
    pre = PreProcessingPipeline(cfg)
    _raw_df, session_df, window_df = pre.run()
    reconcile_activity_config(cfg, session_df)
    shared_cfg = SharedConfig(
        dataset_id=config.dataset_id,
        datasets_dir=config.datasets_dir,
        selected_activities=config.selected_activities,
        window_overlap=config.window_overlap,
        val_subjects=config.val_subjects,
        test_subjects=config.test_subjects,
        seed=config.seed,
        split_strategy=config.split_strategy,
        val_percentage=config.val_percentage,
    )
    manifest_path = output_root / "shared_splits" / "loso_subject_folds.json"
    folds = build_or_load_loso_folds(session_df, window_df, shared_cfg, manifest_path)
    if config.max_folds is not None:
        folds = folds[: int(config.max_folds)]

    all_trial_rows: list[dict[str, Any]] = []
    subject_k_rows: list[dict[str, Any]] = []
    confusion_accumulator: dict[int, np.ndarray] = {}
    skipped_folds: list[str] = []
    labels_all = list(range(int(cfg.num_of_activities)))

    for fold in folds:
        ckpt_path = ce_stage_dir / fold.fold_id / "best_model_with_meta.pt"
        if not ckpt_path.exists():
            if config.skip_missing_folds:
                skipped_folds.append(fold.fold_id)
                print(f"[{fold.fold_id}] skipping missing checkpoint: {ckpt_path}")
                continue
            raise FileNotFoundError(f"Missing CE checkpoint: {ckpt_path}")

        split = split_indices_for_fold(session_df, window_df, fold)
        fold_fp = config_fingerprint(
            {
                "stage": config.eval_stage_name,
                "resolved_eval_stage": eval_stage_name,
                "config": asdict(config),
                "shared_cfg": asdict(shared_cfg),
                "fold": asdict(fold),
                "backbone_checkpoint": str(ckpt_path),
                "paper": "Bohdal et al. Feed-Forward Source-Free Domain Adaptation via Class Prototypes",
                "num_classes": int(cfg.num_of_activities),
                "class_names": class_names(cfg),
            }
        )
        split_dir = eval_dir / fold.fold_id
        split_dir.mkdir(parents=True, exist_ok=True)

        loader = build_loader(cfg, session_df, pre, window_df, split.train_indices)
        test_ds = WindowDataset(loader, split.test_indices)
        test_loader = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        model, _checkpoint = load_ce_backbone(ckpt_path, device)
        test_emb, test_logits, test_y, _test_subjects, embedding_transform = (
            _extract_features_logits(model, test_loader, device, config)
        )
        if embedding_transform != active_embedding_transform:
            raise RuntimeError(
                "Active embedding transform changed unexpectedly: "
                f"{active_embedding_transform} -> {embedding_transform}."
            )
        test_class_ids = {int(x) for x in torch.unique(test_y).tolist()}
        embeddings_by_window = {
            int(window_idx): test_emb[pos]
            for pos, window_idx in enumerate(test_ds.indices)
        }
        logits_by_window = {
            int(window_idx): test_logits[pos]
            for pos, window_idx in enumerate(test_ds.indices)
        }
        by_activity = indices_by_activity(loader, split.test_indices)
        by_activity_session = (
            _indices_by_activity_session(loader, split.test_indices)
            if config.support_query_session_disjoint
            else None
        )

        support_k_values = [int(k) for k in config.k_values if int(k) > 0]
        eligible_activity_ids_by_k: dict[int, list[int]] = {}
        for candidate_k in support_k_values:
            candidate_activity_ids = (
                _eligible_activities_session_disjoint(
                    by_activity_session or {},
                    candidate_k,
                    config.min_query_per_class,
                )
                if config.support_query_session_disjoint
                else _eligible_activities(
                    by_activity,
                    candidate_k,
                    config.min_query_per_class,
                )
            )
            eligible_activity_ids_by_k[int(candidate_k)] = [
                int(x) for x in candidate_activity_ids if int(x) in test_class_ids
            ]
        feasible_common_k_values = [
            int(k)
            for k, values in eligible_activity_ids_by_k.items()
            if len(values) >= int(config.min_activities_per_episode)
        ]
        effective_common_k = (
            max(feasible_common_k_values) if feasible_common_k_values else 0
        )
        common_activity_ids: list[int] = []
        if support_k_values and config.require_all_k_activities:
            common_activity_ids = eligible_activity_ids_by_k.get(
                int(effective_common_k), []
            )
            skipped_common_ks = [
                int(k)
                for k in support_k_values
                if int(k) > int(effective_common_k)
                or len(eligible_activity_ids_by_k.get(int(k), []))
                < int(config.min_activities_per_episode)
            ]
            if skipped_common_ks:
                print(
                    f"[{fold.fold_id}] skipping ineligible k values "
                    f"{skipped_common_ks}; largest feasible shared-activity k is "
                    f"{effective_common_k} with activities {common_activity_ids}."
                )
        if (
            support_k_values
            and config.require_all_k_activities
            and not common_activity_ids
            and not config.skip_ineligible_k_values
        ):
            raise ValueError(
                f"{fold.fold_id} has only {len(common_activity_ids)} eligible "
                f"activities for max k={max(support_k_values)}: {common_activity_ids}."
            )

        fold_rows: list[dict[str, Any]] = []
        fold_summary_by_k: list[dict[str, Any]] = []
        for k in config.k_values:
            k = int(k)
            if k < 0:
                raise ValueError(f"k_values must be >= 0, got {k}.")
            if (
                k > 0
                and config.require_all_k_activities
                and k > int(effective_common_k)
            ):
                if config.skip_ineligible_k_values:
                    print(
                        f"[{fold.fold_id}] skipping k={k}: largest feasible "
                        f"shared-activity k is {effective_common_k}."
                    )
                    continue
                raise ValueError(
                    f"{fold.fold_id} k={k} exceeds largest feasible "
                    f"shared-activity k={effective_common_k}."
                )
            activity_ids = (
                sorted(int(x) for x in test_class_ids)
                if k == 0
                else (
                    common_activity_ids
                    if config.require_all_k_activities
                    else eligible_activity_ids_by_k.get(int(k), [])
                )
            )
            if not activity_ids:
                if config.skip_missing_folds or config.skip_ineligible_k_values:
                    print(f"[{fold.fold_id}] skipping k={k}: no eligible activities.")
                    continue
                raise ValueError(
                    f"{fold.fold_id} has no eligible activities for k={k}."
                )
            if len(activity_ids) < int(config.min_activities_per_episode):
                if config.skip_ineligible_k_values:
                    print(
                        f"[{fold.fold_id}] skipping k={k}: only "
                        f"{len(activity_ids)} eligible activities {activity_ids}."
                    )
                    continue
                raise ValueError(
                    f"{fold.fold_id} k={k} has only {len(activity_ids)} eligible "
                    f"activities: {activity_ids}."
                )

            trial_f1s: list[float] = []
            trial_weighted_f1s: list[float] = []
            trial_accs: list[float] = []
            trial_num_prototypes: list[float] = []
            trial_confidences: list[float] = []
            confusion_accumulator.setdefault(
                k,
                np.zeros(
                    (int(cfg.num_of_activities), int(cfg.num_of_activities)),
                    dtype=np.int64,
                ),
            )
            episode_count = 1 if k == 0 else int(config.episodes_per_k)
            iterator = tqdm(
                range(episode_count),
                desc=f"{fold.fold_id} pda k={k}",
                leave=False,
            )
            for trial in iterator:
                rng = np.random.default_rng(
                    int(config.seed)
                    + 1_000_000 * int(fold.test_subject_ids[0])
                    + 10_000 * k
                    + int(trial)
                )
                episode_session_meta: dict[int, dict[str, Any]] = {}
                if k == 0:
                    support_idx: list[int] = []
                    query_mask = torch.zeros_like(test_y, dtype=torch.bool)
                    for activity_id in activity_ids:
                        query_mask |= test_y == int(activity_id)
                    query_positions = torch.nonzero(query_mask, as_tuple=False).view(-1)
                    query_idx = [
                        int(test_ds.indices[int(pos)])
                        for pos in query_positions.tolist()
                    ]
                    query_y = test_y[query_positions].numpy()
                    pred = _restricted_classifier_predict(
                        test_logits[query_positions],
                        activity_ids,
                    )
                    diagnostics = {
                        "num_prototypes": 0,
                        "prototype_class_labels": [],
                        "missing_prototype_classes": [int(x) for x in activity_ids],
                        "pseudo_label_counts": {},
                        "pseudo_label_confidence_mean": 0.0,
                        "pseudo_label_confidence_min": 0.0,
                        "pseudo_label_confidence_max": 0.0,
                        "prototype_euclidean_norm_mean": 0.0,
                        "initial_pseudo_label_confidence_mean": 0.0,
                        "center_shift_norm": 0.0,
                    }
                elif config.support_query_session_disjoint:
                    (
                        support_idx,
                        query_idx,
                        _support_y,
                        query_y,
                        episode_session_meta,
                    ) = _sample_episode_session_disjoint(
                        by_activity_session or {},
                        activity_ids,
                        k,
                        config.min_query_per_class,
                        rng,
                    )
                    pred, diagnostics = _pda_episode_predict(
                        embeddings_by_window,
                        logits_by_window,
                        support_idx,
                        query_idx,
                        activity_ids,
                        config,
                    )
                else:
                    support_idx, query_idx, _support_y, query_y = _sample_episode(
                        by_activity,
                        activity_ids,
                        k,
                        rng,
                    )
                    pred, diagnostics = _pda_episode_predict(
                        embeddings_by_window,
                        logits_by_window,
                        support_idx,
                        query_idx,
                        activity_ids,
                        config,
                    )

                macro_f1 = float(
                    f1_score(
                        query_y,
                        pred,
                        labels=[int(x) for x in activity_ids],
                        average="macro",
                        zero_division=0,
                    )
                )
                acc = float(accuracy_score(query_y, pred))
                weighted_f1 = float(
                    f1_score(
                        query_y,
                        pred,
                        labels=[int(x) for x in activity_ids],
                        average="weighted",
                        zero_division=0,
                    )
                )
                trial_f1s.append(macro_f1)
                trial_weighted_f1s.append(weighted_f1)
                trial_accs.append(acc)
                trial_num_prototypes.append(float(diagnostics["num_prototypes"]))
                trial_confidences.append(
                    float(diagnostics["pseudo_label_confidence_mean"])
                )
                confusion_accumulator[k] += confusion_matrix(
                    query_y, pred, labels=labels_all
                )
                row = {
                    "fold_id": fold.fold_id,
                    "test_subject_id": int(fold.test_subject_ids[0]),
                    "k": int(k),
                    "trial": int(trial),
                    "activity_ids": json.dumps([int(x) for x in activity_ids]),
                    "num_activities": int(len(activity_ids)),
                    "num_support": int(len(support_idx)),
                    "num_query": int(len(query_idx)),
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                    "accuracy": acc,
                    "backbone_source": "ce",
                    "backbone_checkpoint": str(ckpt_path),
                    "distance_metric": "cosine",
                    "embedding_space": "backbone",
                    "embedding_transform": config.embedding_transform,
                    "active_embedding_transform": active_embedding_transform,
                    "power_transform_exponent": float(config.power_transform_exponent),
                    "normalize_embeddings": bool(config.normalize_embeddings),
                    "cosine_temperature": float(config.cosine_temperature),
                    "pseudo_labeler": config.pseudo_labeler,
                    "pseudo_label_temperature": float(config.pseudo_label_temperature),
                    "confidence_weighting": bool(config.confidence_weighting),
                    "center_support_query": bool(config.center_support_query),
                    "num_prototypes": int(diagnostics["num_prototypes"]),
                    "prototype_class_labels": json.dumps(
                        [int(x) for x in diagnostics["prototype_class_labels"]]
                    ),
                    "missing_prototype_classes": json.dumps(
                        [int(x) for x in diagnostics["missing_prototype_classes"]]
                    ),
                    "pseudo_label_counts": json.dumps(
                        {
                            str(int(cls)): int(count)
                            for cls, count in diagnostics["pseudo_label_counts"].items()
                        }
                    ),
                    "pseudo_label_confidence_mean": float(
                        diagnostics["pseudo_label_confidence_mean"]
                    ),
                    "pseudo_label_confidence_min": float(
                        diagnostics["pseudo_label_confidence_min"]
                    ),
                    "pseudo_label_confidence_max": float(
                        diagnostics["pseudo_label_confidence_max"]
                    ),
                    "initial_pseudo_label_confidence_mean": float(
                        diagnostics["initial_pseudo_label_confidence_mean"]
                    ),
                    "prototype_euclidean_norm_mean": float(
                        diagnostics["prototype_euclidean_norm_mean"]
                    ),
                    "center_shift_norm": float(diagnostics["center_shift_norm"]),
                    "support_query_session_disjoint": bool(
                        config.support_query_session_disjoint
                    ),
                    "episode_session_meta": json.dumps(episode_session_meta),
                    "is_no_support_baseline": bool(k == 0),
                }
                for key in (
                    "mcd_available",
                    "mcd_full_covariance_classes",
                    "mcd_fallback_classes",
                    "mcd_usable_classes",
                ):
                    if key in diagnostics:
                        value = diagnostics[key]
                        row[key] = (
                            json.dumps(value) if isinstance(value, list) else value
                        )
                all_trial_rows.append(row)
                fold_rows.append(row)

            f1_mean, f1_std, f1_ci95 = mean_std_ci(trial_f1s)
            weighted_f1_mean, weighted_f1_std, weighted_f1_ci95 = mean_std_ci(
                trial_weighted_f1s
            )
            acc_mean, acc_std, acc_ci95 = mean_std_ci(trial_accs)
            proto_count_mean, proto_count_std, _ = mean_std_ci(trial_num_prototypes)
            conf_mean, conf_std, _ = mean_std_ci(trial_confidences)
            summary_row = {
                "test_subject_id": int(fold.test_subject_ids[0]),
                "k": int(k),
                "episodes": int(episode_count),
                "activity_ids": [int(x) for x in activity_ids],
                "distance_metric": "cosine",
                "active_embedding_transform": active_embedding_transform,
                "normalize_embeddings": bool(config.normalize_embeddings),
                "pseudo_labeler": config.pseudo_labeler,
                "confidence_weighting": bool(config.confidence_weighting),
                "center_support_query": bool(config.center_support_query),
                "support_query_session_disjoint": bool(
                    config.support_query_session_disjoint
                ),
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
                "macro_f1_ci95": f1_ci95,
                "weighted_f1_mean": weighted_f1_mean,
                "weighted_f1_std": weighted_f1_std,
                "weighted_f1_ci95": weighted_f1_ci95,
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "accuracy_ci95": acc_ci95,
                "num_prototypes_mean": proto_count_mean,
                "num_prototypes_std": proto_count_std,
                "pseudo_label_confidence_mean": conf_mean,
                "pseudo_label_confidence_std": conf_std,
            }
            fold_summary_by_k.append(summary_row)
            subject_k_rows.append(
                {
                    "fold_id": fold.fold_id,
                    **summary_row,
                }
            )
            print(
                f"[{fold.fold_id}] pda k={k} "
                f"macro_f1={f1_mean:.4f}±{f1_std:.4f} "
                f"accuracy={acc_mean:.4f}±{acc_std:.4f}"
            )

        fold_trial_csv = split_dir / "trial_results.csv"
        fold_summary_csv = split_dir / "summary_by_k.csv"
        pd.DataFrame(fold_rows).to_csv(fold_trial_csv, index=False)
        pd.DataFrame(fold_summary_by_k).to_csv(fold_summary_csv, index=False)
        (split_dir / "fold_metrics.json").write_text(
            json.dumps(
                {
                    "config_fingerprint": fold_fp,
                    "fold_id": fold.fold_id,
                    "test_subject_id": int(fold.test_subject_ids[0]),
                    "backbone_source": "ce",
                    "backbone_checkpoint": str(ckpt_path),
                    "distance_metric": "cosine",
                    "embedding_space": "backbone",
                    "embedding_transform": config.embedding_transform,
                    "active_embedding_transform": active_embedding_transform,
                    "normalize_embeddings": bool(config.normalize_embeddings),
                    "cosine_temperature": float(config.cosine_temperature),
                    "pseudo_labeler": config.pseudo_labeler,
                    "pseudo_label_temperature": float(config.pseudo_label_temperature),
                    "confidence_weighting": bool(config.confidence_weighting),
                    "center_support_query": bool(config.center_support_query),
                    "support_query_session_disjoint": bool(
                        config.support_query_session_disjoint
                    ),
                    "trial_results_csv": str(fold_trial_csv),
                    "summary_by_k_csv": str(fold_summary_csv),
                    "summary_by_k": fold_summary_by_k,
                    "rows": fold_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    trial_df = pd.DataFrame(all_trial_rows)
    subject_k_df = pd.DataFrame(subject_k_rows)
    if subject_k_df.empty:
        raise RuntimeError("No PDA evaluation rows were produced.")

    subject_summary_rows: list[dict[str, Any]] = []
    for (subject_id, k), group in trial_df.groupby(["test_subject_id", "k"], sort=True):
        row: dict[str, Any] = {
            "test_subject_id": int(subject_id),
            "k": int(k),
            "num_trials": int(group.shape[0]),
        }
        for metric in ("macro_f1", "weighted_f1", "accuracy"):
            mean, std, ci95 = mean_std_ci(group[metric].astype(float).tolist())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95
        subject_summary_rows.append(row)
    subject_summary_df = pd.DataFrame(subject_summary_rows).sort_values(
        ["test_subject_id", "k"]
    )

    overall_rows: list[dict[str, Any]] = []
    for k, group in subject_k_df.groupby("k", sort=True):
        row: dict[str, Any] = {"k": int(k), "num_subjects": int(group.shape[0])}
        for metric in ("macro_f1_mean", "accuracy_mean"):
            mean, std, ci95 = mean_std_ci(group[metric].astype(float).tolist())
            row[metric] = mean
            row[metric.replace("_mean", "_subject_std")] = std
            row[metric.replace("_mean", "_subject_ci95")] = ci95
        trial_group = trial_df[trial_df["k"].astype(int) == int(k)]
        row["num_trials"] = int(trial_group.shape[0])
        for metric in ("macro_f1", "weighted_f1", "accuracy"):
            mean, std, ci95 = mean_std_ci(trial_group[metric].astype(float).tolist())
            row[f"{metric}_trial_mean"] = mean
            row[f"{metric}_trial_std"] = std
            row[f"{metric}_trial_ci95"] = ci95
        overall_rows.append(row)
    overall_df = pd.DataFrame(overall_rows).sort_values("k")

    trial_csv = eval_dir / "trial_results.csv"
    subject_csv = eval_dir / "subject_by_k_results.csv"
    subject_summary_csv = eval_dir / "subject_episode_summary_by_k.csv"
    overall_csv = eval_dir / "overall_by_k_results.csv"
    trial_df.to_csv(trial_csv, index=False)
    subject_k_df.to_csv(subject_csv, index=False)
    subject_summary_df.to_csv(subject_summary_csv, index=False)
    overall_df.to_csv(overall_csv, index=False)
    plot_path = eval_dir / "k_shot_curve.png"
    _plot_k_curve(overall_df, plot_path)

    confusion_paths: list[str] = []
    for k, matrix in sorted(confusion_accumulator.items()):
        cm_path = eval_dir / f"confusion_pda_support_prototypes_k{k}.png"
        save_confusion_matrix_plot(
            matrix,
            class_names(cfg),
            cm_path,
            title=f"PDA Support Prototypes Confusion Matrix (K={k})",
        )
        confusion_paths.append(str(cm_path))

    summary = {
        "config": asdict(config),
        "paper": {
            "title": "Feed-Forward Source-Free Domain Adaptation via Class Prototypes",
            "arxiv": "2307.10787",
            "method": "PDA confidence-weighted pseudo-labeled target prototypes",
        },
        "splits_manifest_path": str(manifest_path),
        "eval_stage_name": eval_stage_name,
        "eval_dir": str(eval_dir),
        "ce_stage_dir": str(ce_stage_dir),
        "backbone_source": "ce",
        "distance_metric": "cosine",
        "embedding_space": "backbone",
        "embedding_transform": config.embedding_transform,
        "active_embedding_transform": active_embedding_transform,
        "normalize_embeddings": bool(config.normalize_embeddings),
        "cosine_temperature": float(config.cosine_temperature),
        "pseudo_labeler": config.pseudo_labeler,
        "pseudo_label_temperature": float(config.pseudo_label_temperature),
        "confidence_weighting": bool(config.confidence_weighting),
        "center_support_query": bool(config.center_support_query),
        "skipped_folds": skipped_folds,
        "trial_results_csv": str(trial_csv),
        "subject_by_k_results_csv": str(subject_csv),
        "subject_episode_summary_by_k_csv": str(subject_summary_csv),
        "overall_by_k_results_csv": str(overall_csv),
        "plot_path": str(plot_path),
        "confusion_matrix_paths": confusion_paths,
        "overall_by_k": overall_df.to_dict(orient="records"),
        "summary_by_k": [
            {
                "k": int(row["k"]),
                "macro_f1_mean": float(row["macro_f1_trial_mean"]),
                "macro_f1_std": float(row["macro_f1_trial_std"]),
                "weighted_f1_mean": float(row["weighted_f1_trial_mean"]),
                "weighted_f1_std": float(row["weighted_f1_trial_std"]),
                "accuracy_mean": float(row["accuracy_trial_mean"]),
                "accuracy_std": float(row["accuracy_trial_std"]),
                "num_trials": int(row["num_trials"]),
                "num_subjects": int(row["num_subjects"]),
            }
            for row in overall_df.to_dict(orient="records")
        ],
        "summary_by_subject": subject_summary_df.to_dict(orient="records"),
        "num_trial_rows": int(trial_df.shape[0]),
        "num_subject_k_rows": int(subject_k_df.shape[0]),
        "num_subject_summary_rows": int(subject_summary_df.shape[0]),
    }
    (eval_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

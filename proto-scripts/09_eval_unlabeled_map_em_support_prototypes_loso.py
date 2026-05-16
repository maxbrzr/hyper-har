import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from common import (
    ROOT,
    SharedConfig,
    WindowDataset,
    build_loader,
    build_or_load_loso_folds,
    class_names,
    config_fingerprint,
    extract_supcon_embeddings,
    indices_by_activity,
    load_ce_backbone,
    mean_std_ci,
    prepare_cfg,
    prototype_logits,
    resolve_distance_metric,
    save_confusion_matrix_plot,
    set_seed,
    split_indices_for_fold,
)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from whar_datasets import PreProcessingPipeline, WHARDatasetID


@dataclass(frozen=True)
class Config:
    dataset_id: str = WHARDatasetID.WEAR.value
    datasets_dir: str = str("datasets")
    selected_activities: list[str] | None = None
    window_overlap: float = 0.0  # 0.5
    val_subjects: int = 3
    test_subjects: int = 1
    seed: int = 0

    k_values: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 16, 32)
    episodes_per_k: int = 100
    min_query_per_class: int = 1
    support_query_session_disjoint: bool = False
    min_activities_per_episode: int = 2
    require_all_k_activities: bool = True
    batch_size: int = 256
    num_workers: int = 0
    euclidean_temperature: float = 1.0
    normalize_embeddings: bool | None = None
    embedding_space: str = "backbone"  # "projected" or "backbone"
    backbone_source: str = "ce"
    distance_metric: str = "auto"  # "auto", "cosine", or "euclidean"
    embedding_transform: str = "none"  # "signed_power"  # "none" or "signed_power"
    power_transform_exponent: float = 0.5
    power_transform_backbone_sources: tuple[str, ...] = ("ce",)
    prior_variance_floor: float = 1e-4
    normalize_prior_mean_for_update: bool | None = None
    project_posterior_to_sphere: bool | None = None
    em_iterations: int = 10  # 3
    em_temperature: float = 1.0  # 2.0  #
    em_likelihood_variance: float | None = None
    em_min_soft_count: float = 1e-6
    em_uniform_class_prior: bool = True
    skip_missing_folds: bool = False
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_root: str = str(ROOT / "artifacts" / "proto_pipeline")
    ce_stage_name: str = "01_tinierhar_ce_loso"
    eval_stage_name: str = "09_unlabeled_map_em_support_prototypes_loso"
    separate_backbone_source_dir: bool = True
    max_folds: int | None = None
    force_rerun: bool = False


RUN_CONFIG = Config()


def _active_embedding_transform_name(config: Config, backbone_source: str) -> str:
    if config.embedding_transform == "none":
        return "none"
    if config.embedding_transform != "signed_power":
        raise ValueError("embedding_transform must be 'none' or 'signed_power'.")
    sources = {str(source) for source in config.power_transform_backbone_sources}
    if sources and str(backbone_source) not in sources:
        return "none"
    return f"signed_power_{float(config.power_transform_exponent):g}"


def _artifact_safe_name(value: str) -> str:
    return value.replace(".", "p").replace("-", "m").replace("+", "p").replace(" ", "_")


def _apply_embedding_transform(
    embeddings: torch.Tensor,
    config: Config,
    backbone_source: str,
) -> tuple[torch.Tensor, str]:
    transform_name = _active_embedding_transform_name(config, backbone_source)
    if transform_name == "none":
        return embeddings, transform_name
    exponent = float(config.power_transform_exponent)
    if exponent <= 0:
        raise ValueError("power_transform_exponent must be > 0.")
    transformed = torch.sign(embeddings) * torch.abs(embeddings).pow(exponent)
    return transformed, transform_name


def _resolve_optional_bool(value: bool | None, default: bool) -> bool:
    return bool(default) if value is None else bool(value)


@torch.no_grad()
def _extract_preprocessed_embeddings(
    backbone: torch.nn.Module,
    projection_head: torch.nn.Module | None,
    dataloader: DataLoader,
    device: torch.device,
    config: Config,
    backbone_source: str,
    normalize_embeddings: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    embeddings, labels, subjects = extract_supcon_embeddings(
        backbone,
        projection_head,
        dataloader,
        device,
        normalize=False,
    )
    embeddings, transform_name = _apply_embedding_transform(
        embeddings, config, backbone_source
    )
    if normalize_embeddings:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings, labels, subjects, transform_name


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


def _class_diagonal_gaussian_stats(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    variance_floor: float,
    normalize_means: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    means: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    for cls in range(int(num_classes)):
        mask = labels == int(cls)
        if not bool(mask.any()):
            raise ValueError(f"Cannot build prior for missing class {cls}.")
        cls_emb = embeddings[mask]
        means.append(cls_emb.mean(dim=0))
        if cls_emb.shape[0] > 1:
            var = cls_emb.var(dim=0, unbiased=True)
        else:
            var = torch.zeros(cls_emb.shape[1], dtype=cls_emb.dtype)
        variances.append(var.clamp_min(float(variance_floor)))
    mean_tensor = torch.stack(means, dim=0)
    if normalize_means:
        mean_tensor = F.normalize(mean_tensor, p=2, dim=1)
    return mean_tensor, torch.stack(variances, dim=0)


def _em_likelihood_variance(config: Config) -> float:
    if config.em_likelihood_variance is not None:
        variance = float(config.em_likelihood_variance)
    else:
        variance = float(config.em_temperature) / 2.0
    if variance <= 0:
        raise ValueError("EM likelihood variance must be > 0.")
    return variance


def _map_em_update_prototypes(
    prior_means: torch.Tensor,
    prior_variances: torch.Tensor,
    support_emb: torch.Tensor,
    activity_ids: Sequence[int],
    prior_variance_floor: float,
    em_iterations: int,
    em_temperature: float,
    em_likelihood_variance: float,
    em_min_soft_count: float,
    uniform_class_prior: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if support_emb.numel() == 0:
        raise ValueError("MAP-EM needs at least one unlabeled support embedding.")
    if int(em_iterations) < 1:
        raise ValueError("em_iterations must be >= 1.")
    if float(em_temperature) <= 0:
        raise ValueError("em_temperature must be > 0.")
    if float(em_likelihood_variance) <= 0:
        raise ValueError("em_likelihood_variance must be > 0.")
    if float(em_min_soft_count) <= 0:
        raise ValueError("em_min_soft_count must be > 0.")

    activity_tensor = torch.tensor([int(x) for x in activity_ids], dtype=torch.long)
    prior_mu = prior_means[activity_tensor]
    prior_var = prior_variances[activity_tensor].clamp_min(float(prior_variance_floor))
    prototypes = prior_mu.clone()
    class_log_prior = None
    if not uniform_class_prior:
        class_log_prior = torch.zeros(len(activity_ids), dtype=support_emb.dtype)

    responsibilities = torch.empty(
        support_emb.shape[0], len(activity_ids), dtype=support_emb.dtype
    )
    soft_counts = torch.zeros(len(activity_ids), dtype=support_emb.dtype)
    prior_weight_mean = 1.0
    support_weight_mean = 0.0
    for _iteration in range(int(em_iterations)):
        logits = prototype_logits(
            support_emb,
            prototypes,
            float(em_temperature),
            "euclidean",
        )
        if class_log_prior is not None:
            logits = logits + class_log_prior.view(1, -1)
        responsibilities = torch.softmax(logits, dim=1)
        soft_counts = responsibilities.sum(dim=0)
        safe_counts = soft_counts.clamp_min(float(em_min_soft_count))
        soft_means = responsibilities.T @ support_emb
        soft_means = soft_means / safe_counts.view(-1, 1)
        prior_precision = 1.0 / prior_var
        support_precision = safe_counts.view(-1, 1) / float(em_likelihood_variance)
        precision_sum = prior_precision + support_precision
        prototypes = (
            prior_precision * prior_mu + support_precision * soft_means
        ) / precision_sum
        prior_weight = prior_precision / precision_sum
        support_weight = support_precision / precision_sum
        prior_weight_mean = float(prior_weight.mean().item())
        support_weight_mean = float(support_weight.mean().item())

    entropy = -(
        responsibilities * responsibilities.clamp_min(float(em_min_soft_count)).log()
    ).sum(dim=1)
    prototype_shift = (prototypes - prior_mu).norm(p=2, dim=1)
    diagnostics = {
        "posterior_euclidean_norm_mean": float(
            prototypes.norm(p=2, dim=1).mean().item()
        ),
        "prior_weight_mean": prior_weight_mean,
        "support_weight_mean": support_weight_mean,
        "soft_count_min": float(soft_counts.min().item()),
        "soft_count_max": float(soft_counts.max().item()),
        "soft_count_mean": float(soft_counts.mean().item()),
        "responsibility_entropy_mean": float(entropy.mean().item()),
        "responsibility_confidence_mean": float(
            responsibilities.max(dim=1).values.mean().item()
        ),
        "prototype_shift_mean": float(prototype_shift.mean().item()),
        "em_likelihood_variance": float(em_likelihood_variance),
    }
    return prototypes, diagnostics


def _map_em_episode_predict(
    embeddings_by_window: dict[int, torch.Tensor],
    prior_means: torch.Tensor,
    prior_variances: torch.Tensor,
    support_indices: Sequence[int],
    query_indices: Sequence[int],
    activity_ids: Sequence[int],
    query_temperature: float,
    prior_variance_floor: float,
    em_iterations: int,
    em_temperature: float,
    em_likelihood_variance: float,
    em_min_soft_count: float,
    uniform_class_prior: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    support_emb = torch.stack(
        [embeddings_by_window[int(idx)] for idx in support_indices], dim=0
    )
    query_emb = torch.stack(
        [embeddings_by_window[int(idx)] for idx in query_indices], dim=0
    )
    adapted_proto, diagnostics = _map_em_update_prototypes(
        prior_means,
        prior_variances,
        support_emb,
        activity_ids,
        prior_variance_floor,
        em_iterations,
        em_temperature,
        em_likelihood_variance,
        em_min_soft_count,
        uniform_class_prior,
    )
    logits = prototype_logits(query_emb, adapted_proto, query_temperature, "euclidean")
    local_pred = logits.argmax(dim=1).numpy()
    activity_np = np.asarray([int(x) for x in activity_ids], dtype=np.int64)
    return activity_np[local_pred], diagnostics


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
        label="Unlabeled MAP-EM prototypes",
    )
    ax.set_xlabel("K shots per activity")
    ax.set_ylabel("Macro F1")
    ax.set_title("Unlabeled MAP-EM Prototype LOSO Evaluation")
    ax.set_xticks(overall_df["k"].tolist())
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)
    output_root = Path(config.output_root)
    ce_stage_dir = output_root / config.ce_stage_name
    if config.backbone_source != "ce":
        raise ValueError("MAP-EM script currently supports only backbone_source='ce'.")
    active_embedding_transform = _active_embedding_transform_name(
        config, config.backbone_source
    )
    effective_distance_metric = resolve_distance_metric(
        config.distance_metric, config.backbone_source
    )
    if effective_distance_metric != "euclidean":
        raise ValueError("MAP-EM script currently supports only Euclidean distance.")
    spherical_geometry = False
    effective_normalize_embeddings = (
        _resolve_optional_bool(config.normalize_embeddings, spherical_geometry)
        and spherical_geometry
    )
    effective_normalize_prior_mean = _resolve_optional_bool(
        config.normalize_prior_mean_for_update, spherical_geometry
    )
    effective_project_posterior_to_sphere = _resolve_optional_bool(
        config.project_posterior_to_sphere, spherical_geometry
    )
    if effective_project_posterior_to_sphere:
        raise ValueError("MAP-EM CE Euclidean mode must not project to the sphere.")
    em_likelihood_variance = _em_likelihood_variance(config)
    eval_stage_parts = [config.eval_stage_name]
    if config.separate_backbone_source_dir:
        eval_stage_parts.append(f"{config.backbone_source}_backbone")
    eval_stage_parts.append(effective_distance_metric)
    if active_embedding_transform != "none":
        eval_stage_parts.append(_artifact_safe_name(active_embedding_transform))
        eval_stage_parts.append(
            "l2norm" if effective_normalize_embeddings else "rawstats"
        )
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
    shared_cfg = SharedConfig(
        dataset_id=config.dataset_id,
        datasets_dir=config.datasets_dir,
        selected_activities=config.selected_activities,
        window_overlap=config.window_overlap,
        val_subjects=config.val_subjects,
        test_subjects=config.test_subjects,
        seed=config.seed,
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
            raise FileNotFoundError(
                f"Missing {config.backbone_source} checkpoint: {ckpt_path}"
            )

        split = split_indices_for_fold(session_df, window_df, fold)
        fold_fp = config_fingerprint(
            {
                "stage": config.eval_stage_name,
                "resolved_eval_stage": eval_stage_name,
                "config": asdict(config),
                "shared_cfg": asdict(shared_cfg),
                "fold": asdict(fold),
                "backbone_checkpoint": str(ckpt_path),
                "effective_distance_metric": effective_distance_metric,
            }
        )
        split_dir = eval_dir / fold.fold_id
        split_dir.mkdir(parents=True, exist_ok=True)
        fold_metrics_path = split_dir / "fold_metrics.json"

        loader = build_loader(cfg, session_df, pre, window_df, split.train_indices)
        train_ds = WindowDataset(loader, split.train_indices)
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        test_ds = WindowDataset(loader, split.test_indices)
        test_loader = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        backbone, _checkpoint = load_ce_backbone(ckpt_path, device)
        projection_head = None
        effective_embedding_space = "backbone"
        train_emb, train_y, _train_subjects, train_embedding_transform = (
            _extract_preprocessed_embeddings(
                backbone,
                projection_head,
                train_loader,
                device,
                config,
                config.backbone_source,
                effective_normalize_embeddings,
            )
        )
        if train_embedding_transform != active_embedding_transform:
            raise RuntimeError(
                "Active embedding transform changed unexpectedly: "
                f"{active_embedding_transform} -> {train_embedding_transform}."
            )
        prior_means, prior_variances = _class_diagonal_gaussian_stats(
            train_emb,
            train_y,
            num_classes=int(cfg.num_of_activities),
            variance_floor=config.prior_variance_floor,
            normalize_means=effective_normalize_prior_mean,
        )
        test_emb, test_y, _test_subjects, test_embedding_transform = (
            _extract_preprocessed_embeddings(
                backbone,
                projection_head,
                test_loader,
                device,
                config,
                config.backbone_source,
                effective_normalize_embeddings,
            )
        )
        if test_embedding_transform != active_embedding_transform:
            raise RuntimeError(
                "Active embedding transform changed unexpectedly: "
                f"{active_embedding_transform} -> {test_embedding_transform}."
            )
        embeddings_by_window = {
            int(window_idx): test_emb[pos]
            for pos, window_idx in enumerate(test_ds.indices)
        }
        by_activity = indices_by_activity(loader, split.test_indices)
        by_activity_session = (
            _indices_by_activity_session(loader, split.test_indices)
            if config.support_query_session_disjoint
            else None
        )
        support_k_values = [int(k) for k in config.k_values if int(k) > 0]
        max_k = max(support_k_values) if support_k_values else 0
        common_activity_ids: list[int] = []
        if support_k_values:
            common_activity_ids = (
                _eligible_activities_session_disjoint(
                    by_activity_session or {},
                    max_k,
                    config.min_query_per_class,
                )
                if config.support_query_session_disjoint
                else _eligible_activities(
                    by_activity,
                    max_k,
                    config.min_query_per_class,
                )
            )
        if (
            support_k_values
            and config.require_all_k_activities
            and not common_activity_ids
        ):
            raise ValueError(
                f"{fold.fold_id} has no activities with at least "
                f"{max_k + config.min_query_per_class} windows."
            )
        if support_k_values and len(common_activity_ids) < int(
            config.min_activities_per_episode
        ):
            raise ValueError(
                f"{fold.fold_id} has only {len(common_activity_ids)} eligible "
                f"activities for max k={max_k}: {common_activity_ids}. "
                "Prototype evaluation with fewer than "
                f"{config.min_activities_per_episode} activities is trivial. "
                "If support_query_session_disjoint=True, this usually means the "
                "held-out subject does not have multiple sessions for enough "
                "activities. Lower k, disable require_all_k_activities, disable "
                "support_query_session_disjoint, or explicitly lower "
                "min_activities_per_episode."
            )

        fold_rows: list[dict[str, Any]] = []
        fold_summary_by_k: list[dict[str, Any]] = []
        for k in config.k_values:
            k = int(k)
            if k < 0:
                raise ValueError(f"k_values must be >= 0, got {k}.")
            activity_ids = (
                sorted(int(x) for x in torch.unique(test_y).tolist())
                if k == 0
                else (
                    common_activity_ids
                    if config.require_all_k_activities
                    else (
                        _eligible_activities_session_disjoint(
                            by_activity_session or {},
                            k,
                            config.min_query_per_class,
                        )
                        if config.support_query_session_disjoint
                        else _eligible_activities(
                            by_activity,
                            k,
                            config.min_query_per_class,
                        )
                    )
                )
            )
            if not activity_ids:
                if config.skip_missing_folds:
                    continue
                raise ValueError(
                    f"{fold.fold_id} has no eligible activities for k={k}."
                )
            if len(activity_ids) < int(config.min_activities_per_episode):
                raise ValueError(
                    f"{fold.fold_id} k={k} has only {len(activity_ids)} eligible "
                    f"activities: {activity_ids}. Prototype evaluation with fewer "
                    f"than {config.min_activities_per_episode} activities is trivial."
                )
            trial_f1s: list[float] = []
            trial_weighted_f1s: list[float] = []
            trial_accs: list[float] = []
            trial_posterior_norms: list[float] = []
            trial_prior_weights: list[float] = []
            trial_support_weights: list[float] = []
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
                desc=f"{fold.fold_id} map-em k={k}",
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
                    support_idx = []
                    query_idx = [int(x) for x in test_ds.indices]
                    query_y = test_y.numpy()
                    em_diagnostics = {
                        "posterior_euclidean_norm_mean": float(
                            prior_means.norm(p=2, dim=1).mean().item()
                        ),
                        "prior_weight_mean": 1.0,
                        "support_weight_mean": 0.0,
                        "soft_count_min": 0.0,
                        "soft_count_max": 0.0,
                        "soft_count_mean": 0.0,
                        "responsibility_entropy_mean": 0.0,
                        "responsibility_confidence_mean": 0.0,
                        "prototype_shift_mean": 0.0,
                        "em_likelihood_variance": float(em_likelihood_variance),
                    }
                    pred = (
                        prototype_logits(
                            test_emb,
                            prior_means,
                            config.euclidean_temperature,
                            "euclidean",
                        )
                        .argmax(dim=1)
                        .numpy()
                    )
                elif config.support_query_session_disjoint:
                    (
                        support_idx,
                        query_idx,
                        support_y,
                        query_y,
                        episode_session_meta,
                    ) = _sample_episode_session_disjoint(
                        by_activity_session or {},
                        activity_ids,
                        k,
                        config.min_query_per_class,
                        rng,
                    )
                    pred, em_diagnostics = _map_em_episode_predict(
                        embeddings_by_window,
                        prior_means,
                        prior_variances,
                        support_idx,
                        query_idx,
                        activity_ids,
                        config.euclidean_temperature,
                        config.prior_variance_floor,
                        config.em_iterations,
                        config.em_temperature,
                        em_likelihood_variance,
                        config.em_min_soft_count,
                        config.em_uniform_class_prior,
                    )
                else:
                    support_idx, query_idx, support_y, query_y = _sample_episode(
                        by_activity,
                        activity_ids,
                        k,
                        rng,
                    )
                    pred, em_diagnostics = _map_em_episode_predict(
                        embeddings_by_window,
                        prior_means,
                        prior_variances,
                        support_idx,
                        query_idx,
                        activity_ids,
                        config.euclidean_temperature,
                        config.prior_variance_floor,
                        config.em_iterations,
                        config.em_temperature,
                        em_likelihood_variance,
                        config.em_min_soft_count,
                        config.em_uniform_class_prior,
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
                trial_posterior_norms.append(
                    float(em_diagnostics["posterior_euclidean_norm_mean"])
                )
                trial_prior_weights.append(float(em_diagnostics["prior_weight_mean"]))
                trial_support_weights.append(
                    float(em_diagnostics["support_weight_mean"])
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
                    "backbone_source": config.backbone_source,
                    "backbone_checkpoint": str(ckpt_path),
                    "distance_metric": config.distance_metric,
                    "effective_distance_metric": effective_distance_metric,
                    "embedding_space": config.embedding_space,
                    "effective_embedding_space": effective_embedding_space,
                    "embedding_transform": config.embedding_transform,
                    "active_embedding_transform": active_embedding_transform,
                    "power_transform_exponent": float(config.power_transform_exponent),
                    "normalize_embeddings": config.normalize_embeddings,
                    "effective_normalize_embeddings": effective_normalize_embeddings,
                    "prior_variance_floor": float(config.prior_variance_floor),
                    "normalize_prior_mean_for_update": (
                        config.normalize_prior_mean_for_update
                    ),
                    "effective_normalize_prior_mean_for_update": (
                        effective_normalize_prior_mean
                    ),
                    "project_posterior_to_sphere": config.project_posterior_to_sphere,
                    "effective_project_posterior_to_sphere": (
                        effective_project_posterior_to_sphere
                    ),
                    "em_iterations": int(config.em_iterations),
                    "em_temperature": float(config.em_temperature),
                    "em_likelihood_variance": float(em_likelihood_variance),
                    "em_min_soft_count": float(config.em_min_soft_count),
                    "em_uniform_class_prior": bool(config.em_uniform_class_prior),
                    "posterior_euclidean_norm_mean": float(
                        em_diagnostics["posterior_euclidean_norm_mean"]
                    ),
                    "prior_weight_mean": float(em_diagnostics["prior_weight_mean"]),
                    "support_weight_mean": float(em_diagnostics["support_weight_mean"]),
                    "soft_count_min": float(em_diagnostics["soft_count_min"]),
                    "soft_count_max": float(em_diagnostics["soft_count_max"]),
                    "soft_count_mean": float(em_diagnostics["soft_count_mean"]),
                    "responsibility_entropy_mean": float(
                        em_diagnostics["responsibility_entropy_mean"]
                    ),
                    "responsibility_confidence_mean": float(
                        em_diagnostics["responsibility_confidence_mean"]
                    ),
                    "prototype_shift_mean": float(
                        em_diagnostics["prototype_shift_mean"]
                    ),
                    "support_query_session_disjoint": bool(
                        config.support_query_session_disjoint
                    ),
                    "episode_session_meta": json.dumps(episode_session_meta),
                    "is_no_support_baseline": bool(k == 0),
                }
                all_trial_rows.append(row)
                fold_rows.append(row)
            f1_mean, f1_std, f1_ci95 = mean_std_ci(trial_f1s)
            weighted_f1_mean, weighted_f1_std, weighted_f1_ci95 = mean_std_ci(
                trial_weighted_f1s
            )
            acc_mean, acc_std, acc_ci95 = mean_std_ci(trial_accs)
            posterior_norm_mean, posterior_norm_std, _ = mean_std_ci(
                trial_posterior_norms
            )
            prior_weight_mean, prior_weight_std, _ = mean_std_ci(trial_prior_weights)
            support_weight_mean, support_weight_std, _ = mean_std_ci(
                trial_support_weights
            )
            fold_summary_by_k.append(
                {
                    "test_subject_id": int(fold.test_subject_ids[0]),
                    "k": int(k),
                    "episodes": int(episode_count),
                    "activity_ids": [int(x) for x in activity_ids],
                    "distance_metric": config.distance_metric,
                    "effective_distance_metric": effective_distance_metric,
                    "active_embedding_transform": active_embedding_transform,
                    "normalize_embeddings": config.normalize_embeddings,
                    "effective_normalize_embeddings": effective_normalize_embeddings,
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
                    "posterior_euclidean_norm_mean": posterior_norm_mean,
                    "posterior_euclidean_norm_std": posterior_norm_std,
                    "prior_weight_mean": prior_weight_mean,
                    "prior_weight_std": prior_weight_std,
                    "support_weight_mean": support_weight_mean,
                    "support_weight_std": support_weight_std,
                }
            )
            subject_k_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "test_subject_id": int(fold.test_subject_ids[0]),
                    "k": int(k),
                    "episodes": int(episode_count),
                    "activity_ids": [int(x) for x in activity_ids],
                    "distance_metric": config.distance_metric,
                    "effective_distance_metric": effective_distance_metric,
                    "active_embedding_transform": active_embedding_transform,
                    "normalize_embeddings": config.normalize_embeddings,
                    "effective_normalize_embeddings": effective_normalize_embeddings,
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
                    "posterior_euclidean_norm_mean": posterior_norm_mean,
                    "posterior_euclidean_norm_std": posterior_norm_std,
                    "prior_weight_mean": prior_weight_mean,
                    "prior_weight_std": prior_weight_std,
                    "support_weight_mean": support_weight_mean,
                    "support_weight_std": support_weight_std,
                }
            )
            print(
                f"[{fold.fold_id}] map-em k={k} "
                f"macro_f1={f1_mean:.4f}±{f1_std:.4f} "
                f"accuracy={acc_mean:.4f}±{acc_std:.4f}"
            )
        fold_trial_csv = split_dir / "trial_results.csv"
        fold_summary_csv = split_dir / "summary_by_k.csv"
        pd.DataFrame(fold_rows).to_csv(fold_trial_csv, index=False)
        pd.DataFrame(fold_summary_by_k).to_csv(fold_summary_csv, index=False)
        fold_metrics_path.write_text(
            json.dumps(
                {
                    "config_fingerprint": fold_fp,
                    "fold_id": fold.fold_id,
                    "test_subject_id": int(fold.test_subject_ids[0]),
                    "backbone_source": config.backbone_source,
                    "backbone_checkpoint": str(ckpt_path),
                    "distance_metric": config.distance_metric,
                    "effective_distance_metric": effective_distance_metric,
                    "embedding_space": config.embedding_space,
                    "effective_embedding_space": effective_embedding_space,
                    "embedding_transform": config.embedding_transform,
                    "active_embedding_transform": active_embedding_transform,
                    "power_transform_exponent": float(config.power_transform_exponent),
                    "normalize_embeddings": config.normalize_embeddings,
                    "effective_normalize_embeddings": effective_normalize_embeddings,
                    "prior_variance_floor": float(config.prior_variance_floor),
                    "normalize_prior_mean_for_update": (
                        config.normalize_prior_mean_for_update
                    ),
                    "effective_normalize_prior_mean_for_update": (
                        effective_normalize_prior_mean
                    ),
                    "project_posterior_to_sphere": config.project_posterior_to_sphere,
                    "effective_project_posterior_to_sphere": (
                        effective_project_posterior_to_sphere
                    ),
                    "em_iterations": int(config.em_iterations),
                    "em_temperature": float(config.em_temperature),
                    "em_likelihood_variance": float(em_likelihood_variance),
                    "em_min_soft_count": float(config.em_min_soft_count),
                    "em_uniform_class_prior": bool(config.em_uniform_class_prior),
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
        raise RuntimeError("No support-prototype evaluation rows were produced.")

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
        cm_path = eval_dir / f"confusion_unlabeled_map_em_prototypes_k{k}.png"
        save_confusion_matrix_plot(
            matrix,
            class_names(cfg),
            cm_path,
            title=f"Unlabeled MAP-EM Prototypes Confusion Matrix (K={k})",
        )
        confusion_paths.append(str(cm_path))

    summary = {
        "config": asdict(config),
        "splits_manifest_path": str(manifest_path),
        "eval_stage_name": eval_stage_name,
        "eval_dir": str(eval_dir),
        "ce_stage_dir": str(ce_stage_dir),
        "backbone_source": config.backbone_source,
        "distance_metric": config.distance_metric,
        "effective_distance_metric": effective_distance_metric,
        "embedding_space": config.embedding_space,
        "embedding_transform": config.embedding_transform,
        "active_embedding_transform": active_embedding_transform,
        "normalize_embeddings": config.normalize_embeddings,
        "effective_normalize_embeddings": effective_normalize_embeddings,
        "normalize_prior_mean_for_update": config.normalize_prior_mean_for_update,
        "effective_normalize_prior_mean_for_update": effective_normalize_prior_mean,
        "project_posterior_to_sphere": config.project_posterior_to_sphere,
        "effective_project_posterior_to_sphere": (
            effective_project_posterior_to_sphere
        ),
        "em_iterations": int(config.em_iterations),
        "em_temperature": float(config.em_temperature),
        "em_likelihood_variance": float(em_likelihood_variance),
        "em_min_soft_count": float(config.em_min_soft_count),
        "em_uniform_class_prior": bool(config.em_uniform_class_prior),
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

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from common import (
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
    extract_supcon_embeddings,
    indices_by_activity,
    load_ce_backbone,
    prepare_cfg,
    prototype_logits,
    reconcile_activity_config,
    resolve_distance_metric,
    resolve_output_root,
    set_seed,
    split_indices_for_fold,
)
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from whar_datasets import PreProcessingPipeline, WHARDatasetID


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

    fold_id: str | None = "loso_subject_2"
    fold_index: int = 0
    max_folds: int | None = None
    k: int = 16
    episode: int = 0
    num_episodes_per_subject: int = 10
    min_query_per_class: int = 1
    support_query_session_disjoint: bool = False
    min_activities: int = 2
    max_activities: int | None = None

    batch_size: int = 256
    num_workers: int = 0
    normalize_embeddings: bool | None = None
    backbone_source: str = "ce"
    distance_metric: str = "auto"  # "auto", "cosine", or "euclidean"
    embedding_transform: str = "none"  # "none" or "signed_power"
    power_transform_exponent: float = 0.5
    power_transform_backbone_sources: tuple[str, ...] = ("ce",)
    prior_variance_floor: float = 1e-4  # 1e-4
    normalize_prior_mean_for_update: bool | None = None
    project_posterior_to_sphere: bool | None = None
    em_iterations: int = 1  # 10
    em_temperature: float = 0.5  # 1.0
    em_likelihood_variance: float | None = 0.05  # None
    em_likelihood_variance_source: str = "fixed"  # "fixed" or "responsibility"
    em_responsibility_variance_source: str = "fixed"  # "fixed" or "support"
    em_support_variance_floor: float = 1e-4
    em_min_soft_count: float = 1e-6
    em_uniform_class_prior: bool = True
    center_train_support_query: bool = True

    tsne_perplexity: float | None = None
    tsne_learning_rate: str | float = "auto"
    tsne_init: str = "pca"
    tsne_random_state: int = 0
    support_alpha: float = 0.28
    support_size: float = 18.0
    support_color: str = "#ef4444"
    prototype_size: float = 150.0
    final_prototype_size: float = 120.0
    initial_prototype_size: float = 105.0
    trajectory_marker_size: float = 38.0
    trajectory_alpha: float = 1
    arrow_alpha: float = 1.0
    arrow_linewidth: float = 1.2
    seaborn_style: str = "whitegrid"
    frame_facecolor: str = "#ffffff"
    frame_edgecolor: str = "#cbd5e1"
    frame_linewidth: float = 1.2
    remove_tsne_frame: bool = False
    figure_width: float = 4.0
    figure_height: float = 3.5
    panel_width: float = 4
    dpi: int = 220
    show_support_true_colors: bool = False
    show_initial_prototypes_each_frame: bool = True
    show_kde_modes: bool = True
    kde_by_class: bool = False
    kde_levels: int = 22
    kde_thresh: float = 0.005
    kde_cut: float = 3.0
    kde_bw_adjust: float = 0.45
    kde_alpha: float = 0.55
    kde_cubehelix_start: float = 0.6
    kde_cubehelix_rot: float = -0.7

    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_root: str | None = None
    ce_stage_name: str = "01_tinierhar_ce_loso"
    stage_name: str = "10_map_em_tsne_steps_loso"
    separate_backbone_source_dir: bool = True
    force_rerun: bool = True


RUN_CONFIG = Config()


def _loader(dataset: WindowDataset, config: Config) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )


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
    dataloader: DataLoader,
    device: torch.device,
    config: Config,
    backbone_source: str,
    normalize_embeddings: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    embeddings, labels, subjects = extract_supcon_embeddings(
        backbone,
        None,
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
) -> tuple[list[int], np.ndarray]:
    support_indices: list[int] = []
    support_y: list[int] = []
    for activity_id in activity_ids:
        candidates = np.asarray(by_activity[int(activity_id)], dtype=np.int64)
        support = rng.permutation(candidates)[: int(k)]
        support_indices.extend(int(x) for x in support.tolist())
        support_y.extend([int(activity_id)] * len(support))
    support_perm = rng.permutation(len(support_indices))
    support_indices = [support_indices[int(i)] for i in support_perm.tolist()]
    support_y_np = np.asarray(
        [support_y[int(i)] for i in support_perm.tolist()], dtype=np.int64
    )
    return support_indices, support_y_np


def _sample_episode_session_disjoint(
    by_activity_session: dict[int, dict[Any, np.ndarray]],
    activity_ids: Sequence[int],
    k: int,
    min_query_per_class: int,
    rng: np.random.Generator,
) -> tuple[list[int], np.ndarray, dict[int, dict[str, Any]]]:
    support_indices: list[int] = []
    support_y: list[int] = []
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
                f"Activity {activity_id} has no session-disjoint split for k={k}."
            )
        query_session_id = candidate_query_sessions[
            int(rng.integers(0, len(candidate_query_sessions)))
        ]
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
        support_y.extend([int(activity_id)] * len(support))
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
    support_perm = rng.permutation(len(support_indices))
    support_indices = [support_indices[int(i)] for i in support_perm.tolist()]
    support_y_np = np.asarray(
        [support_y[int(i)] for i in support_perm.tolist()], dtype=np.int64
    )
    return support_indices, support_y_np, session_meta


def _class_diagonal_gaussian_stats(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    variance_floor: float,
    normalize_means: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    means: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    class_labels: list[int] = []
    missing_classes: list[int] = []
    feature_dim = int(embeddings.shape[1])
    for cls in range(int(num_classes)):
        mask = labels == int(cls)
        if not bool(mask.any()):
            missing_classes.append(int(cls))
            means.append(torch.zeros(feature_dim, dtype=embeddings.dtype))
            variances.append(
                torch.full(
                    (feature_dim,), float(variance_floor), dtype=embeddings.dtype
                )
            )
            continue
        cls_emb = embeddings[mask]
        class_labels.append(int(cls))
        means.append(cls_emb.mean(dim=0))
        if cls_emb.shape[0] > 1:
            var = cls_emb.var(dim=0, unbiased=True)
        else:
            var = torch.zeros(cls_emb.shape[1], dtype=cls_emb.dtype)
        variances.append(var.clamp_min(float(variance_floor)))
    mean_tensor = torch.stack(means, dim=0)
    if normalize_means:
        mean_tensor = F.normalize(mean_tensor, p=2, dim=1)
    return mean_tensor, torch.stack(variances, dim=0), class_labels, missing_classes


def _em_likelihood_variance(config: Config) -> float:
    if config.em_likelihood_variance is not None:
        variance = float(config.em_likelihood_variance)
    else:
        variance = float(config.em_temperature) / 2.0
    if variance <= 0:
        raise ValueError("EM likelihood variance must be > 0.")
    return variance


def _validate_em_likelihood_variance_source(value: str) -> str:
    if value not in {"fixed", "responsibility"}:
        raise ValueError(
            "em_likelihood_variance_source must be 'fixed' or 'responsibility'."
        )
    return value


def _validate_em_responsibility_variance_source(value: str) -> str:
    if value not in {"fixed", "support"}:
        raise ValueError(
            "em_responsibility_variance_source must be 'fixed' or 'support'."
        )
    return value


def _record_map_em_prototypes(
    prior_means: torch.Tensor,
    prior_variances: torch.Tensor,
    support_emb: torch.Tensor,
    activity_ids: Sequence[int],
    config: Config,
    em_likelihood_variance: float,
    em_likelihood_variance_source: str,
    em_responsibility_variance_source: str,
) -> tuple[list[torch.Tensor], list[dict[str, float]]]:
    if int(config.em_iterations) < 1:
        raise ValueError("em_iterations must be >= 1.")
    if float(config.em_temperature) <= 0:
        raise ValueError("em_temperature must be > 0.")
    if float(config.em_min_soft_count) <= 0:
        raise ValueError("em_min_soft_count must be > 0.")
    em_likelihood_variance_source = _validate_em_likelihood_variance_source(
        em_likelihood_variance_source
    )
    em_responsibility_variance_source = _validate_em_responsibility_variance_source(
        em_responsibility_variance_source
    )
    if float(config.em_support_variance_floor) <= 0:
        raise ValueError("em_support_variance_floor must be > 0.")

    activity_tensor = torch.tensor([int(x) for x in activity_ids], dtype=torch.long)
    prior_mu = prior_means[activity_tensor]
    prior_var = prior_variances[activity_tensor].clamp_min(
        float(config.prior_variance_floor)
    )
    prototypes = prior_mu.clone()
    states = [prototypes.clone()]
    diagnostics: list[dict[str, float]] = [
        {
            "iteration": 0.0,
            "prior_weight_mean": 1.0,
            "support_weight_mean": 0.0,
            "soft_count_min": 0.0,
            "soft_count_max": 0.0,
            "responsibility_confidence_mean": 0.0,
            "prototype_shift_mean": 0.0,
        }
    ]
    class_log_prior = None
    if not config.em_uniform_class_prior:
        class_log_prior = torch.zeros(len(activity_ids), dtype=support_emb.dtype)

    support_var = (
        prior_var.clamp_min(float(config.em_support_variance_floor))
        if em_responsibility_variance_source == "support"
        else torch.full_like(prior_var, float(em_likelihood_variance))
    )
    for iteration in range(1, int(config.em_iterations) + 1):
        if em_responsibility_variance_source == "support":
            responsibility_var = support_var.clamp_min(
                float(config.em_support_variance_floor)
            )
            centered_for_logits = support_emb.unsqueeze(1) - prototypes.unsqueeze(0)
            logits = -0.5 * (
                centered_for_logits.pow(2) / responsibility_var.unsqueeze(0)
                + responsibility_var.unsqueeze(0).log()
            ).sum(dim=2)
        else:
            logits = prototype_logits(
                support_emb,
                prototypes,
                float(config.em_temperature),
                "euclidean",
            )
        if class_log_prior is not None:
            logits = logits + class_log_prior.view(1, -1)
        responsibilities = torch.softmax(logits, dim=1)
        soft_counts = responsibilities.sum(dim=0)
        safe_counts = soft_counts.clamp_min(float(config.em_min_soft_count))
        soft_means = responsibilities.T @ support_emb
        soft_means = soft_means / safe_counts.view(-1, 1)
        if (
            em_likelihood_variance_source == "responsibility"
            or em_responsibility_variance_source == "support"
        ):
            centered = support_emb.unsqueeze(1) - soft_means.unsqueeze(0)
            support_var = (responsibilities.unsqueeze(-1) * centered.pow(2)).sum(dim=0)
            support_var = support_var / safe_counts.view(-1, 1)
            support_var = support_var.clamp_min(float(config.em_support_variance_floor))
        else:
            support_var = torch.full_like(prior_var, float(em_likelihood_variance))
        prior_precision = 1.0 / prior_var
        update_var = (
            support_var
            if em_likelihood_variance_source == "responsibility"
            else torch.full_like(prior_var, float(em_likelihood_variance))
        )
        support_precision = safe_counts.view(-1, 1) / update_var
        precision_sum = prior_precision + support_precision
        prototypes = (
            prior_precision * prior_mu + support_precision * soft_means
        ) / precision_sum
        prior_weight = prior_precision / precision_sum
        support_weight = support_precision / precision_sum
        prototype_shift = (prototypes - prior_mu).norm(p=2, dim=1)
        diagnostics.append(
            {
                "iteration": float(iteration),
                "prior_weight_mean": float(prior_weight.mean().item()),
                "support_weight_mean": float(support_weight.mean().item()),
                "soft_count_min": float(soft_counts.min().item()),
                "soft_count_max": float(soft_counts.max().item()),
                "responsibility_confidence_mean": float(
                    responsibilities.max(dim=1).values.mean().item()
                ),
                "prototype_shift_mean": float(prototype_shift.mean().item()),
                "em_likelihood_variance_source": em_likelihood_variance_source,
                "em_responsibility_variance_source": em_responsibility_variance_source,
                "em_support_variance_mean": float(support_var.mean().item()),
                "em_support_variance_min": float(support_var.min().item()),
                "em_support_variance_max": float(support_var.max().item()),
            }
        )
        states.append(prototypes.clone())
    return states, diagnostics


def _fit_shared_tsne(
    support_emb: torch.Tensor,
    prototype_states: list[torch.Tensor],
    config: Config,
) -> tuple[np.ndarray, list[np.ndarray]]:
    chunks = [support_emb]
    chunks.extend(prototype_states)
    combined = torch.cat(chunks, dim=0).numpy()
    n_samples = int(combined.shape[0])
    if n_samples < 4:
        raise ValueError("Need at least 4 points for t-SNE visualization.")
    if config.tsne_perplexity is None:
        perplexity = min(30.0, max(2.0, float((n_samples - 1) // 3)))
    else:
        perplexity = float(config.tsne_perplexity)
    perplexity = min(perplexity, float(n_samples - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=config.tsne_learning_rate,
        init=config.tsne_init,
        random_state=int(config.tsne_random_state),
    )
    coords = tsne.fit_transform(combined)
    support_end = int(support_emb.shape[0])
    support_xy = coords[:support_end]
    prototype_xy_states: list[np.ndarray] = []
    cursor = support_end
    for prototypes in prototype_states:
        next_cursor = cursor + int(prototypes.shape[0])
        prototype_xy_states.append(coords[cursor:next_cursor])
        cursor = next_cursor
    return support_xy, prototype_xy_states


def _plot_trajectory(
    support_emb: torch.Tensor,
    support_y: np.ndarray,
    prototype_states: list[torch.Tensor],
    activity_ids: Sequence[int],
    names: Sequence[str],
    config: Config,
    out_png: Path,
    out_pdf: Path,
) -> None:
    sns.set_theme(style=config.seaborn_style, context="talk")
    palette = sns.color_palette("tab20", n_colors=max(20, len(activity_ids)))
    proto_colors = {
        int(activity_id): palette[i % len(palette)]
        for i, activity_id in enumerate(activity_ids)
    }
    fig, ax = plt.subplots(
        1, 1, figsize=(float(config.figure_width), float(config.figure_height))
    )
    support_xy, prototype_xy_states = _fit_shared_tsne(
        support_emb,
        prototype_states,
        config,
    )
    all_xy = np.concatenate([support_xy, *prototype_xy_states], axis=0)
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    x_pad = max((float(x_max) - float(x_min)) * 0.15, 1e-6)
    y_pad = max((float(y_max) - float(y_min)) * 0.15, 1e-6)
    x_lim_min = float(x_min) - x_pad
    x_lim_max = float(x_max) + x_pad
    y_lim_min = float(y_min) - y_pad
    y_lim_max = float(y_max) + y_pad

    if bool(config.show_kde_modes) and support_xy.shape[0] >= 4:
        support_min_x = float(support_xy[:, 0].min())
        support_max_x = float(support_xy[:, 0].max())
        support_min_y = float(support_xy[:, 1].min())
        support_max_y = float(support_xy[:, 1].max())
        clearance = max(
            min(
                support_min_x - x_lim_min,
                x_lim_max - support_max_x,
                support_min_y - y_lim_min,
                y_lim_max - support_max_y,
            ),
            0.0,
        )
        bw = max(float(config.kde_bw_adjust), 1e-6)
        support_std = max(
            float(np.std(support_xy[:, 0])), float(np.std(support_xy[:, 1])), 1e-6
        )
        max_safe_cut = clearance / (bw * support_std)
        effective_kde_cut = config.kde_cut  # min(float(config.kde_cut), max_safe_cut)
        if bool(config.kde_by_class):
            unique_ids = [int(a) for a in activity_ids if np.any(support_y == int(a))]
            denom = max(1, len(unique_ids))
            for i, activity_id in enumerate(unique_ids):
                mask = support_y == int(activity_id)
                if int(mask.sum()) < 4:
                    continue
                start = float(config.kde_cubehelix_start) + (3.0 * i / denom)
                cmap = sns.cubehelix_palette(
                    start=start,
                    rot=float(config.kde_cubehelix_rot),
                    light=1.0,
                    as_cmap=True,
                )
                sns.kdeplot(
                    x=support_xy[mask, 0],
                    y=support_xy[mask, 1],
                    cmap=cmap,
                    fill=True,
                    levels=int(config.kde_levels),
                    thresh=float(config.kde_thresh),
                    cut=effective_kde_cut,
                    bw_adjust=float(config.kde_bw_adjust),
                    alpha=float(config.kde_alpha),
                    ax=ax,
                    zorder=0,
                )
        else:
            cmap = sns.cubehelix_palette(
                start=float(config.kde_cubehelix_start),
                rot=float(config.kde_cubehelix_rot),
                light=1.0,
                as_cmap=True,
            )
            sns.kdeplot(
                x=support_xy[:, 0],
                y=support_xy[:, 1],
                cmap=cmap,
                fill=True,
                levels=int(config.kde_levels),
                thresh=float(config.kde_thresh),
                cut=effective_kde_cut,
                bw_adjust=float(config.kde_bw_adjust),
                alpha=float(config.kde_alpha),
                ax=ax,
                zorder=0,
            )

    if config.show_support_true_colors:
        for activity_id in activity_ids:
            mask = support_y == int(activity_id)
            ax.scatter(
                support_xy[mask, 0],
                support_xy[mask, 1],
                s=float(config.support_size),
                color=config.support_color,
                alpha=float(config.support_alpha),
                linewidths=0,
            )
    else:
        ax.scatter(
            support_xy[:, 0],
            support_xy[:, 1],
            s=float(config.support_size),
            color=config.support_color,
            alpha=float(config.support_alpha),
            linewidths=0,
            label="Unlabeled Support",
            zorder=1,
        )

    for proto_idx, activity_id in enumerate(activity_ids):
        path = np.asarray([state[proto_idx] for state in prototype_xy_states])
        color = proto_colors[int(activity_id)]
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linewidth=float(config.arrow_linewidth),
            alpha=float(config.trajectory_alpha),
            zorder=3,
        )
        for step_idx in range(path.shape[0] - 1):
            start = path[step_idx]
            end = path[step_idx + 1]
            if np.allclose(start, end):
                continue
            arrow_ann = ax.annotate(
                "",
                xy=(end[0], end[1]),
                xytext=(start[0], start[1]),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#111827",
                    "alpha": float(config.arrow_alpha),
                    "lw": float(config.arrow_linewidth),
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "mutation_scale": 12,
                },
                zorder=10,
            )
            if arrow_ann.arrow_patch is not None:
                arrow_ann.arrow_patch.set_zorder(10)
        if path.shape[0] > 2:
            ax.scatter(
                path[1:-1, 0],
                path[1:-1, 1],
                s=float(config.trajectory_marker_size),
                marker="o",
                color=color,
                alpha=float(config.trajectory_alpha),
                edgecolor="white",
                linewidth=0.4,
                zorder=4,
            )
        ax.scatter(
            path[0, 0],
            path[0, 1],
            s=float(config.initial_prototype_size),
            marker="o",
            facecolors="none",
            edgecolors=[color],
            linewidth=2.2,
            alpha=0.98,
            zorder=5,
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            s=float(config.final_prototype_size),
            marker="o",
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=6,
        )

    fig.suptitle(
        f"MAP-EM Prototype Trajectories\n(HARTH, {int(config.k)}-Shot, {int(config.em_iterations)} Iter)",
        fontsize=11,
        fontweight="bold",
        y=0.975,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x_lim_min, x_lim_max)
    ax.set_ylim(y_lim_min, y_lim_max)
    ax.set_facecolor(config.frame_facecolor)
    ax.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.7)
    for spine in ax.spines.values():
        if bool(config.remove_tsne_frame):
            spine.set_visible(False)
        else:
            spine.set_visible(True)
            spine.set_color(config.frame_edgecolor)
            spine.set_linewidth(float(config.frame_linewidth))

    class_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=proto_colors[int(activity_id)],
            markerfacecolor=proto_colors[int(activity_id)],
            markeredgecolor="white",
            linewidth=2.0,
            markersize=7,
            label=names[int(activity_id)],
        )
        for activity_id in activity_ids
    ]
    semantic_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=config.support_color,
            markeredgecolor="none",
            alpha=float(config.support_alpha),
            markersize=7,
            label="Unlabeled Support",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#111827",
            markersize=9,
            label="Prior Proto",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#111827",
            markeredgecolor="black",
            markeredgewidth=1.3,
            markersize=8,
            label="Updated Proto",
        ),
    ]
    content_left = 0.03
    content_right = 0.97
    content_width = content_right - content_left

    upper_legend = fig.legend(
        handles=semantic_handles,
        loc="upper left",
        bbox_to_anchor=(content_left, 0.875, content_width, 0.0),
        mode="expand",
        ncol=min(len(semantic_handles), 5),
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        fontsize=8.0,
        borderpad=0.60,
        labelspacing=0.35,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    upper_legend.get_frame().set_edgecolor(config.frame_edgecolor)
    upper_legend.get_frame().set_linewidth(float(config.frame_linewidth))
    for text in upper_legend.get_texts():
        text.set_fontweight("bold")
    top_row = class_handles[:3]
    bottom_row = class_handles[3:]
    if top_row:
        top_class_legend = fig.legend(
            handles=top_row,
            loc="lower center",
            ncol=len(top_row),
            frameon=False,
            fontsize=8.5,
            bbox_to_anchor=(0.5, 0.055),
            columnspacing=1.0,
            handletextpad=0.5,
            handlelength=1.8,
            labelspacing=0.25,
            borderaxespad=0.0,
        )
        for text in top_class_legend.get_texts():
            text.set_fontweight("bold")
    if bottom_row:
        bottom_class_legend = fig.legend(
            handles=bottom_row,
            loc="lower center",
            ncol=len(bottom_row),
            frameon=False,
            fontsize=8.5,
            bbox_to_anchor=(0.5, 0.02),
            columnspacing=1.0,
            handletextpad=0.5,
            handlelength=1.8,
            labelspacing=0.25,
            borderaxespad=0.0,
        )
        for text in bottom_class_legend.get_texts():
            text.set_fontweight("bold")
    fig.subplots_adjust(left=content_left, right=content_right, top=0.775, bottom=0.14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(config.dpi))
    fig.savefig(out_pdf)
    plt.close(fig)


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    if int(config.k) <= 0:
        raise ValueError("k must be > 0 for an unlabeled support visualization.")
    if config.embedding_transform not in {"none", "signed_power"}:
        raise ValueError("embedding_transform must be 'none' or 'signed_power'.")

    device = torch.device(config.device)
    output_root = resolve_output_root(config.output_root, config.dataset_id)
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
    em_likelihood_variance_source = _validate_em_likelihood_variance_source(
        config.em_likelihood_variance_source
    )
    em_responsibility_variance_source = _validate_em_responsibility_variance_source(
        config.em_responsibility_variance_source
    )
    if float(config.em_support_variance_floor) <= 0:
        raise ValueError("em_support_variance_floor must be > 0.")
    stage_parts = [config.stage_name, f"k{int(config.k)}"]
    if config.separate_backbone_source_dir:
        stage_parts.append(f"{config.backbone_source}_backbone")
    stage_parts.append(effective_distance_metric)
    if active_embedding_transform != "none":
        stage_parts.append(_artifact_safe_name(active_embedding_transform))
        stage_parts.append("l2norm" if effective_normalize_embeddings else "rawstats")
    if config.center_train_support_query:
        stage_parts.append("centered")
    if em_likelihood_variance_source == "responsibility":
        stage_parts.append("respvar")
        stage_parts.append(
            f"varfloor_{_artifact_safe_name(f'{float(config.em_support_variance_floor):g}')}"
        )
    if em_responsibility_variance_source == "support":
        stage_parts.append("diaglike")
    stage_dir = output_root / "_".join(stage_parts)
    stage_dir.mkdir(parents=True, exist_ok=True)

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
    if config.fold_id is not None:
        selected_folds = [next(f for f in folds if f.fold_id == config.fold_id)]
    else:
        start = int(config.fold_index)
        stop = None if config.max_folds is None else start + int(config.max_folds)
        selected_folds = folds[start:stop]

    names = class_names(cfg)
    all_metadata: list[dict[str, Any]] = []
    for fold in selected_folds:
        ckpt_path = ce_stage_dir / fold.fold_id / "best_model_with_meta.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing CE checkpoint: {ckpt_path}")

        split = split_indices_for_fold(session_df, window_df, fold)
        loader = build_loader(cfg, session_df, pre, window_df, split.train_indices)
        train_ds = WindowDataset(loader, split.train_indices)
        test_ds = WindowDataset(loader, split.test_indices)
        backbone, _checkpoint = load_ce_backbone(ckpt_path, device)
        train_emb, train_y, _train_subjects, train_transform = (
            _extract_preprocessed_embeddings(
                backbone,
                _loader(train_ds, config),
                device,
                config,
                config.backbone_source,
                effective_normalize_embeddings,
            )
        )
        if train_transform != active_embedding_transform:
            raise RuntimeError(
                "Active embedding transform changed unexpectedly: "
                f"{active_embedding_transform} -> {train_transform}."
            )
        train_center = train_emb.mean(dim=0)
        train_emb_for_prior = (
            train_emb - train_center if config.center_train_support_query else train_emb
        )
        prior_means, prior_variances, prior_class_labels, missing_train_classes = (
            _class_diagonal_gaussian_stats(
                train_emb_for_prior,
                train_y,
                num_classes=int(cfg.num_of_activities),
                variance_floor=config.prior_variance_floor,
                normalize_means=effective_normalize_prior_mean,
            )
        )
        available_prior_class_ids = {int(x) for x in prior_class_labels}
        test_emb, _test_y, _test_subjects, test_transform = (
            _extract_preprocessed_embeddings(
                backbone,
                _loader(test_ds, config),
                device,
                config,
                config.backbone_source,
                effective_normalize_embeddings,
            )
        )
        if test_transform != active_embedding_transform:
            raise RuntimeError(
                "Active embedding transform changed unexpectedly: "
                f"{active_embedding_transform} -> {test_transform}."
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
        activity_ids = (
            _eligible_activities_session_disjoint(
                by_activity_session or {},
                int(config.k),
                config.min_query_per_class,
            )
            if config.support_query_session_disjoint
            else _eligible_activities(
                by_activity, int(config.k), config.min_query_per_class
            )
        )
        activity_ids = [
            int(x) for x in activity_ids if int(x) in available_prior_class_ids
        ]
        if config.max_activities is not None:
            activity_ids = activity_ids[: int(config.max_activities)]
        if len(activity_ids) < int(config.min_activities):
            raise ValueError(
                f"{fold.fold_id} has only {len(activity_ids)} eligible activities for "
                f"k={config.k}: {activity_ids}."
            )

        fold_dir = stage_dir / fold.fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)
        for episode in range(
            int(config.episode),
            int(config.episode) + int(config.num_episodes_per_subject),
        ):
            out_png = (
                fold_dir
                / f"map_em_tsne_trajectory_k{int(config.k)}_episode{episode}.png"
            )
            metadata_path = out_png.with_suffix(".json")
            if not config.force_rerun and out_png.exists() and metadata_path.exists():
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
                all_metadata.append(existing)
                print(f"[{fold.fold_id}] skipping existing episode {episode}")
                continue

            rng = np.random.default_rng(
                int(config.seed)
                + 1_000_000 * int(fold.test_subject_ids[0])
                + 10_000 * int(config.k)
                + int(episode)
            )
            episode_session_meta: dict[int, dict[str, Any]] = {}
            if config.support_query_session_disjoint:
                support_indices, support_y, episode_session_meta = (
                    _sample_episode_session_disjoint(
                        by_activity_session or {},
                        activity_ids,
                        int(config.k),
                        config.min_query_per_class,
                        rng,
                    )
                )
            else:
                support_indices, support_y = _sample_episode(
                    by_activity,
                    activity_ids,
                    int(config.k),
                    rng,
                )
            support_emb = torch.stack(
                [embeddings_by_window[int(idx)] for idx in support_indices], dim=0
            )
            center_shift_norm = 0.0
            if config.center_train_support_query:
                support_center = support_emb.mean(dim=0)
                support_emb = support_emb - support_center
                center_shift_norm = float(
                    (support_center - train_center).norm(p=2).item()
                )

            prototype_states, diagnostics = _record_map_em_prototypes(
                prior_means=prior_means,
                prior_variances=prior_variances,
                support_emb=support_emb,
                activity_ids=activity_ids,
                config=config,
                em_likelihood_variance=em_likelihood_variance,
                em_likelihood_variance_source=em_likelihood_variance_source,
                em_responsibility_variance_source=em_responsibility_variance_source,
            )
            out_pdf = out_png.with_suffix(".pdf")
            _plot_trajectory(
                support_emb=support_emb,
                support_y=support_y,
                prototype_states=prototype_states,
                activity_ids=activity_ids,
                names=names,
                config=config,
                out_png=out_png,
                out_pdf=out_pdf,
            )
            metadata = {
                "config": asdict(config),
                "fold_id": fold.fold_id,
                "episode": int(episode),
                "train_subject_ids": fold.train_subject_ids,
                "test_subject_ids": fold.test_subject_ids,
                "ce_checkpoint": str(ckpt_path),
                "activity_ids": [int(x) for x in activity_ids],
                "activity_names": [names[int(x)] for x in activity_ids],
                "support_indices": [int(x) for x in support_indices],
                "support_labels_for_debug_only": [int(x) for x in support_y.tolist()],
                "missing_train_classes": [int(x) for x in missing_train_classes],
                "prototype_step_indices": list(range(len(prototype_states))),
                "tsne_mode": "single_shared_support_and_all_prototype_states",
                "em_diagnostics": diagnostics,
                "em_likelihood_variance_source": em_likelihood_variance_source,
                "em_responsibility_variance_source": em_responsibility_variance_source,
                "center_train_support_query": bool(config.center_train_support_query),
                "center_shift_norm": center_shift_norm,
                "episode_session_meta": episode_session_meta,
                "png_path": str(out_png),
                "pdf_path": str(out_pdf),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            all_metadata.append(metadata)
            print(f"[{fold.fold_id}] saved MAP-EM t-SNE episode {episode}: {out_png}")

    summary = {
        "config": asdict(config),
        "stage_dir": str(stage_dir),
        "num_subjects": len(selected_folds),
        "num_plots": len(all_metadata),
        "plots": all_metadata,
    }
    (stage_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import (
    DEFAULT_TRAIN_MAX_K_PER_CLASS,
    DEFAULT_TRAIN_MIN_K_PER_CLASS,
    ROOT,
    SharedConfig,
    build_or_load_loso_folds,
    config_fingerprint,
    k_choices_from_range,
    prepare_cfg,
    set_seed,
    split_indices_for_fold,
)
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from whar_datasets import (
    Loader,
    PostProcessingPipeline,
    PreProcessingPipeline,
    WHARDatasetID,
)

from hyper_har.backbone.tinierhar import TinierHAR
from hyper_har.config import DEFAULT_CONFIG

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SET_ENCODER_ATTENTION_MODULE = _load_module_from_path(
    "final_phase2_attention_set_encoder",
    SRC / "hyper_har" / "set-encoder" / "attention.py",
)
SET_ENCODER_SIMPLE_MODULE = _load_module_from_path(
    "final_phase2_simple_set_encoder",
    SRC / "hyper_har" / "set-encoder" / "simple.py",
)
AttentionSetEncoder = SET_ENCODER_ATTENTION_MODULE.AttentionSetEncoder
PrototypicalSetEncoder = SET_ENCODER_SIMPLE_MODULE.PrototypicalSetEncoder


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

    encoder: str = "attention"
    projection_hidden_dim: int = 128
    projection_dim: int = 64
    train_min_k_per_class: int = DEFAULT_TRAIN_MIN_K_PER_CLASS
    train_max_k_per_class: int = DEFAULT_TRAIN_MAX_K_PER_CLASS
    n_subjects_per_batch: int = 10
    m_sets_per_subject: int = 2
    val_m_sets_per_subject: int = 1
    train_batches_per_epoch: int = 128
    val_batches_per_epoch: int = 32
    eval_sets_per_subject: int = 24
    tsne_k_values: tuple[int, ...] = (1, 4, 8, 16, 32)
    val_support_sets_per_subject: int = 5
    val_max_query_sets_per_subject: int | None = 64
    knn_neighbors: int = 1
    supcon_temperature: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 5  # 10
    patience: int = 10
    tsne_every_n_epochs: int = 5
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    output_root: str = str(ROOT / "artifacts" / "final_pipeline")
    max_folds: int | None = None
    force_rerun: bool = False


RUN_CONFIG = Config()


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.shape[0], device=features.device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - sim_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        return (-mean_log_prob_pos).mean()


class ContrastiveSetEncoder(nn.Module):
    def __init__(
        self, set_encoder: nn.Module, projection_hidden_dim: int, projection_dim: int
    ):
        super().__init__()
        self.set_encoder = set_encoder
        c_subject_dim = int(getattr(set_encoder, "output_dim"))
        self.projection_head = nn.Sequential(
            nn.Linear(c_subject_dim, projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, projection_dim),
        )

    def encode_subject(
        self, x_support: torch.Tensor, y_support: torch.Tensor
    ) -> torch.Tensor:
        return self.set_encoder(x_support, y_support)

    def forward(self, x_support: torch.Tensor, y_support: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self.encode_subject(x_support, y_support))


@dataclass(frozen=True)
class SubjectSetIndices:
    subject_id: int
    set_indices: np.ndarray
    set_labels: np.ndarray


class NMSetsBatchSampler(Sampler[list[SubjectSetIndices]]):
    def __init__(
        self,
        indices_by_subject_activity: Mapping[int, Mapping[int, np.ndarray]],
        activity_ids: Sequence[int],
        min_k_per_class: int,
        max_k_per_class: int,
        n_subjects: int,
        m_sets_per_subject: int,
        batches_per_epoch: int,
        seed: int,
    ) -> None:
        self.indices_by_subject_activity = {
            int(sid): {
                int(aid): np.asarray(v, dtype=np.int64)
                for aid, v in per_activity.items()
            }
            for sid, per_activity in indices_by_subject_activity.items()
        }
        self.activity_ids = [int(a) for a in activity_ids]
        self.min_k_per_class = int(min_k_per_class)
        self.max_k_per_class = int(max_k_per_class)
        self.n_subjects = int(n_subjects)
        self.m_sets_per_subject = int(m_sets_per_subject)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

        self.eligible_subject_ids_by_k: dict[int, list[int]] = {}
        for k in range(self.min_k_per_class, self.max_k_per_class + 1):
            needed = self.m_sets_per_subject * k
            eligible: list[int] = []
            for sid, per_activity in self.indices_by_subject_activity.items():
                if all(
                    len(per_activity.get(aid, np.empty(0, dtype=np.int64))) >= needed
                    for aid in self.activity_ids
                ):
                    eligible.append(int(sid))
            if len(eligible) >= self.n_subjects:
                self.eligible_subject_ids_by_k[k] = eligible
        self.valid_set_sizes = tuple(sorted(self.eligible_subject_ids_by_k.keys()))
        if not self.valid_set_sizes:
            raise ValueError("No feasible K values for N x M SupCon sampling.")

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[SubjectSetIndices]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        k_values = np.asarray(self.valid_set_sizes, dtype=np.int64)
        for _ in range(self.batches_per_epoch):
            k = int(rng.choice(k_values))
            eligible = np.asarray(self.eligible_subject_ids_by_k[k], dtype=np.int64)
            sampled_subjects = rng.choice(eligible, size=self.n_subjects, replace=False)
            batch: list[SubjectSetIndices] = []
            for subject_id in sampled_subjects.tolist():
                per_activity = self.indices_by_subject_activity[int(subject_id)]
                activity_chunks: dict[int, np.ndarray] = {}
                for aid in self.activity_ids:
                    activity_chunks[int(aid)] = rng.choice(
                        per_activity[int(aid)],
                        size=self.m_sets_per_subject * k,
                        replace=False,
                    )
                for m_idx in range(self.m_sets_per_subject):
                    indices_parts: list[np.ndarray] = []
                    labels_parts: list[np.ndarray] = []
                    start = m_idx * k
                    end = (m_idx + 1) * k
                    for aid in self.activity_ids:
                        picked = activity_chunks[int(aid)][start:end]
                        indices_parts.append(picked)
                        labels_parts.append(np.full(k, int(aid), dtype=np.int64))
                    set_indices = np.concatenate(indices_parts, axis=0)
                    set_labels = np.concatenate(labels_parts, axis=0)
                    perm = rng.permutation(set_indices.shape[0])
                    batch.append(
                        SubjectSetIndices(
                            int(subject_id),
                            set_indices[perm],
                            set_labels[perm],
                        )
                    )
            rng.shuffle(batch)
            yield batch


def _build_feasible_train_sampler(
    subject_train: Mapping[int, Mapping[int, np.ndarray]],
    activity_ids: Sequence[int],
    config: Config,
) -> tuple[NMSetsBatchSampler, int, int]:
    max_n = int(config.n_subjects_per_batch)
    min_k = int(config.train_min_k_per_class)
    max_k = int(config.train_max_k_per_class)
    for n_subj in range(max_n, 1, -1):
        for k_hi in range(max_k, min_k - 1, -1):
            try:
                sampler = NMSetsBatchSampler(
                    indices_by_subject_activity=subject_train,
                    activity_ids=activity_ids,
                    min_k_per_class=min_k,
                    max_k_per_class=k_hi,
                    n_subjects=n_subj,
                    m_sets_per_subject=config.m_sets_per_subject,
                    batches_per_epoch=config.train_batches_per_epoch,
                    seed=config.seed,
                )
                return sampler, n_subj, k_hi
            except ValueError:
                continue
    raise ValueError(
        "Could not construct a feasible train sampler. "
        "Try lowering m_sets_per_subject, train_min_k_per_class, or selected activities."
    )


class NMSetsDataset(Dataset[dict[str, torch.Tensor | int]]):
    def __init__(self, loader: Loader):
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader.window_df)

    def __getitem__(self, item: SubjectSetIndices) -> dict[str, torch.Tensor | int]:  # type: ignore
        windows = []
        labels = []
        for idx in item.set_indices.tolist():
            x_np = np.asarray(self.loader.get_sample(int(idx))[0])
            if x_np.ndim == 3 and x_np.shape[0] == 1:
                x_np = x_np[0]
            windows.append(torch.from_numpy(x_np.copy()).float().unsqueeze(0))
        labels.extend(int(x) for x in item.set_labels.tolist())
        return {
            "x_set": torch.stack(windows, dim=0),
            "y_set": torch.tensor(labels, dtype=torch.long),
            "subject_id": int(item.subject_id),
        }


def nm_collate(
    samples: Sequence[dict[str, torch.Tensor | int]],
) -> dict[str, torch.Tensor]:
    x = torch.stack(
        [
            sample["x_set"]
            for sample in samples
            if isinstance(sample["x_set"], torch.Tensor)
        ],
        dim=0,
    )
    y_support = torch.stack(
        [
            sample["y_set"]
            for sample in samples
            if isinstance(sample["y_set"], torch.Tensor)
        ],
        dim=0,
    )
    y = torch.tensor(
        [int(sample["subject_id"]) for sample in samples], dtype=torch.long
    )
    return {"x": x, "y_support": y_support, "subject_id": y}


def _build_subject_activity_index(
    loader: Loader,
    indices: Sequence[int],
) -> tuple[dict[int, dict[int, np.ndarray]], list[int]]:
    meta = loader.window_df.loc[list(indices), ["session_id"]].copy()
    meta["window_index"] = meta.index.astype(int)
    session_meta = loader.session_df[
        ["session_id", "subject_id", "activity_id"]
    ].drop_duplicates("session_id")
    merged = meta.merge(session_meta, on="session_id", how="left")
    activity_ids = sorted(
        int(x) for x in merged["activity_id"].dropna().unique().tolist()
    )
    grouped = merged.groupby(["subject_id", "activity_id"])["window_index"]
    nested: dict[int, dict[int, np.ndarray]] = {}
    for (sid, aid), vals in grouped:
        nested.setdefault(int(sid), {})[int(aid)] = vals.to_numpy(dtype=np.int64)
    return nested, activity_ids


def _build_model(
    cfg: Config, num_channels: int, num_classes: int, window_size: int
) -> Any:
    backbone = TinierHAR(
        num_channels=num_channels,
        num_classes=num_classes,
        window_size=window_size,
        backbone_config=DEFAULT_CONFIG.backbone,
    )
    se_cfg = replace(DEFAULT_CONFIG.set_encoder, include_global_context=False)
    if cfg.encoder == "attention":
        set_encoder = AttentionSetEncoder(
            backbone=backbone,
            num_classes=num_classes,
            backbone_train_mode="unfreeze_all",
            force_conv_bn_eval=False,
            set_encoder_config=se_cfg,
        )
    else:
        set_encoder = PrototypicalSetEncoder(
            backbone=backbone,
            num_classes=num_classes,
            backbone_train_mode="unfreeze_all",
            force_conv_bn_eval=False,
            set_encoder_config=se_cfg,
        )
    model = ContrastiveSetEncoder(
        set_encoder=set_encoder,
        projection_hidden_dim=cfg.projection_hidden_dim,
        projection_dim=cfg.projection_dim,
    )
    return model


@torch.no_grad()
def _extract_subject_embeddings_stratified(
    model: ContrastiveSetEncoder,
    loader: Loader,
    subject_activity_index: Mapping[int, Mapping[int, np.ndarray]],
    activity_ids: Sequence[int],
    k_per_class: int,
    sets_per_subject: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    model.eval()
    all_sets_x: list[np.ndarray] = []
    all_sets_y: list[np.ndarray] = []
    all_subjects: list[int] = []

    k = int(k_per_class)
    for subject_id, per_activity in sorted(subject_activity_index.items()):
        if not all(
            len(per_activity.get(int(aid), np.empty(0, dtype=np.int64))) >= k
            for aid in activity_ids
        ):
            continue
        for _ in range(int(sets_per_subject)):
            set_indices_parts: list[np.ndarray] = []
            set_labels_parts: list[np.ndarray] = []
            for aid in activity_ids:
                picked = rng.choice(per_activity[int(aid)], size=k, replace=False)
                set_indices_parts.append(picked)
                set_labels_parts.append(np.full(k, int(aid), dtype=np.int64))
            set_indices = np.concatenate(set_indices_parts, axis=0)
            set_labels = np.concatenate(set_labels_parts, axis=0)
            perm = rng.permutation(set_indices.shape[0])
            set_indices = set_indices[perm]
            set_labels = set_labels[perm]
            x_windows = []
            for idx in set_indices.tolist():
                x_np = np.asarray(loader.get_sample(int(idx))[0])
                if x_np.ndim == 3 and x_np.shape[0] == 1:
                    x_np = x_np[0]
                x_windows.append(x_np)
            all_sets_x.append(np.stack(x_windows, axis=0))
            all_sets_y.append(set_labels)
            all_subjects.append(int(subject_id))

    if not all_sets_x:
        raise ValueError("No stratified evaluation sets could be sampled for t-SNE.")

    embeddings: list[np.ndarray] = []
    for start in range(0, len(all_sets_x), int(batch_size)):
        batch_x = np.stack(all_sets_x[start : start + int(batch_size)], axis=0)
        batch_y = np.stack(all_sets_y[start : start + int(batch_size)], axis=0)
        x_support = torch.from_numpy(batch_x).float().unsqueeze(2).to(device)
        y_support = torch.from_numpy(batch_y).long().to(device)
        raw = model.encode_subject(x_support, y_support)
        embeddings.append(raw.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0), np.asarray(all_subjects, dtype=np.int64)


def _save_tsne_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    seed: int,
) -> str | None:
    if embeddings.shape[0] < 3:
        return None
    perplexity = min(30.0, max(2.0, float(embeddings.shape[0] - 1) / 3.0))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        metric="cosine",
        random_state=seed,
    )
    coords = reducer.fit_transform(embeddings)
    unique_labels = sorted(int(x) for x in np.unique(labels).tolist())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = np.asarray([label_to_idx[int(x)] for x in labels], dtype=np.int64)
    n_classes = max(1, len(unique_labels))
    cmap = plt.get_cmap("tab20", n_classes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=encoded,
        cmap=cmap,
        vmin=0,
        vmax=max(0, n_classes - 1),
        s=34,
        alpha=0.85,
        linewidths=0.0,
    )
    ax.set_title(title)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=cmap(idx),
            markeredgecolor="none",
            markersize=6,
            label=str(label),
        )
        for idx, label in enumerate(unique_labels)
    ]
    ax.legend(handles=handles, title="Subject", loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


def _save_tsne_split_plot(
    embeddings: np.ndarray,
    subject_labels: np.ndarray,
    split_labels: np.ndarray,
    output_path: Path,
    title: str,
    seed: int,
) -> str | None:
    if embeddings.shape[0] < 3:
        return None
    perplexity = min(30.0, max(2.0, float(embeddings.shape[0] - 1) / 3.0))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        metric="cosine",
        random_state=seed,
    )
    coords = reducer.fit_transform(embeddings)
    unique_subjects = sorted(int(x) for x in np.unique(subject_labels).tolist())
    subject_to_idx = {label: idx for idx, label in enumerate(unique_subjects)}
    subject_encoded = np.asarray(
        [subject_to_idx[int(x)] for x in subject_labels], dtype=np.int64
    )
    n_subjects = max(1, len(unique_subjects))
    cmap = plt.get_cmap("tab20", n_subjects)
    split_markers = {
        "train": "o",
        "val": "^",
        "test": "X",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=160)
    for split_name in sorted(set(str(x) for x in split_labels.tolist())):
        mask = split_labels == split_name
        marker = split_markers.get(split_name, "s")
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=subject_encoded[mask],
            cmap=cmap,
            vmin=0,
            vmax=max(0, n_subjects - 1),
            s=42 if split_name != "train" else 34,
            alpha=0.86,
            marker=marker,
            linewidths=0.45 if split_name != "train" else 0.0,
            edgecolors="black" if split_name != "train" else "none",
            label=split_name,
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")

    subject_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=cmap(idx),
            markeredgecolor="none",
            markersize=6,
            label=str(label),
        )
        for idx, label in enumerate(unique_subjects)
    ]
    split_handles = [
        Line2D(
            [0],
            [0],
            marker=split_markers.get(split_name, "s"),
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label=split_name,
        )
        for split_name in sorted(set(str(x) for x in split_labels.tolist()))
    ]
    subject_legend = ax.legend(
        handles=subject_handles,
        title="Subject",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
    )
    ax.add_artist(subject_legend)
    ax.legend(
        handles=split_handles,
        title="Split",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_split_tsne_stratified(
    model: ContrastiveSetEncoder,
    loader: Loader,
    subject_activity_index: Mapping[int, Mapping[int, np.ndarray]],
    activity_ids: Sequence[int],
    split_name: str,
    output_dir: Path,
    k_per_class: int,
    sets_per_subject: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> str | None:
    try:
        embeddings, labels = _extract_subject_embeddings_stratified(
            model=model,
            loader=loader,
            subject_activity_index=subject_activity_index,
            activity_ids=activity_ids,
            k_per_class=k_per_class,
            sets_per_subject=sets_per_subject,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
    except ValueError:
        return None
    return _save_tsne_plot(
        embeddings=embeddings,
        labels=labels,
        output_path=output_dir / f"tsne_{split_name}_k{k_per_class}.png",
        title=f"Raw c_subject t-SNE ({split_name}, K/class={k_per_class})",
        seed=seed,
    )


def _plot_split_tsne_stratified_by_k(
    model: ContrastiveSetEncoder,
    loader: Loader,
    subject_activity_index: Mapping[int, Mapping[int, np.ndarray]],
    activity_ids: Sequence[int],
    split_name: str,
    output_dir: Path,
    k_values: Sequence[int],
    sets_per_subject: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, str | None]:
    paths: dict[str, str | None] = {}
    for k in k_values:
        paths[str(int(k))] = _plot_split_tsne_stratified(
            model=model,
            loader=loader,
            subject_activity_index=subject_activity_index,
            activity_ids=activity_ids,
            split_name=split_name,
            output_dir=output_dir,
            k_per_class=int(k),
            sets_per_subject=sets_per_subject,
            batch_size=batch_size,
            device=device,
            seed=seed + int(k),
        )
    return paths


def _plot_combined_tsne_stratified(
    model: ContrastiveSetEncoder,
    loader: Loader,
    split_subject_activity_indices: Mapping[
        str, Mapping[int, Mapping[int, np.ndarray]]
    ],
    activity_ids: Sequence[int],
    output_dir: Path,
    output_name: str,
    title_name: str,
    k_per_class: int,
    sets_per_subject: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> str | None:
    embeddings_parts: list[np.ndarray] = []
    subject_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    for split_offset, (split_name, subject_activity_index) in enumerate(
        split_subject_activity_indices.items()
    ):
        try:
            embeddings, subjects = _extract_subject_embeddings_stratified(
                model=model,
                loader=loader,
                subject_activity_index=subject_activity_index,
                activity_ids=activity_ids,
                k_per_class=k_per_class,
                sets_per_subject=sets_per_subject,
                batch_size=batch_size,
                seed=seed + split_offset * 10_000,
                device=device,
            )
        except ValueError:
            continue
        embeddings_parts.append(embeddings)
        subject_parts.append(subjects)
        split_parts.append(np.full(subjects.shape[0], str(split_name), dtype=object))
    if len(embeddings_parts) < 2:
        return None
    return _save_tsne_split_plot(
        embeddings=np.concatenate(embeddings_parts, axis=0),
        subject_labels=np.concatenate(subject_parts, axis=0),
        split_labels=np.concatenate(split_parts, axis=0),
        output_path=output_dir / f"tsne_{output_name}_k{k_per_class}.png",
        title=f"Raw c_subject t-SNE ({title_name}, K/class={k_per_class})",
        seed=seed,
    )


def _plot_combined_tsne_stratified_by_k(
    model: ContrastiveSetEncoder,
    loader: Loader,
    split_subject_activity_indices: Mapping[
        str, Mapping[int, Mapping[int, np.ndarray]]
    ],
    activity_ids: Sequence[int],
    output_dir: Path,
    output_name: str,
    title_name: str,
    k_values: Sequence[int],
    sets_per_subject: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, str | None]:
    paths: dict[str, str | None] = {}
    for k in k_values:
        paths[str(int(k))] = _plot_combined_tsne_stratified(
            model=model,
            loader=loader,
            split_subject_activity_indices=split_subject_activity_indices,
            activity_ids=activity_ids,
            output_dir=output_dir,
            output_name=output_name,
            title_name=title_name,
            k_per_class=int(k),
            sets_per_subject=sets_per_subject,
            batch_size=batch_size,
            device=device,
            seed=seed + int(k),
        )
    return paths


@torch.no_grad()
def _evaluate_val_knn_macro_f1(
    model: ContrastiveSetEncoder,
    loader: Loader,
    subject_activity_index: Mapping[int, Mapping[int, np.ndarray]],
    activity_ids: Sequence[int],
    k_per_class: int,
    support_sets_per_subject: int,
    max_query_sets_per_subject: int | None,
    knn_neighbors: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> float:
    rng = np.random.default_rng(seed)
    k = int(k_per_class)
    c = len(activity_ids)
    support_x: list[np.ndarray] = []
    support_set_labels: list[np.ndarray] = []
    support_y: list[int] = []
    query_x: list[np.ndarray] = []
    query_set_labels: list[np.ndarray] = []
    query_y: list[int] = []

    for subject_id, per_activity in sorted(subject_activity_index.items()):
        if not all(
            len(per_activity.get(int(aid), np.empty(0, dtype=np.int64))) >= k
            for aid in activity_ids
        ):
            continue

        max_sets = min(len(per_activity[int(aid)]) // k for aid in activity_ids)
        if max_sets <= int(support_sets_per_subject):
            continue

        support_n = max(1, min(int(support_sets_per_subject), max_sets - 1))
        query_n = max_sets - support_n
        if max_query_sets_per_subject is not None:
            query_n = min(query_n, int(max_query_sets_per_subject))
        if query_n <= 0:
            continue

        activity_chunks: dict[int, np.ndarray] = {}
        for aid in activity_ids:
            needed = (support_n + query_n) * k
            activity_chunks[int(aid)] = rng.choice(
                per_activity[int(aid)], size=needed, replace=False
            )

        def build_set(set_idx: int) -> tuple[np.ndarray, np.ndarray]:
            idx_parts: list[np.ndarray] = []
            label_parts: list[np.ndarray] = []
            start = set_idx * k
            end = (set_idx + 1) * k
            for aid in activity_ids:
                picked = activity_chunks[int(aid)][start:end]
                idx_parts.append(picked)
                label_parts.append(np.full(k, int(aid), dtype=np.int64))
            set_indices = np.concatenate(idx_parts, axis=0)
            set_labels = np.concatenate(label_parts, axis=0)
            perm = rng.permutation(k * c)
            return set_indices[perm], set_labels[perm]

        for s_idx in range(support_n):
            set_indices, set_labels = build_set(s_idx)
            x_windows = []
            for idx in set_indices.tolist():
                x_np = np.asarray(loader.get_sample(int(idx))[0])
                if x_np.ndim == 3 and x_np.shape[0] == 1:
                    x_np = x_np[0]
                x_windows.append(x_np)
            support_x.append(np.stack(x_windows, axis=0))
            support_set_labels.append(set_labels)
            support_y.append(int(subject_id))

        for q_idx in range(support_n, support_n + query_n):
            set_indices, set_labels = build_set(q_idx)
            x_windows = []
            for idx in set_indices.tolist():
                x_np = np.asarray(loader.get_sample(int(idx))[0])
                if x_np.ndim == 3 and x_np.shape[0] == 1:
                    x_np = x_np[0]
                x_windows.append(x_np)
            query_x.append(np.stack(x_windows, axis=0))
            query_set_labels.append(set_labels)
            query_y.append(int(subject_id))

    if not support_x or not query_x:
        return 0.0

    def encode_sets(
        all_sets: list[np.ndarray],
        all_set_labels: list[np.ndarray],
    ) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(all_sets), int(batch_size)):
            batch_x = np.stack(all_sets[start : start + int(batch_size)], axis=0)
            batch_y = np.stack(
                all_set_labels[start : start + int(batch_size)],
                axis=0,
            )
            x_support = torch.from_numpy(batch_x).float().unsqueeze(2).to(device)
            y_support = torch.from_numpy(batch_y).long().to(device)
            emb = model.encode_subject(x_support, y_support)
            chunks.append(emb.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    support_emb = encode_sets(support_x, support_set_labels)
    query_emb = encode_sets(query_x, query_set_labels)
    support_labels = np.asarray(support_y, dtype=np.int64)
    query_labels = np.asarray(query_y, dtype=np.int64)

    k_neighbors = max(1, min(int(knn_neighbors), len(support_labels)))
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(support_emb, support_labels)
    pred = knn.predict(query_emb)
    return float(f1_score(query_labels, pred, average="macro"))


@torch.no_grad()
def _evaluate_knn_macro_f1_by_k(
    model: ContrastiveSetEncoder,
    loader: Loader,
    subject_activity_index: Mapping[int, Mapping[int, np.ndarray]],
    activity_ids: Sequence[int],
    k_values: Sequence[int],
    support_sets_per_subject: int,
    max_query_sets_per_subject: int | None,
    knn_neighbors: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[str(int(k))] = _evaluate_val_knn_macro_f1(
            model=model,
            loader=loader,
            subject_activity_index=subject_activity_index,
            activity_ids=activity_ids,
            k_per_class=int(k),
            support_sets_per_subject=support_sets_per_subject,
            max_query_sets_per_subject=max_query_sets_per_subject,
            knn_neighbors=knn_neighbors,
            batch_size=batch_size,
            seed=seed + int(k),
            device=device,
        )
    return metrics


def _mean_metric(metrics: Mapping[str, float]) -> float:
    if not metrics:
        return 0.0
    return float(np.mean([float(v) for v in metrics.values()]))


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    output_root = Path(config.output_root)
    stage_dir = output_root / "02_set_encoder_supcon"
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
    manifest_path = output_root / "shared_splits" / "loso_subject_folds.json"
    folds = build_or_load_loso_folds(session_df, window_df, shared_cfg, manifest_path)
    if config.max_folds is not None:
        folds = folds[: int(config.max_folds)]

    summary_rows: list[dict[str, Any]] = []
    skipped_folds: list[str] = []
    for fold in folds:
        split = split_indices_for_fold(
            session_df,
            window_df,
            type(
                "Tmp",
                (),
                {
                    "train_subject_ids": fold.base_train_subject_ids,
                    "val_subject_ids": fold.val_subject_ids,
                    "test_subject_ids": fold.test_subject_ids,
                },
            )(),
        )
        split_dir = stage_dir / fold.fold_id
        split_dir.mkdir(parents=True, exist_ok=True)
        fold_fp = config_fingerprint(
            {
                "stage": "02_set_encoder_supcon",
                "config": asdict(config),
                "shared_cfg": asdict(shared_cfg),
                "fold": asdict(fold),
            }
        )
        metrics_path = split_dir / "metrics.json"
        ckpt_path = split_dir / "best_set_encoder_supcon.pt"
        if not config.force_rerun and metrics_path.exists() and ckpt_path.exists():
            try:
                existing = json.loads(metrics_path.read_text(encoding="utf-8"))
                if existing.get("config_fingerprint") == fold_fp:
                    print(
                        f"[{fold.fold_id}] skipping (already complete with same settings)"
                    )
                    summary_rows.append(existing)
                    skipped_folds.append(fold.fold_id)
                    continue
            except Exception:
                pass

        post = PostProcessingPipeline(cfg, pre, window_df, split.train_indices)
        samples = post.run()
        loader = Loader(session_df, window_df, post.samples_dir, samples)

        subject_train, activity_ids = _build_subject_activity_index(
            loader, split.train_indices
        )
        subject_val, _ = _build_subject_activity_index(loader, split.val_indices)
        subject_test, _ = _build_subject_activity_index(loader, split.test_indices)
        window_size = int(
            np.asarray(loader.get_sample(split.train_indices[0])[0]).shape[0]
        )
        model = _build_model(
            config,
            cfg.num_of_channels,
            cfg.num_of_activities,
            window_size,
        ).to(torch.device(config.device))
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        criterion = SupConLoss(config.supcon_temperature)

        train_sampler, used_n_subjects, used_k_max = _build_feasible_train_sampler(
            subject_train=subject_train,
            activity_ids=activity_ids,
            config=config,
        )
        if (
            used_n_subjects != config.n_subjects_per_batch
            or used_k_max != config.train_max_k_per_class
        ):
            print(
                f"[{fold.fold_id}] adjusted train sampler: "
                f"n_subjects {config.n_subjects_per_batch}->{used_n_subjects}, "
                f"k_max {config.train_max_k_per_class}->{used_k_max}"
            )
        eval_k_values = k_choices_from_range(config.train_min_k_per_class, used_k_max)
        final_tsne_k_values = tuple(
            int(k)
            for k in config.tsne_k_values
            if int(k) >= int(config.train_min_k_per_class) and int(k) <= int(used_k_max)
        )
        val_n_subjects = max(1, min(config.n_subjects_per_batch, len(subject_val)))
        val_sampler: NMSetsBatchSampler | None = None
        try:
            val_sampler = NMSetsBatchSampler(
                indices_by_subject_activity=subject_val,
                activity_ids=activity_ids,
                min_k_per_class=config.train_min_k_per_class,
                max_k_per_class=used_k_max,
                n_subjects=val_n_subjects,
                m_sets_per_subject=config.val_m_sets_per_subject,
                batches_per_epoch=config.val_batches_per_epoch,
                seed=config.seed + 10_000,
            )
        except ValueError:
            val_sampler = None
        train_loader = DataLoader(
            NMSetsDataset(loader),
            batch_sampler=train_sampler,
            collate_fn=nm_collate,
            num_workers=0,
        )
        val_loader = (
            DataLoader(
                NMSetsDataset(loader),
                batch_sampler=val_sampler,
                collate_fn=nm_collate,
                num_workers=0,
            )
            if val_sampler is not None
            else None
        )

        best_val_macro_f1 = -1.0
        best_epoch = -1
        patience = 0
        history: list[dict] = []
        ckpt_path = split_dir / "best_set_encoder_supcon.pt"

        for epoch in range(1, config.epochs + 1):
            model.train()
            train_losses: list[float] = []
            for batch in tqdm(
                train_loader,
                desc=f"{fold.fold_id} SupCon train {epoch}/{config.epochs}",
                leave=False,
            ):
                x = batch["x"].to(torch.device(config.device))
                y_support = batch["y_support"].to(torch.device(config.device))
                labels = batch["subject_id"].to(torch.device(config.device))
                optimizer.zero_grad(set_to_none=True)
                projected = model(x, y_support)
                loss = criterion(projected, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            model.eval()
            train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
            val_by_k = _evaluate_knn_macro_f1_by_k(
                model=model,
                loader=loader,
                subject_activity_index=subject_val,
                activity_ids=activity_ids,
                k_values=eval_k_values,
                support_sets_per_subject=config.val_support_sets_per_subject,
                max_query_sets_per_subject=config.val_max_query_sets_per_subject,
                knn_neighbors=config.knn_neighbors,
                batch_size=used_n_subjects * config.m_sets_per_subject,
                seed=config.seed + 30_000,
                device=torch.device(config.device),
            )
            val_macro_f1 = _mean_metric(val_by_k)
            row = {
                "epoch": epoch,
                "train_supcon_loss": train_loss,
                "val_knn_macro_f1": val_macro_f1,
                "val_knn_macro_f1_by_k": val_by_k,
            }
            history.append(row)
            print(
                f"[{fold.fold_id}] epoch={epoch} "
                f"train_supcon={train_loss:.4f} val_knn_macro_f1={val_macro_f1:.4f}"
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_epoch = epoch
                patience = 0
                torch.save(
                    {
                        "contrastive_model": model.state_dict(),
                        "set_encoder": model.set_encoder.state_dict(),
                        "projection_head": model.projection_head.state_dict(),
                        "window_size": window_size,
                        "best_epoch": best_epoch,
                        "best_val_knn_macro_f1": best_val_macro_f1,
                        "fold": asdict(fold),
                    },
                    ckpt_path,
                )
            else:
                patience += 1
            if patience >= config.patience:
                break

            if (
                config.tsne_every_n_epochs > 0
                and epoch % config.tsne_every_n_epochs == 0
            ):
                _plot_split_tsne_stratified_by_k(
                    model=model,
                    loader=loader,
                    subject_activity_index=subject_train,
                    activity_ids=activity_ids,
                    split_name=f"train_epoch_{epoch:03d}",
                    output_dir=split_dir,
                    k_values=final_tsne_k_values,
                    sets_per_subject=config.eval_sets_per_subject,
                    batch_size=used_n_subjects * config.m_sets_per_subject,
                    device=torch.device(config.device),
                    seed=config.seed + epoch,
                )
                _plot_split_tsne_stratified_by_k(
                    model=model,
                    loader=loader,
                    subject_activity_index=subject_val,
                    activity_ids=activity_ids,
                    split_name=f"val_epoch_{epoch:03d}",
                    output_dir=split_dir,
                    k_values=final_tsne_k_values,
                    sets_per_subject=config.eval_sets_per_subject,
                    batch_size=used_n_subjects * config.m_sets_per_subject,
                    device=torch.device(config.device),
                    seed=config.seed + 10_000 + epoch,
                )
                _plot_combined_tsne_stratified_by_k(
                    model=model,
                    loader=loader,
                    split_subject_activity_indices={
                        "train": subject_train,
                        "val": subject_val,
                    },
                    activity_ids=activity_ids,
                    output_dir=split_dir,
                    output_name=f"train_val_epoch_{epoch:03d}",
                    title_name=f"train+val, epoch {epoch:03d}",
                    k_values=final_tsne_k_values,
                    sets_per_subject=config.eval_sets_per_subject,
                    batch_size=used_n_subjects * config.m_sets_per_subject,
                    device=torch.device(config.device),
                    seed=config.seed + 20_000 + epoch,
                )

        checkpoint = torch.load(
            ckpt_path, map_location=config.device, weights_only=False
        )
        model.load_state_dict(checkpoint["contrastive_model"])
        val_knn_by_k = _evaluate_knn_macro_f1_by_k(
            model=model,
            loader=loader,
            subject_activity_index=subject_val,
            activity_ids=activity_ids,
            k_values=eval_k_values,
            support_sets_per_subject=config.val_support_sets_per_subject,
            max_query_sets_per_subject=config.val_max_query_sets_per_subject,
            knn_neighbors=config.knn_neighbors,
            batch_size=used_n_subjects * config.m_sets_per_subject,
            seed=config.seed + 30_000,
            device=torch.device(config.device),
        )
        test_knn_by_k = _evaluate_knn_macro_f1_by_k(
            model=model,
            loader=loader,
            subject_activity_index=subject_test,
            activity_ids=activity_ids,
            k_values=eval_k_values,
            support_sets_per_subject=config.val_support_sets_per_subject,
            max_query_sets_per_subject=config.val_max_query_sets_per_subject,
            knn_neighbors=config.knn_neighbors,
            batch_size=used_n_subjects * config.m_sets_per_subject,
            seed=config.seed + 40_000,
            device=torch.device(config.device),
        )
        train_tsne = _plot_split_tsne_stratified_by_k(
            model=model,
            loader=loader,
            subject_activity_index=subject_train,
            activity_ids=activity_ids,
            split_name="train",
            output_dir=split_dir,
            k_values=final_tsne_k_values,
            sets_per_subject=config.eval_sets_per_subject,
            batch_size=used_n_subjects * config.m_sets_per_subject,
            device=torch.device(config.device),
            seed=config.seed + 1,
        )
        val_tsne = _plot_split_tsne_stratified_by_k(
            model=model,
            loader=loader,
            subject_activity_index=subject_val,
            activity_ids=activity_ids,
            split_name="val",
            output_dir=split_dir,
            k_values=final_tsne_k_values,
            sets_per_subject=config.eval_sets_per_subject,
            batch_size=used_n_subjects * config.m_sets_per_subject,
            device=torch.device(config.device),
            seed=config.seed + 2,
        )
        test_tsne = _plot_split_tsne_stratified_by_k(
            model=model,
            loader=loader,
            subject_activity_index=subject_test,
            activity_ids=activity_ids,
            split_name="test",
            output_dir=split_dir,
            k_values=final_tsne_k_values,
            sets_per_subject=config.eval_sets_per_subject,
            batch_size=used_n_subjects * config.m_sets_per_subject,
            device=torch.device(config.device),
            seed=config.seed + 3,
        )
        train_val_tsne = _plot_combined_tsne_stratified_by_k(
            model=model,
            loader=loader,
            split_subject_activity_indices={
                "train": subject_train,
                "val": subject_val,
            },
            activity_ids=activity_ids,
            output_dir=split_dir,
            output_name="train_val",
            title_name="train+val",
            k_values=final_tsne_k_values,
            sets_per_subject=config.eval_sets_per_subject,
            batch_size=used_n_subjects * config.m_sets_per_subject,
            device=torch.device(config.device),
            seed=config.seed + 4,
        )
        train_val_test_tsne = _plot_combined_tsne_stratified_by_k(
            model=model,
            loader=loader,
            split_subject_activity_indices={
                "train": subject_train,
                "val": subject_val,
                "test": subject_test,
            },
            activity_ids=activity_ids,
            output_dir=split_dir,
            output_name="train_val_test",
            title_name="train+val+test",
            k_values=final_tsne_k_values,
            sets_per_subject=config.eval_sets_per_subject,
            batch_size=used_n_subjects * config.m_sets_per_subject,
            device=torch.device(config.device),
            seed=config.seed + 5,
        )
        fold_result = {
            "config_fingerprint": fold_fp,
            "fold_id": fold.fold_id,
            "base_train_subject_ids": fold.base_train_subject_ids,
            "meta_train_subject_ids": fold.meta_train_subject_ids,
            "val_subject_ids": fold.val_subject_ids,
            "test_subject_ids": fold.test_subject_ids,
            "best_epoch": int(best_epoch),
            "used_n_subjects_per_batch": int(used_n_subjects),
            "used_train_max_k_per_class": int(used_k_max),
            "eval_k_values": [int(k) for k in eval_k_values],
            "tsne_k_values": [int(k) for k in final_tsne_k_values],
            "best_val_knn_macro_f1": float(best_val_macro_f1),
            "val_knn_macro_f1_by_k": val_knn_by_k,
            "test_knn_macro_f1_by_k": test_knn_by_k,
            "test_knn_macro_f1_mean": _mean_metric(test_knn_by_k),
            "tsne_train_path": train_tsne,
            "tsne_val_path": val_tsne,
            "tsne_test_path": test_tsne,
            "tsne_train_val_path": train_val_tsne,
            "tsne_train_val_test_path": train_val_test_tsne,
        }
        (split_dir / "metrics.json").write_text(
            json.dumps(fold_result, indent=2), encoding="utf-8"
        )
        (split_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        summary_rows.append(fold_result)

    summary = {
        "config": asdict(config),
        "splits_manifest_path": str(manifest_path),
        "num_folds": len(summary_rows),
        "skipped_folds": skipped_folds,
        "mean_best_val_knn_macro_f1": float(
            sum(r["best_val_knn_macro_f1"] for r in summary_rows)
            / max(1, len(summary_rows))
        ),
        "folds": summary_rows,
    }
    (stage_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

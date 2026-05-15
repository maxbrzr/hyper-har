from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import (
    ROOT,
    SRC,
    SharedConfig,
    build_loader,
    build_or_load_loso_folds,
    build_subject_activity_index,
    config_fingerprint,
    infer_window_size,
    k_choices_from_range,
    prepare_cfg,
    sample_window_array,
    set_seed,
    split_indices_for_fold,
)
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from whar_datasets import PreProcessingPipeline, WHARDatasetID

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


AttentionSetEncoder = _load_module_from_path(
    "meta_attention_set_encoder",
    SRC / "hyper_har" / "set-encoder" / "attention.py",
).AttentionSetEncoder
PrototypicalSetEncoder = _load_module_from_path(
    "meta_simple_set_encoder",
    SRC / "hyper_har" / "set-encoder" / "simple.py",
).PrototypicalSetEncoder


@dataclass(frozen=True)
class Config:
    dataset_id: str = WHARDatasetID.WEAR.value
    datasets_dir: str = str(ROOT / "datasets")
    selected_activities: list[str] | None = None
    window_overlap: float = 0.0
    val_subjects: int = 6
    test_subjects: int = 1
    seed: int = 0

    encoder: str = "attention"
    projection_hidden_dim: int = 128
    projection_dim: int = 64
    train_min_k_per_class: int = 1
    train_max_k_per_class: int = 32
    n_subjects_per_batch: int = 10
    m_sets_per_subject: int = 2
    train_batches_per_epoch: int = 64
    val_batches_per_epoch: int = 32
    val_sets_per_subject: int = 8
    knn_neighbors: int = 1
    validation_metric: str = "silhouette"
    tsne_k_values: tuple[int, ...] = (1, 4, 8, 16, 32)
    supcon_temperature: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 80
    patience: int = 2
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    output_root: str = str(ROOT / "artifacts" / "blackbox_meta_loso")
    max_folds: int | None = None
    force_rerun: bool = False


RUN_CONFIG = Config()


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.shape[0], device=features.device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)
        mask_sum = mask.sum(1).clamp(min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        return (-mean_log_prob_pos).mean()


class ContrastiveSetEncoder(nn.Module):
    def __init__(
        self,
        set_encoder: nn.Module,
        projection_hidden_dim: int,
        projection_dim: int,
    ) -> None:
        super().__init__()
        self.set_encoder = set_encoder
        subject_dim = int(getattr(set_encoder, "output_dim"))
        self.projection_head = nn.Sequential(
            nn.Linear(subject_dim, int(projection_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(projection_hidden_dim), int(projection_dim)),
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
                int(aid): np.asarray(vals, dtype=np.int64)
                for aid, vals in per_activity.items()
            }
            for sid, per_activity in indices_by_subject_activity.items()
        }
        self.activity_ids = [int(aid) for aid in activity_ids]
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
            eligible = [
                int(sid)
                for sid, per_activity in self.indices_by_subject_activity.items()
                if all(
                    len(per_activity.get(aid, np.empty(0, dtype=np.int64))) >= needed
                    for aid in self.activity_ids
                )
            ]
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
                activity_chunks = {
                    int(aid): rng.choice(
                        per_activity[int(aid)],
                        size=self.m_sets_per_subject * k,
                        replace=False,
                    )
                    for aid in self.activity_ids
                }
                for m_idx in range(self.m_sets_per_subject):
                    indices_parts: list[np.ndarray] = []
                    labels_parts: list[np.ndarray] = []
                    for aid in self.activity_ids:
                        start = m_idx * k
                        end = (m_idx + 1) * k
                        indices_parts.append(activity_chunks[int(aid)][start:end])
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


class NMSetsDataset(Dataset[dict[str, torch.Tensor | int]]):
    def __init__(self, loader: Any) -> None:
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader.window_df)

    def __getitem__(self, item: SubjectSetIndices) -> dict[str, torch.Tensor | int]:  # type: ignore[override]
        windows = [
            torch.from_numpy(sample_window_array(self.loader, int(idx)).copy())
            .float()
            .unsqueeze(0)
            for idx in item.set_indices.tolist()
        ]
        return {
            "x_set": torch.stack(windows, dim=0),
            "y_set": torch.tensor(item.set_labels.tolist(), dtype=torch.long),
            "subject_id": int(item.subject_id),
        }


def nm_collate(
    samples: Sequence[dict[str, torch.Tensor | int]],
) -> dict[str, torch.Tensor]:
    x = torch.stack([sample["x_set"] for sample in samples], dim=0)  # type: ignore[list-item]
    y_support = torch.stack([sample["y_set"] for sample in samples], dim=0)  # type: ignore[list-item]
    y = torch.tensor(
        [int(sample["subject_id"]) for sample in samples], dtype=torch.long
    )
    return {"x": x, "y_support": y_support, "subject_id": y}


def _build_feasible_sampler(
    subject_index: Mapping[int, Mapping[int, np.ndarray]],
    activity_ids: Sequence[int],
    config: Config,
    batches_per_epoch: int,
    seed: int,
) -> tuple[NMSetsBatchSampler, int, int]:
    for n_subj in range(int(config.n_subjects_per_batch), 1, -1):
        for k_hi in range(
            int(config.train_max_k_per_class), int(config.train_min_k_per_class) - 1, -1
        ):
            try:
                sampler = NMSetsBatchSampler(
                    indices_by_subject_activity=subject_index,
                    activity_ids=activity_ids,
                    min_k_per_class=config.train_min_k_per_class,
                    max_k_per_class=k_hi,
                    n_subjects=n_subj,
                    m_sets_per_subject=config.m_sets_per_subject,
                    batches_per_epoch=batches_per_epoch,
                    seed=seed,
                )
                return sampler, n_subj, k_hi
            except ValueError:
                continue
    raise ValueError("Could not construct a feasible SupCon sampler.")


def _build_model(
    config: Config, num_channels: int, num_classes: int, window_size: int
) -> ContrastiveSetEncoder:
    backbone = TinierHAR(
        num_channels=num_channels,
        num_classes=num_classes,
        window_size=window_size,
        backbone_config=DEFAULT_CONFIG.backbone,
    )
    se_cfg = replace(DEFAULT_CONFIG.set_encoder, include_global_context=False)
    if config.encoder == "attention":
        set_encoder = AttentionSetEncoder(
            backbone=backbone,
            num_classes=num_classes,
            backbone_train_mode="unfreeze_all",
            force_conv_bn_eval=False,
            set_encoder_config=se_cfg,
        )
    elif config.encoder == "prototypical":
        set_encoder = PrototypicalSetEncoder(
            backbone=backbone,
            num_classes=num_classes,
            backbone_train_mode="unfreeze_all",
            force_conv_bn_eval=False,
            set_encoder_config=se_cfg,
        )
    else:
        raise ValueError("encoder must be 'attention' or 'prototypical'.")
    return ContrastiveSetEncoder(
        set_encoder=set_encoder,
        projection_hidden_dim=config.projection_hidden_dim,
        projection_dim=config.projection_dim,
    )


@torch.no_grad()
def _extract_subject_embeddings(
    model: ContrastiveSetEncoder,
    loader: Any,
    subject_index: Mapping[int, Mapping[int, np.ndarray]],
    activity_ids: Sequence[int],
    k_per_class: int,
    sets_per_subject: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    model.eval()
    embeddings: list[np.ndarray] = []
    subjects: list[int] = []
    for subject_id, per_activity in sorted(subject_index.items()):
        if not all(
            len(per_activity.get(int(aid), np.empty(0, dtype=np.int64)))
            >= int(k_per_class)
            for aid in activity_ids
        ):
            continue
        x_sets: list[np.ndarray] = []
        y_sets: list[np.ndarray] = []
        for _ in range(int(sets_per_subject)):
            indices_parts: list[np.ndarray] = []
            labels_parts: list[np.ndarray] = []
            for aid in activity_ids:
                picked = rng.choice(
                    per_activity[int(aid)], size=int(k_per_class), replace=False
                )
                indices_parts.append(picked)
                labels_parts.append(np.full(int(k_per_class), int(aid), dtype=np.int64))
            set_indices = np.concatenate(indices_parts, axis=0)
            set_labels = np.concatenate(labels_parts, axis=0)
            perm = rng.permutation(set_indices.shape[0])
            x_sets.append(
                np.stack(
                    [
                        sample_window_array(loader, int(idx))
                        for idx in set_indices[perm].tolist()
                    ],
                    axis=0,
                )
            )
            y_sets.append(set_labels[perm])
        x_support = (
            torch.from_numpy(np.stack(x_sets, axis=0)).float().unsqueeze(2).to(device)
        )
        y_support = torch.from_numpy(np.stack(y_sets, axis=0)).long().to(device)
        emb = model.encode_subject(x_support, y_support).detach().cpu().numpy()
        embeddings.append(emb)
        subjects.extend([int(subject_id)] * emb.shape[0])
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(embeddings, axis=0), np.asarray(subjects, dtype=np.int64)


def _knn_val_macro_f1(
    model: ContrastiveSetEncoder,
    loader: Any,
    val_index: Mapping[int, Mapping[int, np.ndarray]],
    activity_ids: Sequence[int],
    config: Config,
    k_per_class: int,
    seed: int,
    device: torch.device,
) -> float:
    ref_x, ref_y = _extract_subject_embeddings(
        model,
        loader,
        val_index,
        activity_ids,
        k_per_class,
        config.val_sets_per_subject,
        seed,
        device,
    )
    query_x, query_y = _extract_subject_embeddings(
        model,
        loader,
        val_index,
        activity_ids,
        k_per_class,
        config.val_sets_per_subject,
        seed + 999,
        device,
    )
    if ref_x.size == 0 or query_x.size == 0 or len(set(ref_y.tolist())) < 2:
        return 0.0
    clf = KNeighborsClassifier(
        n_neighbors=min(int(config.knn_neighbors), len(ref_y)), metric="cosine"
    )
    clf.fit(ref_x, ref_y)
    pred = clf.predict(query_x)
    return float(f1_score(query_y, pred, average="macro", zero_division=0))


def _silhouette_val_score(
    model: ContrastiveSetEncoder,
    loader: Any,
    val_index: Mapping[int, Mapping[int, np.ndarray]],
    activity_ids: Sequence[int],
    config: Config,
    k_per_class: int,
    seed: int,
    device: torch.device,
) -> float:
    embeddings, labels = _extract_subject_embeddings(
        model,
        loader,
        val_index,
        activity_ids,
        k_per_class,
        config.val_sets_per_subject,
        seed,
        device,
    )
    if embeddings.shape[0] < 3:
        return -1.0
    unique_labels = np.unique(labels)
    if unique_labels.shape[0] < 2 or unique_labels.shape[0] >= embeddings.shape[0]:
        return -1.0
    try:
        return float(silhouette_score(embeddings, labels, metric="cosine"))
    except ValueError:
        return -1.0


def _save_tsne_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    seed: int,
) -> str | None:
    if embeddings.shape[0] < 3:
        return None
    unique_labels = sorted(int(x) for x in np.unique(labels).tolist())
    if len(unique_labels) < 2:
        return None
    perplexity = min(30.0, max(2.0, float(embeddings.shape[0] - 1) / 3.0))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        metric="cosine",
        random_state=int(seed),
    )
    coords = reducer.fit_transform(embeddings)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = np.asarray([label_to_idx[int(x)] for x in labels], dtype=np.int64)
    cmap = plt.get_cmap("tab20", max(1, len(unique_labels)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 7.0), dpi=170)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=encoded,
        cmap=cmap,
        vmin=0,
        vmax=max(0, len(unique_labels) - 1),
        s=34,
        alpha=0.86,
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
    ax.legend(
        handles=handles,
        title="Subject",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
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
    unique_subjects = sorted(int(x) for x in np.unique(subject_labels).tolist())
    if len(unique_subjects) < 2:
        return None
    perplexity = min(30.0, max(2.0, float(embeddings.shape[0] - 1) / 3.0))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        metric="cosine",
        random_state=int(seed),
    )
    coords = reducer.fit_transform(embeddings)
    subject_to_idx = {label: idx for idx, label in enumerate(unique_subjects)}
    encoded = np.asarray([subject_to_idx[int(x)] for x in subject_labels], dtype=np.int64)
    cmap = plt.get_cmap("tab20", max(1, len(unique_subjects)))
    split_markers = {
        "train": "o",
        "val": "^",
        "test": "X",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.0, 7.3), dpi=170)
    for split_name in ("train", "val", "test"):
        mask = split_labels == split_name
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=encoded[mask],
            cmap=cmap,
            vmin=0,
            vmax=max(0, len(unique_subjects) - 1),
            s=34 if split_name == "train" else 48,
            alpha=0.86,
            marker=split_markers[split_name],
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
            marker=split_markers[split_name],
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label=split_name,
        )
        for split_name in ("train", "val", "test")
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


def _save_split_tsne_for_k_values(
    model: ContrastiveSetEncoder,
    loader: Any,
    train_index: Mapping[int, Mapping[int, np.ndarray]],
    val_index: Mapping[int, Mapping[int, np.ndarray]],
    test_index: Mapping[int, Mapping[int, np.ndarray]],
    activity_ids: Sequence[int],
    config: Config,
    split_dir: Path,
    fold_id: str,
    split_idx: int,
    device: torch.device,
) -> dict[str, str | None]:
    paths: dict[str, str | None] = {}
    split_indexes = {
        "train": train_index,
        "val": val_index,
        "test": test_index,
    }
    for k in config.tsne_k_values:
        k = int(k)
        embeddings_parts: list[np.ndarray] = []
        subject_parts: list[np.ndarray] = []
        split_parts: list[np.ndarray] = []
        for split_name, subject_index in split_indexes.items():
            embeddings, subjects = _extract_subject_embeddings(
                model,
                loader,
                subject_index,
                activity_ids,
                k,
                config.val_sets_per_subject,
                config.seed + 70_000 + 1_000 * split_idx + 10 * k + len(split_name),
                device,
            )
            if embeddings.size == 0:
                continue
            embeddings_parts.append(embeddings)
            subject_parts.append(subjects)
            split_parts.append(np.full(subjects.shape[0], split_name, dtype=object))
        output_path = split_dir / f"tsne_splits_k{k}.png"
        if not embeddings_parts:
            paths[str(k)] = None
            continue
        paths[str(k)] = _save_tsne_split_plot(
            np.concatenate(embeddings_parts, axis=0),
            np.concatenate(subject_parts, axis=0),
            np.concatenate(split_parts, axis=0),
            output_path,
            f"{fold_id} set-encoder embeddings by split (K={k})",
            config.seed + 80_000 + split_idx + k,
        )
    return paths


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)
    output_root = Path(config.output_root)
    stage_dir = output_root / "01_set_encoder_supcon"
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
    for split_idx, fold in enumerate(folds):
        split = split_indices_for_fold(session_df, window_df, fold)
        split_dir = stage_dir / fold.fold_id
        split_dir.mkdir(parents=True, exist_ok=True)
        fold_fp = config_fingerprint(
            {
                "stage": "01_set_encoder_supcon",
                "config": asdict(config),
                "shared_cfg": asdict(shared_cfg),
                "fold": asdict(fold),
            }
        )
        metrics_path = split_dir / "metrics.json"
        ckpt_path = split_dir / "best_set_encoder_supcon.pt"
        if not config.force_rerun and metrics_path.exists() and ckpt_path.exists():
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if existing.get("config_fingerprint") == fold_fp:
                print(f"[{fold.fold_id}] skipping (already complete)")
                summary_rows.append(existing)
                skipped_folds.append(fold.fold_id)
                continue

        sample_indices = sorted(
            set(split.train_indices + split.val_indices + split.test_indices)
        )
        loader = build_loader(cfg, session_df, pre, window_df, sample_indices)
        window_size = infer_window_size(loader, sample_indices)
        num_channels = int(cfg.num_of_channels)
        num_classes = int(cfg.num_of_activities)

        train_index, activity_ids = build_subject_activity_index(
            loader, split.train_indices
        )
        val_index, _ = build_subject_activity_index(loader, split.val_indices)
        test_index, _ = build_subject_activity_index(loader, split.test_indices)
        train_sampler, effective_n, effective_k_hi = _build_feasible_sampler(
            train_index,
            activity_ids,
            config,
            config.train_batches_per_epoch,
            config.seed + split_idx,
        )
        dataset = NMSetsDataset(loader)
        train_loader = DataLoader(
            dataset, batch_sampler=train_sampler, collate_fn=nm_collate
        )

        model = _build_model(config, num_channels, num_classes, window_size).to(device)
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        criterion = SupConLoss(config.supcon_temperature)

        best_val_silhouette = float("-inf")
        best_val_knn_f1 = 0.0
        best_epoch = -1
        patience_counter = 0
        history: list[dict[str, Any]] = []
        eval_k = min(k_choices_from_range(config.train_min_k_per_class, effective_k_hi))
        val_knn_seed = config.seed + 1000 * split_idx
        val_silhouette_seed = config.seed + 2000 * split_idx
        for epoch in range(1, int(config.epochs) + 1):
            model.train()
            losses: list[float] = []
            for batch in tqdm(
                train_loader,
                desc=f"{fold.fold_id} supcon {epoch}/{config.epochs}",
                leave=False,
            ):
                x = batch["x"].to(device)
                y_support = batch["y_support"].to(device)
                subject_id = batch["subject_id"].to(device)
                optimizer.zero_grad(set_to_none=True)
                z = model(x, y_support)
                loss = criterion(z, subject_id)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite SupCon loss in {fold.fold_id}."
                    )
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            val_knn_f1 = _knn_val_macro_f1(
                model,
                loader,
                val_index,
                activity_ids,
                config,
                eval_k,
                val_knn_seed,
                device,
            )
            val_silhouette = _silhouette_val_score(
                model,
                loader,
                val_index,
                activity_ids,
                config,
                eval_k,
                val_silhouette_seed,
                device,
            )
            row = {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "val_silhouette": float(val_silhouette),
                "val_knn_macro_f1": float(val_knn_f1),
                "eval_k_per_class": int(eval_k),
                "effective_subjects_per_batch": int(effective_n),
                "effective_max_k_per_class": int(effective_k_hi),
            }
            history.append(row)
            print(
                f"[{fold.fold_id}] epoch={epoch} loss={row['train_loss']:.4f} "
                f"val_silhouette={val_silhouette:.4f} "
                f"val_knn_f1={val_knn_f1:.4f}"
            )
            if val_silhouette > best_val_silhouette:
                best_val_silhouette = float(val_silhouette)
                best_val_knn_f1 = float(val_knn_f1)
                best_epoch = int(epoch)
                patience_counter = 0
                torch.save(
                    {
                        "set_encoder": model.set_encoder.state_dict(),
                        "projection_head": model.projection_head.state_dict(),
                        "best_epoch": best_epoch,
                        "best_val_silhouette": best_val_silhouette,
                        "best_val_knn_macro_f1": best_val_knn_f1,
                        "config": asdict(config),
                    },
                    ckpt_path,
                )
            else:
                patience_counter += 1
            if patience_counter >= int(config.patience):
                break

        if not ckpt_path.exists():
            torch.save(
                {
                    "set_encoder": model.set_encoder.state_dict(),
                    "projection_head": model.projection_head.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_silhouette": best_val_silhouette,
                    "best_val_knn_macro_f1": best_val_knn_f1,
                    "config": asdict(config),
                },
                ckpt_path,
            )
        best_payload = torch.load(
            ckpt_path, map_location=config.device, weights_only=False
        )
        model.set_encoder.load_state_dict(best_payload["set_encoder"])
        model.projection_head.load_state_dict(best_payload["projection_head"])
        tsne_paths_by_k = _save_split_tsne_for_k_values(
            model,
            loader,
            train_index,
            val_index,
            test_index,
            activity_ids,
            config,
            split_dir,
            fold.fold_id,
            split_idx,
            device,
        )
        fold_result = {
            "config_fingerprint": fold_fp,
            "fold_id": fold.fold_id,
            "train_subject_ids": fold.train_subject_ids,
            "val_subject_ids": fold.val_subject_ids,
            "test_subject_ids": fold.test_subject_ids,
            "best_epoch": int(best_epoch),
            "best_val_silhouette": float(best_val_silhouette),
            "best_val_knn_macro_f1": float(best_val_knn_f1),
            "validation_metric": config.validation_metric,
            "eval_k_per_class": int(eval_k),
            "tsne_k_values": [int(k) for k in config.tsne_k_values],
            "tsne_paths_by_k": tsne_paths_by_k,
            "checkpoint_path": str(ckpt_path),
            "effective_subjects_per_batch": int(effective_n),
            "effective_max_k_per_class": int(effective_k_hi),
        }
        metrics_path.write_text(json.dumps(fold_result, indent=2), encoding="utf-8")
        (split_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        summary_rows.append(fold_result)

    summary = {
        "config": asdict(config),
        "splits_manifest_path": str(manifest_path),
        "num_folds": len(summary_rows),
        "skipped_folds": skipped_folds,
        "mean_best_val_silhouette": float(
            np.mean([row["best_val_silhouette"] for row in summary_rows])
        )
        if summary_rows
        else 0.0,
        "mean_best_val_knn_macro_f1": float(
            np.mean([row["best_val_knn_macro_f1"] for row in summary_rows])
        )
        if summary_rows
        else 0.0,
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

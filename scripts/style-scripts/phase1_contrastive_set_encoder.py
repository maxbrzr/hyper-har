from __future__ import annotations

import importlib.util
import json
import random
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from whar_datasets import (
    Loader,
    PostProcessingPipeline,
    PreProcessingPipeline,
    WHARDatasetID,
    get_dataset_cfg,
)

from hyper_har.backbone.tinierhar import TinierHAR
from hyper_har.config import DEFAULT_CONFIG

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
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
    "hyper_har_set_encoder_attention_contrastive",
    SRC / "hyper_har" / "set-encoder" / "attention.py",
)
SET_ENCODER_SIMPLE_MODULE = _load_module_from_path(
    "hyper_har_set_encoder_simple_contrastive",
    SRC / "hyper_har" / "set-encoder" / "simple.py",
)
AttentionSetEncoder = SET_ENCODER_ATTENTION_MODULE.AttentionSetEncoder
PrototypicalSetEncoder = SET_ENCODER_SIMPLE_MODULE.PrototypicalSetEncoder


@dataclass(frozen=True)
class ContrastiveRunConfig:
    dataset_id: str = WHARDatasetID.WEAR.value
    activity: str = "Walking"
    encoder: str = "attention"
    set_size: int = 8
    train_min_set_size: int = 1
    train_max_set_size: int = 32
    triplets_per_batch: int = 16
    batches_per_epoch: int = 128
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    margin: float = 1.0
    projection_hidden_dim: int = 128
    projection_dim: int = 64
    eval_sets_per_subject: int = 16
    tsne_every_n_epochs: int = 1
    knn_neighbors: int = 5
    seed: int = 0
    val_size: float = 0.15
    test_size: float = 0.15
    output_dir: str = str(ROOT / "artifacts" / "contrastive_set_encoder")
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


RUN_CONFIG = ContrastiveRunConfig(
    dataset_id=WHARDatasetID.WEAR.value,
    activity="Walking",
    encoder="attention",  # "attention" or "proto"
    set_size=8,  # validation/test K during training
    train_min_set_size=1,
    train_max_set_size=32,
    triplets_per_batch=16,
    batches_per_epoch=128,
    epochs=50,
    learning_rate=1e-4,
    weight_decay=1e-4,
    margin=1.0,
    projection_hidden_dim=128,
    projection_dim=64,
    eval_sets_per_subject=16,
    tsne_every_n_epochs=1,
    knn_neighbors=5,
    seed=0,
    val_size=0.15,
    test_size=0.15,
    output_dir=str(ROOT / "artifacts" / "contrastive_set_encoder"),
)


@dataclass(frozen=True)
class WindowSplits:
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]


@dataclass(frozen=True)
class TripletSetIndices:
    anchor_subject_id: int
    negative_subject_id: int
    anchor_indices: np.ndarray
    positive_indices: np.ndarray
    negative_indices: np.ndarray


@dataclass(frozen=True)
class EvaluationResult:
    split_name: str
    accuracy: float
    num_train_embeddings: int
    num_eval_embeddings: int
    num_subjects: int
    plot_path: str | None


class ContrastiveSetEncoder(nn.Module):
    """Set encoder plus projection head for triplet training.

    Evaluation should call ``encode_subject`` to bypass this projection head and
    use raw c_subject vectors.
    """

    def __init__(
        self,
        set_encoder: nn.Module,
        projection_hidden_dim: int = 128,
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        c_subject_dim = int(getattr(set_encoder, "output_dim"))
        self.set_encoder = set_encoder
        self.projection_head = nn.Sequential(
            nn.Linear(c_subject_dim, projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(projection_hidden_dim, projection_dim),
        )

    def encode_subject(
        self, x_support: torch.Tensor, y_support: torch.Tensor
    ) -> torch.Tensor:
        return self.set_encoder(x_support, y_support)

    def forward(self, x_support: torch.Tensor, y_support: torch.Tensor) -> torch.Tensor:
        c_subject = self.encode_subject(x_support, y_support)
        return self.projection_head(c_subject)


class SubjectTripletBatchSampler(Sampler[list[TripletSetIndices]]):
    def __init__(
        self,
        indices_by_subject: Mapping[int, np.ndarray],
        min_set_size: int,
        max_set_size: int,
        triplets_per_batch: int,
        batches_per_epoch: int,
        seed: int,
    ) -> None:
        self.indices_by_subject = {
            int(subject_id): np.asarray(indices, dtype=np.int64)
            for subject_id, indices in indices_by_subject.items()
        }
        self.min_set_size = int(min_set_size)
        self.max_set_size = int(max_set_size)
        if self.min_set_size < 1:
            raise ValueError("min_set_size must be at least 1.")
        if self.max_set_size < self.min_set_size:
            raise ValueError("max_set_size must be >= min_set_size.")
        self.triplets_per_batch = int(triplets_per_batch)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

        self.eligible_subject_ids_by_k: dict[int, list[int]] = {}
        for k in range(self.min_set_size, self.max_set_size + 1):
            eligible = sorted(
                subject_id
                for subject_id, indices in self.indices_by_subject.items()
                if len(indices) >= 2 * k
            )
            if len(eligible) >= 2:
                self.eligible_subject_ids_by_k[k] = eligible
        self.valid_set_sizes = tuple(sorted(self.eligible_subject_ids_by_k))
        if not self.valid_set_sizes:
            raise ValueError(
                "Triplet sampling needs at least two subjects with 2*K windows. "
                f"No feasible K found in range [{self.min_set_size}, {self.max_set_size}]."
            )

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[TripletSetIndices]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        valid_set_sizes = np.asarray(self.valid_set_sizes, dtype=np.int64)
        for _ in range(self.batches_per_epoch):
            set_size = int(rng.choice(valid_set_sizes))
            subject_ids = np.asarray(
                self.eligible_subject_ids_by_k[set_size], dtype=np.int64
            )
            batch: list[TripletSetIndices] = []
            for _triplet_idx in range(self.triplets_per_batch):
                anchor_subject = int(rng.choice(subject_ids))
                negative_pool = subject_ids[subject_ids != anchor_subject]
                negative_subject = int(rng.choice(negative_pool))

                anchor_source = self.indices_by_subject[anchor_subject]
                negative_source = self.indices_by_subject[negative_subject]
                anchor_positive = rng.choice(
                    anchor_source, size=2 * set_size, replace=False
                )
                negative = rng.choice(negative_source, size=set_size, replace=False)
                batch.append(
                    TripletSetIndices(
                        anchor_subject_id=anchor_subject,
                        negative_subject_id=negative_subject,
                        anchor_indices=anchor_positive[:set_size],
                        positive_indices=anchor_positive[set_size:],
                        negative_indices=negative,
                    )
                )
            yield batch


class ContrastiveTripletDataset(Dataset[dict[str, torch.Tensor | int]]):
    def __init__(self, loader: Loader) -> None:
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader.window_df)

    def _window_tensor(self, index: int) -> torch.Tensor:
        sample = self.loader.get_sample(int(index))
        if not sample:
            raise ValueError(f"Empty sample for window index {index}.")
        x_np = np.asarray(sample[0])
        if x_np.ndim == 3 and x_np.shape[0] == 1:
            x_np = x_np[0]
        if x_np.ndim != 2:
            raise ValueError(
                f"Expected sample shape (window, sensors), got {tuple(x_np.shape)}."
            )
        return torch.from_numpy(x_np.copy()).float().unsqueeze(0)

    def _set_tensor(self, indices: Sequence[int]) -> torch.Tensor:
        windows = [self._window_tensor(int(index)) for index in indices]
        return torch.stack(windows, dim=0)

    def __getitem__(self, triplet: TripletSetIndices) -> dict[str, torch.Tensor | int]:
        if not isinstance(triplet, TripletSetIndices):
            raise TypeError(
                "ContrastiveTripletDataset expects TripletSetIndices from "
                "SubjectTripletBatchSampler."
            )
        return {
            "anchor": self._set_tensor(triplet.anchor_indices),
            "positive": self._set_tensor(triplet.positive_indices),
            "negative": self._set_tensor(triplet.negative_indices),
            "anchor_subject_id": triplet.anchor_subject_id,
            "negative_subject_id": triplet.negative_subject_id,
        }


def triplet_collate(
    samples: Sequence[dict[str, torch.Tensor | int]],
) -> dict[str, torch.Tensor]:
    def stack_tensor_key(key: str) -> torch.Tensor:
        tensors: list[torch.Tensor] = []
        for sample in samples:
            value = sample[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Expected tensor for batch key '{key}'.")
            tensors.append(value)
        return torch.stack(tensors)

    return {
        "anchor": stack_tensor_key("anchor"),
        "positive": stack_tensor_key("positive"),
        "negative": stack_tensor_key("negative"),
        "anchor_subject_id": torch.tensor(
            [int(sample["anchor_subject_id"]) for sample in samples], dtype=torch.long
        ),
        "negative_subject_id": torch.tensor(
            [int(sample["negative_subject_id"]) for sample in samples], dtype=torch.long
        ),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_activity(cfg: Any, requested_activity: str) -> str:
    available = list(getattr(cfg, "available_activities"))
    by_lower = {name.lower(): name for name in available}
    requested_lower = requested_activity.lower()
    if requested_lower in by_lower:
        return by_lower[requested_lower]

    for name in available:
        if "walk" in name.lower():
            return name

    dynamic_preferences = (
        "Jogging",
        "Running",
        "Run",
        "Jogging Butt Kicks",
        "Jogging Sidesteps",
        "Jogging Skipping",
    )
    for preferred in dynamic_preferences:
        if preferred.lower() in by_lower:
            return by_lower[preferred.lower()]

    raise ValueError(
        f"Could not find walking or a dynamic fallback activity in {available}."
    )


def prepare_one_activity_cfg(
    dataset_id: WHARDatasetID, datasets_dir: Path, activity: str
) -> tuple[Any, str]:
    base_cfg = get_dataset_cfg(dataset_id, datasets_dir=str(datasets_dir))
    cfg = (
        base_cfg.model_copy(deep=True)
        if hasattr(base_cfg, "model_copy")
        else base_cfg.copy(deep=True)
    )
    selected_activity = resolve_activity(cfg, activity)
    cfg.selected_activities = [selected_activity]
    cfg.num_of_activities = 1
    cfg.window_overlap = 0.0
    if hasattr(cfg, "overlap"):
        cfg.overlap = 0.0
    return cfg, selected_activity


def metadata_for_indices(
    session_df: pd.DataFrame,
    window_df: pd.DataFrame,
    indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    if indices is None:
        subset = window_df[["session_id"]].copy()
    else:
        subset = window_df.loc[list(indices), ["session_id"]].copy()
    subset["window_index"] = subset.index.astype(int)
    session_meta = (
        session_df[["session_id", "subject_id", "activity_id"]]
        .drop_duplicates("session_id")
        .copy()
    )
    merged = subset.merge(session_meta, on="session_id", how="left")
    if merged[["subject_id", "activity_id"]].isna().any().any():
        raise ValueError("Missing subject/activity metadata for one or more windows.")
    return merged


def split_windows_by_subject(
    session_df: pd.DataFrame,
    window_df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> WindowSplits:
    meta = metadata_for_indices(session_df, window_df)
    indices = meta["window_index"].to_numpy(dtype=np.int64)
    subjects = meta["subject_id"].to_numpy(dtype=np.int64)

    train_val_indices, test_indices, train_val_subjects, _test_subjects = (
        train_test_split(
            indices,
            subjects,
            test_size=test_size,
            random_state=seed,
            stratify=subjects,
        )
    )
    relative_val_size = val_size / max(1e-12, 1.0 - test_size)
    train_indices, val_indices, _train_subjects, _val_subjects = train_test_split(
        train_val_indices,
        train_val_subjects,
        test_size=relative_val_size,
        random_state=seed + 1,
        stratify=train_val_subjects,
    )
    return WindowSplits(
        train_indices=sorted(int(index) for index in train_indices.tolist()),
        val_indices=sorted(int(index) for index in val_indices.tolist()),
        test_indices=sorted(int(index) for index in test_indices.tolist()),
    )


def indices_by_subject(
    loader: Loader,
    indices: Sequence[int],
    min_windows: int = 1,
) -> dict[int, np.ndarray]:
    meta = metadata_for_indices(loader.session_df, loader.window_df, indices)
    grouped = meta.groupby("subject_id")["window_index"]
    return {
        int(subject_id): values.to_numpy(dtype=np.int64)
        for subject_id, values in grouped
        if len(values) >= min_windows
    }


def sample_window_array(loader: Loader, index: int) -> np.ndarray:
    sample = loader.get_sample(index)
    if not sample:
        raise ValueError(f"Empty sample for window index {index}.")
    x_np = np.asarray(sample[0])
    if x_np.ndim == 3 and x_np.shape[0] == 1:
        x_np = x_np[0]
    if x_np.ndim != 2:
        raise ValueError(
            f"Expected sample shape (window, sensors), got {tuple(x_np.shape)}."
        )
    return x_np


def infer_window_size(loader: Loader, indices: Sequence[int]) -> int:
    if not indices:
        raise ValueError("Cannot infer window size from an empty split.")
    x_np = sample_window_array(loader, int(indices[0]))
    return int(x_np.shape[0])


def build_model(
    encoder_name: str,
    num_channels: int,
    window_size: int,
    projection_hidden_dim: int,
    projection_dim: int,
) -> ContrastiveSetEncoder:
    backbone = TinierHAR(
        num_channels=num_channels,
        num_classes=1,
        window_size=window_size,
        backbone_config=DEFAULT_CONFIG.backbone,
    )
    set_encoder_cfg = replace(
        DEFAULT_CONFIG.set_encoder,
        include_global_context=False,
    )
    if encoder_name == "attention":
        set_encoder = AttentionSetEncoder(
            backbone=backbone,
            num_classes=1,
            backbone_train_mode="unfreeze_all",
            force_conv_bn_eval=False,
            set_encoder_config=set_encoder_cfg,
        )
    elif encoder_name == "proto":
        set_encoder = PrototypicalSetEncoder(
            backbone=backbone,
            num_classes=1,
            backbone_train_mode="unfreeze_all",
            force_conv_bn_eval=False,
            set_encoder_config=set_encoder_cfg,
        )
    else:
        raise ValueError(f"Unsupported encoder '{encoder_name}'.")

    return ContrastiveSetEncoder(
        set_encoder=set_encoder,
        projection_hidden_dim=projection_hidden_dim,
        projection_dim=projection_dim,
    )


@torch.no_grad()
def extract_subject_embeddings(
    model: ContrastiveSetEncoder,
    loader: Loader,
    indices: Sequence[int],
    set_size: int,
    sets_per_subject: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    subject_to_indices = indices_by_subject(loader, indices, min_windows=set_size)
    rng = np.random.default_rng(seed)
    model.eval()

    all_sets: list[np.ndarray] = []
    all_labels: list[int] = []
    for subject_id, subject_indices in sorted(subject_to_indices.items()):
        replace_samples = len(subject_indices) < set_size
        for _ in range(sets_per_subject):
            picked = rng.choice(subject_indices, size=set_size, replace=replace_samples)
            all_sets.append(
                np.stack(
                    [sample_window_array(loader, int(idx)) for idx in picked], axis=0
                )
            )
            all_labels.append(int(subject_id))

    if not all_sets:
        raise ValueError("No evaluation sets could be sampled.")

    embeddings: list[np.ndarray] = []
    for start in range(0, len(all_sets), batch_size):
        batch_np = np.stack(all_sets[start : start + batch_size], axis=0)
        x_support = torch.from_numpy(batch_np).float().unsqueeze(2).to(device)
        y_support = torch.zeros(
            (x_support.size(0), x_support.size(1)),
            dtype=torch.long,
            device=device,
        )
        raw = model.encode_subject(x_support, y_support)
        embeddings.append(raw.detach().cpu().numpy())

    return np.concatenate(embeddings, axis=0), np.asarray(all_labels, dtype=np.int64)


def save_tsne_plot(
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
    cmap = plt.cm.get_cmap("tab20", n_classes)

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
    subject_handles = [
        plt.Line2D(
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
    ax.legend(handles=subject_handles, title="Subject", loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


def evaluate_subject_knn(
    model: ContrastiveSetEncoder,
    loader: Loader,
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    split_name: str,
    output_dir: Path,
    config: ContrastiveRunConfig,
    device: torch.device,
) -> EvaluationResult:
    train_embeddings, train_labels = extract_subject_embeddings(
        model=model,
        loader=loader,
        indices=train_indices,
        set_size=config.set_size,
        sets_per_subject=config.eval_sets_per_subject,
        batch_size=config.triplets_per_batch,
        seed=config.seed + 10,
        device=device,
    )
    eval_embeddings, eval_labels = extract_subject_embeddings(
        model=model,
        loader=loader,
        indices=eval_indices,
        set_size=config.set_size,
        sets_per_subject=config.eval_sets_per_subject,
        batch_size=config.triplets_per_batch,
        seed=config.seed + 20,
        device=device,
    )
    n_neighbors = max(1, min(config.knn_neighbors, len(train_embeddings)))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_embeddings, train_labels)
    predictions = knn.predict(eval_embeddings)
    accuracy = float(accuracy_score(eval_labels, predictions))

    return EvaluationResult(
        split_name=split_name,
        accuracy=accuracy,
        num_train_embeddings=int(len(train_embeddings)),
        num_eval_embeddings=int(len(eval_embeddings)),
        num_subjects=int(len(set(eval_labels.tolist()))),
        plot_path=None,
    )


def plot_split_tsne(
    model: ContrastiveSetEncoder,
    loader: Loader,
    indices: Sequence[int],
    split_name: str,
    output_dir: Path,
    config: ContrastiveRunConfig,
    device: torch.device,
    seed_offset: int,
) -> str | None:
    embeddings, labels = extract_subject_embeddings(
        model=model,
        loader=loader,
        indices=indices,
        set_size=config.set_size,
        sets_per_subject=config.eval_sets_per_subject,
        batch_size=config.triplets_per_batch,
        seed=config.seed + seed_offset,
        device=device,
    )
    return save_tsne_plot(
        embeddings=embeddings,
        labels=labels,
        output_path=output_dir / f"tsne_{split_name}.png",
        title=f"Raw c_subject t-SNE ({split_name}, K={config.set_size})",
        seed=config.seed + seed_offset,
    )


def run_training(config: ContrastiveRunConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = WHARDatasetID(config.dataset_id)
    cfg, selected_activity = prepare_one_activity_cfg(
        dataset_id=dataset_id,
        datasets_dir=ROOT / "datasets",
        activity=config.activity,
    )
    print(
        f"Using dataset={dataset_id.value}, activity='{selected_activity}', "
        f"selected_activities={cfg.selected_activities}, window_overlap={cfg.window_overlap}"
    )

    pre_pipeline = PreProcessingPipeline(cfg)
    _raw_df, session_df, window_df = pre_pipeline.run()
    activity_ids = sorted(int(x) for x in session_df["activity_id"].unique().tolist())
    if activity_ids != [0]:
        raise ValueError(
            "Expected one selected activity remapped to activity_id=0, "
            f"got activity_ids={activity_ids}."
        )

    splits = split_windows_by_subject(
        session_df=session_df,
        window_df=window_df,
        val_size=config.val_size,
        test_size=config.test_size,
        seed=config.seed,
    )
    print(
        "Window split sizes: "
        f"train={len(splits.train_indices)}, val={len(splits.val_indices)}, "
        f"test={len(splits.test_indices)}"
    )

    post_pipeline = PostProcessingPipeline(
        cfg, pre_pipeline, window_df, splits.train_indices
    )
    samples = post_pipeline.run()
    loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)

    window_size = infer_window_size(loader, splits.train_indices)
    model = build_model(
        encoder_name=config.encoder,
        num_channels=cfg.num_of_channels,
        window_size=window_size,
        projection_hidden_dim=config.projection_hidden_dim,
        projection_dim=config.projection_dim,
    )
    device = torch.device(config.device)
    model.to(device)

    train_subject_indices = indices_by_subject(
        loader, splits.train_indices, min_windows=2 * config.train_min_set_size
    )
    sampler = SubjectTripletBatchSampler(
        indices_by_subject=train_subject_indices,
        min_set_size=config.train_min_set_size,
        max_set_size=config.train_max_set_size,
        triplets_per_batch=config.triplets_per_batch,
        batches_per_epoch=config.batches_per_epoch,
        seed=config.seed,
    )
    print(
        f"Training with random K in [{config.train_min_set_size}, {config.train_max_set_size}]; "
        f"feasible K values in this split={sampler.valid_set_sizes}; "
        f"validation/test during training use K={config.set_size}."
    )
    dataset = ContrastiveTripletDataset(loader)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=triplet_collate,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.TripletMarginLoss(margin=config.margin)

    history: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    checkpoint_path = output_dir / "best_contrastive_set_encoder.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        positive_distances: list[float] = []
        negative_distances: list[float] = []

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for batch in progress:
            anchor = batch["anchor"].to(device)
            positive = batch["positive"].to(device)
            negative = batch["negative"].to(device)
            y_support = torch.zeros(
                (anchor.size(0), anchor.size(1)), dtype=torch.long, device=device
            )

            optimizer.zero_grad(set_to_none=True)
            embed_anchor = model(anchor, y_support)
            embed_positive = model(positive, y_support)
            embed_negative = model(negative, y_support)
            loss = criterion(embed_anchor, embed_positive, embed_negative)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pos_dist = torch.norm(embed_anchor - embed_positive, p=2, dim=1).mean()
                neg_dist = torch.norm(embed_anchor - embed_negative, p=2, dim=1).mean()
            epoch_losses.append(float(loss.item()))
            positive_distances.append(float(pos_dist.item()))
            negative_distances.append(float(neg_dist.item()))
            progress.set_postfix(
                loss=f"{np.mean(epoch_losses):.4f}",
                pos=f"{np.mean(positive_distances):.3f}",
                neg=f"{np.mean(negative_distances):.3f}",
            )

        val_result = evaluate_subject_knn(
            model=model,
            loader=loader,
            train_indices=splits.train_indices,
            eval_indices=splits.val_indices,
            split_name="val",
            output_dir=output_dir,
            config=config,
            device=device,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "positive_distance": float(np.mean(positive_distances)),
            "negative_distance": float(np.mean(negative_distances)),
            "val_subject_knn_accuracy": val_result.accuracy,
        }
        if config.tsne_every_n_epochs > 0 and (epoch % config.tsne_every_n_epochs == 0):
            train_plot_epoch = plot_split_tsne(
                model=model,
                loader=loader,
                indices=splits.train_indices,
                split_name=f"train_epoch_{epoch:03d}",
                output_dir=output_dir,
                config=config,
                device=device,
                seed_offset=1000 + epoch,
            )
            val_plot_epoch = plot_split_tsne(
                model=model,
                loader=loader,
                indices=splits.val_indices,
                split_name=f"val_epoch_{epoch:03d}",
                output_dir=output_dir,
                config=config,
                device=device,
                seed_offset=2000 + epoch,
            )
            epoch_row["train_tsne_path"] = train_plot_epoch or ""
            epoch_row["val_tsne_path"] = val_plot_epoch or ""
        history.append(epoch_row)
        print(
            f"Epoch {epoch:03d}: loss={epoch_row['train_loss']:.4f}, "
            f"pos={epoch_row['positive_distance']:.3f}, "
            f"neg={epoch_row['negative_distance']:.3f}, "
            f"val_knn_acc={val_result.accuracy:.4f}"
        )

        if val_result.accuracy > best_val_accuracy:
            best_val_accuracy = val_result.accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "selected_activity": selected_activity,
                    "window_size": window_size,
                    "best_epoch": epoch,
                    "best_val_subject_knn_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_tsne_path = plot_split_tsne(
        model=model,
        loader=loader,
        indices=splits.train_indices,
        split_name="train",
        output_dir=output_dir,
        config=config,
        device=device,
        seed_offset=30,
    )
    val_tsne_path = plot_split_tsne(
        model=model,
        loader=loader,
        indices=splits.val_indices,
        split_name="val",
        output_dir=output_dir,
        config=config,
        device=device,
        seed_offset=40,
    )
    test_result = evaluate_subject_knn(
        model=model,
        loader=loader,
        train_indices=splits.train_indices,
        eval_indices=splits.test_indices,
        split_name="test",
        output_dir=output_dir,
        config=config,
        device=device,
    )

    result = {
        "config": asdict(config),
        "selected_activity": selected_activity,
        "splits": asdict(splits),
        "history": history,
        "best_checkpoint_path": str(checkpoint_path),
        "best_val_subject_knn_accuracy": best_val_accuracy,
        "tsne_train_path": train_tsne_path,
        "tsne_val_path": val_tsne_path,
        "test": asdict(test_result),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(
        f"Finished. Best val KNN accuracy={best_val_accuracy:.4f}; "
        f"test KNN accuracy={test_result.accuracy:.4f}"
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    test_tsne_path = plot_split_tsne(
        model=model,
        loader=loader,
        indices=splits.test_indices,
        split_name="test",
        output_dir=output_dir,
        config=config,
        device=device,
        seed_offset=50,
    )
    if train_tsne_path is not None:
        print(f"Saved train t-SNE: {train_tsne_path}")
    if val_tsne_path is not None:
        print(f"Saved val t-SNE: {val_tsne_path}")
    if test_tsne_path is not None:
        print(f"Saved test t-SNE: {test_tsne_path}")
    return result


def main() -> None:
    run_training(RUN_CONFIG)


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import Dataset
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


@dataclass(frozen=True)
class LOSOFold:
    fold_id: str
    train_subject_ids: list[int]
    val_subject_ids: list[int]
    test_subject_ids: list[int]


@dataclass(frozen=True)
class IndexSplit:
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]


@dataclass(frozen=True)
class SharedConfig:
    dataset_id: str
    datasets_dir: str
    selected_activities: list[str] | None
    window_overlap: float
    val_subjects: int
    test_subjects: int
    seed: int


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, loader: Loader, indices: Sequence[int]) -> None:
        self.loader = loader
        self.indices = [int(idx) for idx in indices]
        labels, subjects = labels_subjects_for_indices(loader, self.indices)
        self.labels = labels
        self.subjects = subjects

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:  # type: ignore
        window_index = int(self.indices[int(item)])
        x = sample_window_array(self.loader, window_index)
        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.tensor(int(self.labels[int(item)]), dtype=torch.long),
            "subject_id": torch.tensor(int(self.subjects[int(item)]), dtype=torch.long),
            "window_index": torch.tensor(window_index, dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_cfg(
    dataset_id: WHARDatasetID,
    datasets_dir: Path,
    selected_activities: Sequence[str] | None,
    window_overlap: float,
) -> Any:
    cfg = get_dataset_cfg(dataset_id, datasets_dir=str(datasets_dir))
    cfg = (
        cfg.model_copy(deep=True) if hasattr(cfg, "model_copy") else cfg.copy(deep=True)
    )
    if selected_activities is not None:
        cfg.selected_activities = list(selected_activities)
        cfg.num_of_activities = len(cfg.selected_activities)
    cfg.window_overlap = float(window_overlap)
    if hasattr(cfg, "overlap"):
        cfg.overlap = float(window_overlap)
    return cfg


def subject_index_map(
    session_df: pd.DataFrame,
    window_df: pd.DataFrame,
) -> dict[int, list[int]]:
    meta = window_df[["session_id"]].copy()
    meta["window_index"] = meta.index.astype(int)
    session_meta = (
        session_df[["session_id", "subject_id"]].drop_duplicates("session_id").copy()
    )
    merged = meta.merge(session_meta, on="session_id", how="left")
    grouped = merged.groupby("subject_id")["window_index"]
    return {
        int(subject_id): sorted(int(x) for x in indices.tolist())
        for subject_id, indices in grouped
    }


def build_or_load_loso_folds(
    session_df: pd.DataFrame,
    window_df: pd.DataFrame,
    shared_cfg: SharedConfig,
    manifest_path: Path,
) -> list[LOSOFold]:
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("shared_config", {}) == asdict(shared_cfg):
            return [LOSOFold(**row) for row in payload["folds"]]

    if int(shared_cfg.test_subjects) != 1:
        raise ValueError(
            f"LOSO requires test_subjects=1, got {shared_cfg.test_subjects}."
        )
    subject_ids = sorted(subject_index_map(session_df, window_df).keys())
    if len(subject_ids) <= int(shared_cfg.val_subjects) + 1:
        raise ValueError(
            "Not enough subjects for LOSO with validation subjects: "
            f"found={len(subject_ids)}, val_subjects={shared_cfg.val_subjects}."
        )

    folds: list[LOSOFold] = []
    for idx, test_subject_id in enumerate(subject_ids, start=1):
        remaining = [
            int(sid) for sid in subject_ids if int(sid) != int(test_subject_id)
        ]
        rng = np.random.default_rng(int(shared_cfg.seed) + 10_000 * idx)
        shuffled = rng.permutation(np.asarray(remaining, dtype=np.int64)).tolist()
        val_subjects = sorted(int(x) for x in shuffled[: int(shared_cfg.val_subjects)])
        train_subjects = sorted(
            int(x) for x in shuffled[int(shared_cfg.val_subjects) :]
        )
        folds.append(
            LOSOFold(
                fold_id=f"loso_subject_{int(test_subject_id)}",
                train_subject_ids=train_subjects,
                val_subject_ids=val_subjects,
                test_subject_ids=[int(test_subject_id)],
            )
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "shared_config": asdict(shared_cfg),
        "folds": [asdict(fold) for fold in folds],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return folds


def split_indices_for_fold(
    session_df: pd.DataFrame,
    window_df: pd.DataFrame,
    fold: LOSOFold,
) -> IndexSplit:
    subject_map = subject_index_map(session_df, window_df)

    def gather(subject_ids: Sequence[int]) -> list[int]:
        out: list[int] = []
        for sid in subject_ids:
            out.extend(subject_map[int(sid)])
        return sorted(out)

    return IndexSplit(
        train_indices=gather(fold.train_subject_ids),
        val_indices=gather(fold.val_subject_ids),
        test_indices=gather(fold.test_subject_ids),
    )


def build_loader(
    cfg: Any,
    session_df: pd.DataFrame,
    pre_pipeline: PreProcessingPipeline,
    window_df: pd.DataFrame,
    fit_indices: Sequence[int],
) -> Loader:
    post = PostProcessingPipeline(cfg, pre_pipeline, window_df, list(fit_indices))
    samples = post.run()
    return Loader(session_df, window_df, post.samples_dir, samples)


def config_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sample_window_array(loader: Loader, index: int) -> np.ndarray:
    sample = loader.get_sample(int(index))
    if not sample:
        raise ValueError(f"Empty sample for window index {index}.")
    x_np = np.asarray(sample[0])
    if x_np.ndim == 2:
        return np.array(x_np, dtype=np.float32, copy=True)
    if x_np.ndim == 3 and x_np.shape[0] == 1:
        return np.array(x_np[0], dtype=np.float32, copy=True)
    raise ValueError(
        f"Expected sample shape (window, sensors), got {tuple(x_np.shape)}."
    )


def labels_subjects_for_indices(
    loader: Loader,
    indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    subset["window_index"] = subset.index.astype(int)
    session_meta = loader.session_df[
        ["session_id", "subject_id", "activity_id"]
    ].drop_duplicates("session_id")
    merged = subset.merge(session_meta, on="session_id", how="left")
    if merged["subject_id"].isna().any() or merged["activity_id"].isna().any():
        raise ValueError("Missing subject/activity metadata.")
    merged = merged.sort_values("window_index")
    return (
        merged["activity_id"].to_numpy(dtype=np.int64),
        merged["subject_id"].to_numpy(dtype=np.int64),
    )


def indices_by_activity(
    loader: Loader,
    indices: Sequence[int],
) -> dict[int, np.ndarray]:
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    subset["window_index"] = subset.index.astype(int)
    session_meta = loader.session_df[["session_id", "activity_id"]].drop_duplicates(
        "session_id"
    )
    merged = subset.merge(session_meta, on="session_id", how="left")
    if merged["activity_id"].isna().any():
        raise ValueError("Missing activity metadata.")
    grouped = merged.groupby("activity_id")["window_index"]
    return {
        int(activity_id): np.asarray(group.tolist(), dtype=np.int64)
        for activity_id, group in grouped
    }


def infer_window_size(loader: Loader, indices: Sequence[int]) -> int:
    if not indices:
        raise ValueError("Cannot infer window size from empty indices.")
    x_np = sample_window_array(loader, int(indices[0]))
    return int(x_np.shape[0])


def prepare_inputs(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.dim() != 4:
        raise ValueError(f"Expected input with 3 or 4 dims, got {tuple(x.shape)}.")
    return x


def class_names(cfg: Any) -> list[str]:
    names = getattr(cfg, "selected_activities", None)
    if names is not None:
        return [str(x) for x in names]
    return [str(i) for i in range(int(cfg.num_of_activities))]


def build_tinierhar(
    num_channels: int,
    num_classes: int,
    window_size: int,
) -> TinierHAR:
    return TinierHAR(
        num_channels=int(num_channels),
        num_classes=int(num_classes),
        window_size=int(window_size),
        backbone_config=DEFAULT_CONFIG.backbone,
    )


def load_supcon_backbone(
    checkpoint_path: Path,
    device: str | torch.device,
) -> tuple[TinierHAR, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_meta = dict(checkpoint.get("model_meta", {}))
    backbone = build_tinierhar(
        num_channels=int(model_meta["num_channels"]),
        num_classes=int(model_meta["num_classes"]),
        window_size=int(model_meta["window_size"]),
    )
    state = checkpoint.get("backbone_state_dict", checkpoint.get("model_state_dict"))
    if state is None:
        raise KeyError(f"No backbone_state_dict/model_state_dict in {checkpoint_path}.")
    backbone.load_state_dict(state)
    backbone.to(device)
    return backbone, checkpoint


def load_ce_backbone(
    checkpoint_path: Path,
    device: str | torch.device,
) -> tuple[TinierHAR, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_meta = dict(checkpoint.get("model_meta", {}))
    required = ("num_channels", "num_classes", "window_size")
    missing = [key for key in required if key not in model_meta]
    if missing:
        raise KeyError(
            f"CE checkpoint is missing model_meta keys {missing}. "
            "Use 01_train_tinierhar_ce_loso.py checkpoints saved as "
            "best_model_with_meta.pt."
        )
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError(f"No model_state_dict in CE checkpoint {checkpoint_path}.")
    backbone = build_tinierhar(
        num_channels=int(model_meta["num_channels"]),
        num_classes=int(model_meta["num_classes"]),
        window_size=int(model_meta["window_size"]),
    )
    backbone.load_state_dict(state)
    backbone.to(device)
    return backbone, checkpoint


def build_supcon_projection_head(
    checkpoint: dict[str, Any],
    device: str | torch.device,
) -> nn.Module:
    model_meta = dict(checkpoint.get("model_meta", {}))
    feature_dim = int(model_meta["feature_dim"])
    hidden_dim = int(model_meta["projection_hidden_dim"])
    projection_dim = int(model_meta.get("projection_dim", feature_dim))
    projection_head = nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, projection_dim),
    )
    state = checkpoint.get("projection_head_state_dict")
    if state is None:
        raise KeyError("No projection_head_state_dict in SupCon checkpoint.")
    projection_head.load_state_dict(state)
    projection_head.to(device)
    projection_head.eval()
    return projection_head


@torch.no_grad()
def extract_embeddings(
    backbone: TinierHAR,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    backbone.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    subjects: list[torch.Tensor] = []
    for batch in dataloader:
        x = prepare_inputs(batch["x"]).to(device).float()
        emb = backbone.encode(x)
        if normalize:
            emb = F.normalize(emb, p=2, dim=1)
        embeddings.append(emb.cpu())
        labels.append(batch["y"].long().view(-1).cpu())
        subjects.append(batch["subject_id"].long().view(-1).cpu())
    return torch.cat(embeddings), torch.cat(labels), torch.cat(subjects)


@torch.no_grad()
def extract_supcon_embeddings(
    backbone: TinierHAR,
    projection_head: nn.Module | None,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    backbone.eval()
    if projection_head is not None:
        projection_head.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    subjects: list[torch.Tensor] = []
    for batch in dataloader:
        x = prepare_inputs(batch["x"]).to(device).float()
        emb = backbone.encode(x)
        if projection_head is not None:
            emb = projection_head(emb)
        if normalize:
            emb = F.normalize(emb, p=2, dim=1)
        embeddings.append(emb.cpu())
        labels.append(batch["y"].long().view(-1).cpu())
        subjects.append(batch["subject_id"].long().view(-1).cpu())
    return torch.cat(embeddings), torch.cat(labels), torch.cat(subjects)


def make_class_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    normalize: bool = True,
) -> torch.Tensor:
    proto_list: list[torch.Tensor] = []
    for cls in range(int(num_classes)):
        mask = labels == int(cls)
        if not bool(mask.any()):
            raise ValueError(f"Cannot build prototype for missing class {cls}.")
        proto = embeddings[mask].mean(dim=0)
        proto_list.append(proto)
    prototypes = torch.stack(proto_list, dim=0)
    if normalize:
        prototypes = F.normalize(prototypes, p=2, dim=1)
    return prototypes


def cosine_logits(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    emb = F.normalize(embeddings, p=2, dim=1)
    proto = F.normalize(prototypes, p=2, dim=1)
    return torch.matmul(emb, proto.T) / float(temperature)


def euclidean_logits(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    dists = torch.cdist(embeddings, prototypes, p=2)
    return -dists.pow(2) / float(temperature)


def resolve_distance_metric(distance_metric: str, backbone_source: str) -> str:
    if distance_metric == "auto":
        return "cosine" if backbone_source == "supcon" else "euclidean"
    if distance_metric not in {"cosine", "euclidean"}:
        raise ValueError("distance_metric must be 'auto', 'cosine', or 'euclidean'.")
    return distance_metric


def prototype_logits(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
    distance_metric: str,
) -> torch.Tensor:
    if distance_metric == "cosine":
        return cosine_logits(embeddings, prototypes, temperature)
    if distance_metric == "euclidean":
        return euclidean_logits(embeddings, prototypes, temperature)
    raise ValueError("distance_metric must be 'cosine' or 'euclidean'.")


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    true_np = np.asarray(y_true, dtype=np.int64)
    pred_np = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(int(num_classes)))
    return {
        "accuracy": float(accuracy_score(true_np, pred_np)),
        "macro_f1": float(
            f1_score(true_np, pred_np, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                true_np, pred_np, labels=labels, average="weighted", zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(true_np, pred_np, labels=labels).tolist(),
    }


def mean_std_ci(values: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(max(1, arr.size)))
    return mean, std, ci95


def save_confusion_matrix_plot(
    matrix: Sequence[Sequence[int]] | np.ndarray,
    labels: Sequence[str],
    out_path: Path,
    title: str,
    normalize: bool = True,
) -> None:
    cm = np.asarray(matrix, dtype=np.float64)
    if normalize:
        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        cm = cm / row_sums
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=list(labels),
        yticklabels=list(labels),
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

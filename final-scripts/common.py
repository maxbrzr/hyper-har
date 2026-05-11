from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from whar_datasets import (
    Loader,
    PostProcessingPipeline,
    PreProcessingPipeline,
    WHARDatasetID,
    get_dataset_cfg,
)

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRAIN_MIN_K_PER_CLASS = 1
DEFAULT_TRAIN_MAX_K_PER_CLASS = 8
DEFAULT_EVAL_K_PER_CLASS = 1


@dataclass(frozen=True)
class SubjectFold:
    fold_id: str
    base_train_subject_ids: list[int]
    meta_train_subject_ids: list[int]
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
    subjects_per_group: int
    seed: int


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
    cfg = cfg.model_copy(deep=True) if hasattr(cfg, "model_copy") else cfg.copy(deep=True)
    if selected_activities is not None:
        cfg.selected_activities = list(selected_activities)
        cfg.num_of_activities = len(cfg.selected_activities)
    cfg.window_overlap = float(window_overlap)
    if hasattr(cfg, "overlap"):
        cfg.overlap = float(window_overlap)
    return cfg


def _subject_index_map(
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
) -> list[SubjectFold]:
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_cfg = payload.get("shared_config", {})
        if existing_cfg == asdict(shared_cfg):
            return [SubjectFold(**row) for row in payload["folds"]]

    subject_to_indices = _subject_index_map(session_df, window_df)
    subject_ids = sorted(subject_to_indices.keys())
    n_groups = 4
    expected_total = int(shared_cfg.subjects_per_group) * n_groups
    if len(subject_ids) != expected_total:
        raise ValueError(
            f"Expected exactly {expected_total} subjects for 4-fold grouped CV, "
            f"but found {len(subject_ids)} subjects: {subject_ids}"
        )

    groups: dict[str, list[int]] = {}
    labels = ["A", "B", "C", "D"]
    for i, label in enumerate(labels):
        start = i * int(shared_cfg.subjects_per_group)
        end = (i + 1) * int(shared_cfg.subjects_per_group)
        groups[label] = [int(x) for x in subject_ids[start:end]]

    # Rotation protocol:
    # Fold 1: Base(A), Meta(B), Val(C), Test(D)
    # Fold 2: Base(D), Meta(A), Val(B), Test(C)
    # Fold 3: Base(C), Meta(D), Val(A), Test(B)
    # Fold 4: Base(B), Meta(C), Val(D), Test(A)
    rotations = [
        ("A", "B", "C", "D"),
        ("D", "A", "B", "C"),
        ("C", "D", "A", "B"),
        ("B", "C", "D", "A"),
    ]
    folds: list[SubjectFold] = []
    for idx, (base_g, meta_g, val_g, test_g) in enumerate(rotations, start=1):
        folds.append(
            SubjectFold(
                fold_id=f"fold_{idx}",
                base_train_subject_ids=sorted(groups[base_g]),
                meta_train_subject_ids=sorted(groups[meta_g]),
                val_subject_ids=sorted(groups[val_g]),
                test_subject_ids=sorted(groups[test_g]),
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
    fold: Any,
) -> IndexSplit:
    subject_map = _subject_index_map(session_df, window_df)

    def gather(subject_ids: Sequence[int]) -> list[int]:
        out: list[int] = []
        for sid in subject_ids:
            out.extend(subject_map[int(sid)])
        return sorted(out)

    train_subject_ids = getattr(fold, "train_subject_ids", None)
    if train_subject_ids is None:
        train_subject_ids = getattr(fold, "meta_train_subject_ids", None)
    if train_subject_ids is None:
        raise ValueError("Fold object must define train_subject_ids or meta_train_subject_ids.")

    val_subject_ids = getattr(fold, "val_subject_ids", None)
    test_subject_ids = getattr(fold, "test_subject_ids", None)
    if val_subject_ids is None or test_subject_ids is None:
        raise ValueError("Fold object must define val_subject_ids and test_subject_ids.")

    train_indices = gather(train_subject_ids)
    val_indices = gather(val_subject_ids)
    test_indices = gather(test_subject_ids)
    return IndexSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )


def subject_ids_for_indices(
    loader: Loader, indices: Sequence[int]
) -> list[int]:
    if not indices:
        return []
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    session_meta = (
        loader.session_df[["session_id", "subject_id"]]
        .drop_duplicates("session_id")
        .set_index("session_id")
    )
    merged = subset.join(session_meta, on="session_id", how="left")
    return sorted(set(int(x) for x in merged["subject_id"].dropna().tolist()))


def build_loader(
    cfg: Any,
    session_df: pd.DataFrame,
    pre_pipeline: PreProcessingPipeline,
    window_df: pd.DataFrame,
    train_indices: Sequence[int],
) -> Loader:
    post = PostProcessingPipeline(cfg, pre_pipeline, window_df, list(train_indices))
    samples = post.run()
    return Loader(session_df, window_df, post.samples_dir, samples)


def config_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def k_choices_from_range(min_k_per_class: int, max_k_per_class: int) -> tuple[int, ...]:
    min_k = int(min_k_per_class)
    max_k = int(max_k_per_class)
    if min_k < 1:
        raise ValueError(f"min_k_per_class must be >=1, got {min_k}")
    if max_k < min_k:
        raise ValueError(
            f"max_k_per_class must be >= min_k_per_class, got {max_k} < {min_k}"
        )
    return tuple(range(min_k, max_k + 1))

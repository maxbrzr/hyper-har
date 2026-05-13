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

DEFAULT_TRAIN_MIN_K_PER_CLASS = 1  # 1
DEFAULT_TRAIN_MAX_K_PER_CLASS = 32  # 16


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
    base_train_subjects: int = 14
    meta_train_subjects: int = 6
    val_subjects: int = 3
    test_subjects: int = 1


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
    base_n = int(shared_cfg.base_train_subjects)
    meta_n = int(shared_cfg.meta_train_subjects)
    val_n = int(shared_cfg.val_subjects)
    test_n = int(shared_cfg.test_subjects)
    expected_total = base_n + meta_n + val_n + test_n
    if test_n != 1:
        raise ValueError(f"True LOSO requires test_subjects=1, got {test_n}.")
    if len(subject_ids) != expected_total:
        raise ValueError(
            f"Expected exactly {expected_total} subjects for LOSO "
            f"({base_n}-{meta_n}-{val_n}-{test_n}), "
            f"but found {len(subject_ids)} subjects: {subject_ids}"
        )

    folds: list[SubjectFold] = []
    for idx, test_subject_id in enumerate(subject_ids, start=1):
        remaining = [
            int(sid) for sid in subject_ids if int(sid) != int(test_subject_id)
        ]
        rng = np.random.default_rng(int(shared_cfg.seed) + 10_000 * idx)
        shuffled = rng.permutation(np.asarray(remaining, dtype=np.int64)).tolist()
        base_subjects = sorted(int(x) for x in shuffled[:base_n])
        meta_subjects = sorted(int(x) for x in shuffled[base_n : base_n + meta_n])
        val_subjects = sorted(
            int(x) for x in shuffled[base_n + meta_n : base_n + meta_n + val_n]
        )
        folds.append(
            SubjectFold(
                fold_id=f"loso_subject_{int(test_subject_id)}",
                base_train_subject_ids=base_subjects,
                meta_train_subject_ids=meta_subjects,
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
        raise ValueError(
            "Fold object must define train_subject_ids or meta_train_subject_ids."
        )

    val_subject_ids = getattr(fold, "val_subject_ids", None)
    test_subject_ids = getattr(fold, "test_subject_ids", None)
    if val_subject_ids is None or test_subject_ids is None:
        raise ValueError(
            "Fold object must define val_subject_ids and test_subject_ids."
        )

    train_indices = gather(train_subject_ids)
    val_indices = gather(val_subject_ids)
    test_indices = gather(test_subject_ids)
    return IndexSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )


def subject_ids_for_indices(loader: Loader, indices: Sequence[int]) -> list[int]:
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

from __future__ import annotations

import hashlib
import json
import random
import sys
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
        raise ValueError(f"LOSO requires test_subjects=1, got {shared_cfg.test_subjects}.")
    subject_ids = sorted(subject_index_map(session_df, window_df).keys())
    if len(subject_ids) <= int(shared_cfg.val_subjects) + 1:
        raise ValueError(
            "Not enough subjects for LOSO with validation subjects: "
            f"found={len(subject_ids)}, val={shared_cfg.val_subjects}."
        )

    folds: list[LOSOFold] = []
    for idx, test_subject_id in enumerate(subject_ids, start=1):
        remaining = [int(sid) for sid in subject_ids if int(sid) != int(test_subject_id)]
        rng = np.random.default_rng(int(shared_cfg.seed) + 10_000 * idx)
        shuffled = rng.permutation(np.asarray(remaining, dtype=np.int64)).tolist()
        val_subjects = sorted(int(x) for x in shuffled[: int(shared_cfg.val_subjects)])
        train_subjects = sorted(int(x) for x in shuffled[int(shared_cfg.val_subjects) :])
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


def k_choices_from_range(min_k_per_class: int, max_k_per_class: int) -> tuple[int, ...]:
    min_k = int(min_k_per_class)
    max_k = int(max_k_per_class)
    if min_k < 1:
        raise ValueError(f"min_k_per_class must be >= 1, got {min_k}.")
    if max_k < min_k:
        raise ValueError(
            f"max_k_per_class must be >= min_k_per_class, got {max_k} < {min_k}."
        )
    return tuple(range(min_k, max_k + 1))


def build_subject_activity_index(
    loader: Loader,
    indices: Sequence[int],
) -> tuple[dict[int, dict[int, np.ndarray]], list[int]]:
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    subset["window_index"] = subset.index.astype(int)
    session_meta = loader.session_df[
        ["session_id", "subject_id", "activity_id"]
    ].drop_duplicates("session_id")
    merged = subset.merge(session_meta, on="session_id", how="left")
    if merged["subject_id"].isna().any() or merged["activity_id"].isna().any():
        raise ValueError("Missing subject/activity metadata.")
    activity_ids = sorted(int(x) for x in merged["activity_id"].unique().tolist())
    grouped = merged.groupby(["subject_id", "activity_id"])["window_index"]
    nested: dict[int, dict[int, np.ndarray]] = {}
    for (sid, aid), vals in grouped:
        nested.setdefault(int(sid), {})[int(aid)] = vals.to_numpy(dtype=np.int64)
    return nested, activity_ids


def sample_window_array(loader: Loader, index: int) -> np.ndarray:
    sample = loader.get_sample(int(index))
    if not sample:
        raise ValueError(f"Empty sample for window index {index}.")
    x_np = np.asarray(sample[0])
    if x_np.ndim == 2:
        return x_np
    if x_np.ndim == 3 and x_np.shape[0] == 1:
        return x_np[0]
    raise ValueError(f"Expected sample shape (window, sensors), got {tuple(x_np.shape)}.")


def infer_window_size(loader: Loader, indices: Sequence[int]) -> int:
    if not indices:
        raise ValueError("Cannot infer window size from empty indices.")
    x_np = sample_window_array(loader, int(indices[0]))
    return int(x_np.shape[0])

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from whar_datasets.config.config import WHARConfig
from whar_datasets.splitting.split import Split


@dataclass(frozen=True)
class StrictAdaptationFold:
    identifier: str
    test_subject_id: int
    base_pretrain_subject_ids: List[int]
    pretrain_train_subject_ids: List[int]
    pretrain_val_subject_ids: List[int]
    pretrain_val_split_level: str
    meta_train_subject_ids: List[int]
    meta_val_subject_ids: List[int]
    pretrain_split: Split
    meta_split: Split


class StrictAdaptationSplitter:
    """Subject-disjoint split for studying adaptation to base-unseen subjects.

    For each outer held-out test subject, the remaining subjects are partitioned
    into three disjoint roles:

    - base-pretrain subjects: train the frozen base model
    - meta-train subjects: train the adapter/hypernetwork rule
    - meta-val subjects: select checkpoints/hyperparameters

    This differs from ``MetaLOSOSplitter`` because meta-train subjects are not
    used to pretrain the base model.

    By default, the base model's own train/validation split is window-level
    within the base-pretrain subject pool. This keeps the important adaptation
    boundary subject-disjoint while giving the base model access to all
    base-pretrain subjects during checkpoint selection.
    """

    def __init__(
        self,
        cfg: WHARConfig,
        subject_ids: List[int] | None = None,
        meta_train_percentage: float = 0.25,
        meta_val_percentage: float = 0.15,
        pretrain_val_percentage: float | None = None,
        pretrain_val_split_level: str = "window",
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.subject_ids = subject_ids
        self.meta_train_percentage = meta_train_percentage
        self.meta_val_percentage = meta_val_percentage
        self.pretrain_val_percentage = (
            cfg.val_percentage if pretrain_val_percentage is None else pretrain_val_percentage
        )
        if pretrain_val_split_level not in {"window", "subject"}:
            raise ValueError(
                "pretrain_val_split_level must be either 'window' or 'subject'."
            )
        self.pretrain_val_split_level = pretrain_val_split_level
        cfg_seed = getattr(cfg, "seed", None)
        self.seed = int(seed if seed is not None else (cfg_seed if cfg_seed is not None else 0))
        self._cached_folds: Optional[List[StrictAdaptationFold]] = None

    @staticmethod
    def _indices_for_subjects(
        session_df: pd.DataFrame, window_df: pd.DataFrame, subject_ids: List[int]
    ) -> List[int]:
        if not subject_ids:
            return []
        sessions = session_df[session_df["subject_id"].isin(subject_ids)][
            "session_id"
        ].tolist()
        return window_df[window_df["session_id"].isin(sessions)].index.tolist()

    @staticmethod
    def _subject_level_partition(
        rng: np.random.Generator,
        subject_ids: List[int],
        val_percentage: float,
    ) -> tuple[List[int], List[int]]:
        if not subject_ids:
            return [], []
        if len(subject_ids) == 1:
            return subject_ids.copy(), []
        n_val = int(round(len(subject_ids) * val_percentage))
        n_val = max(1, min(n_val, len(subject_ids) - 1))
        shuffled = [int(x) for x in rng.permutation(subject_ids).tolist()]
        return shuffled[n_val:], shuffled[:n_val]

    @staticmethod
    def _subject_ids_for_indices(
        session_df: pd.DataFrame, window_df: pd.DataFrame, indices: List[int]
    ) -> List[int]:
        if not indices:
            return []
        subset = window_df.loc[list(indices), ["session_id"]].copy()
        session_meta = (
            session_df[["session_id", "subject_id"]]
            .drop_duplicates("session_id")
            .set_index("session_id")
        )
        merged = subset.join(session_meta, on="session_id", how="left")
        if merged["subject_id"].isna().any():
            raise ValueError("Missing subject_id while inferring split subjects.")
        return sorted(set(int(x) for x in merged["subject_id"].tolist()))

    @staticmethod
    def _window_level_partition(
        rng: np.random.Generator,
        session_df: pd.DataFrame,
        window_df: pd.DataFrame,
        indices: List[int],
        val_percentage: float,
    ) -> tuple[List[int], List[int]]:
        if not indices:
            return [], []
        if len(indices) == 1:
            return indices.copy(), []

        subset = window_df.loc[list(indices), ["session_id"]].copy()
        subset["window_index"] = subset.index.astype(int)
        session_meta = session_df[
            ["session_id", "activity_id"]
        ].drop_duplicates("session_id")
        merged = subset.merge(session_meta, on="session_id", how="left")

        if merged["activity_id"].isna().any():
            shuffled = [int(x) for x in rng.permutation(indices).tolist()]
            n_val = int(round(len(shuffled) * val_percentage))
            n_val = max(1, min(n_val, len(shuffled) - 1))
            return shuffled[n_val:], shuffled[:n_val]

        train_indices: List[int] = []
        val_indices: List[int] = []
        for _, group in merged.groupby("activity_id"):
            group_indices = [int(x) for x in group["window_index"].tolist()]
            if len(group_indices) <= 1:
                train_indices.extend(group_indices)
                continue
            shuffled = [int(x) for x in rng.permutation(group_indices).tolist()]
            n_val = int(round(len(shuffled) * val_percentage))
            n_val = max(1, min(n_val, len(shuffled) - 1))
            val_indices.extend(shuffled[:n_val])
            train_indices.extend(shuffled[n_val:])

        if not val_indices and len(train_indices) > 1:
            shuffled_train = [int(x) for x in rng.permutation(train_indices).tolist()]
            val_indices = shuffled_train[:1]
            train_indices = shuffled_train[1:]
        if not train_indices:
            raise ValueError("Could not create non-empty pretrain train indices.")

        return sorted(train_indices), sorted(val_indices)

    def _role_partition(
        self,
        rng: np.random.Generator,
        remaining_subjects: List[int],
    ) -> tuple[List[int], List[int], List[int]]:
        if len(remaining_subjects) < 4:
            raise ValueError(
                "StrictAdaptationSplitter needs at least 5 total subjects: "
                "1 test, at least 2 base-pretrain, 1 meta-train, 1 meta-val."
            )

        shuffled = [int(x) for x in rng.permutation(remaining_subjects).tolist()]
        n_remaining = len(shuffled)
        n_meta_val = max(1, int(round(n_remaining * self.meta_val_percentage)))
        n_meta_train = max(1, int(round(n_remaining * self.meta_train_percentage)))
        if n_meta_val + n_meta_train > n_remaining - 2:
            overflow = (n_meta_val + n_meta_train) - (n_remaining - 2)
            n_meta_train = max(1, n_meta_train - overflow)
        if n_meta_val + n_meta_train > n_remaining - 2:
            n_meta_val = max(1, n_remaining - 2 - n_meta_train)

        meta_val_subjects = sorted(shuffled[:n_meta_val])
        meta_train_subjects = sorted(shuffled[n_meta_val : n_meta_val + n_meta_train])
        base_pretrain_subjects = sorted(shuffled[n_meta_val + n_meta_train :])
        if not base_pretrain_subjects or not meta_train_subjects or not meta_val_subjects:
            raise ValueError("Could not create non-empty strict adaptation roles.")
        return base_pretrain_subjects, meta_train_subjects, meta_val_subjects

    def get_folds(
        self,
        session_df: pd.DataFrame,
        window_df: pd.DataFrame,
    ) -> List[StrictAdaptationFold]:
        if self._cached_folds is not None:
            return self._cached_folds

        subject_ids = sorted(
            int(x)
            for x in (self.subject_ids or session_df["subject_id"].unique().tolist())
        )
        folds: List[StrictAdaptationFold] = []
        for test_subject_id in subject_ids:
            remaining_subjects = [s for s in subject_ids if s != test_subject_id]
            rng = np.random.default_rng(self.seed + int(test_subject_id))
            base_subjects, meta_train_subjects, meta_val_subjects = self._role_partition(
                rng, remaining_subjects
            )
            test_indices = self._indices_for_subjects(
                session_df, window_df, [test_subject_id]
            )
            base_indices = self._indices_for_subjects(
                session_df, window_df, base_subjects
            )
            if self.pretrain_val_split_level == "subject":
                pretrain_train_subjects, pretrain_val_subjects = (
                    self._subject_level_partition(
                        rng, base_subjects, self.pretrain_val_percentage
                    )
                )
                if not pretrain_val_subjects:
                    pretrain_val_subjects = pretrain_train_subjects[-1:]
                    pretrain_train_subjects = pretrain_train_subjects[:-1]
                if not pretrain_train_subjects:
                    raise ValueError(
                        f"Not enough base-pretrain subjects for test subject {test_subject_id}."
                    )
                pretrain_train_indices = self._indices_for_subjects(
                    session_df, window_df, pretrain_train_subjects
                )
                pretrain_val_indices = self._indices_for_subjects(
                    session_df, window_df, pretrain_val_subjects
                )
            else:
                pretrain_train_indices, pretrain_val_indices = (
                    self._window_level_partition(
                        rng,
                        session_df,
                        window_df,
                        base_indices,
                        self.pretrain_val_percentage,
                    )
                )
                pretrain_train_subjects = self._subject_ids_for_indices(
                    session_df, window_df, pretrain_train_indices
                )
                pretrain_val_subjects = self._subject_ids_for_indices(
                    session_df, window_df, pretrain_val_indices
                )

            meta_train_indices = self._indices_for_subjects(
                session_df, window_df, meta_train_subjects
            )
            meta_val_indices = self._indices_for_subjects(
                session_df, window_df, meta_val_subjects
            )

            pretrain_split = Split(
                identifier=f"subject_{test_subject_id}",
                train_indices=pretrain_train_indices,
                val_indices=pretrain_val_indices,
                test_indices=test_indices,
            )
            meta_split = Split(
                identifier=f"subject_{test_subject_id}",
                train_indices=meta_train_indices,
                val_indices=meta_val_indices,
                test_indices=test_indices,
            )
            folds.append(
                StrictAdaptationFold(
                    identifier=f"subject_{test_subject_id}",
                    test_subject_id=int(test_subject_id),
                    base_pretrain_subject_ids=base_subjects,
                    pretrain_train_subject_ids=sorted(pretrain_train_subjects),
                    pretrain_val_subject_ids=sorted(pretrain_val_subjects),
                    pretrain_val_split_level=self.pretrain_val_split_level,
                    meta_train_subject_ids=meta_train_subjects,
                    meta_val_subject_ids=meta_val_subjects,
                    pretrain_split=pretrain_split,
                    meta_split=meta_split,
                )
            )

        self._cached_folds = folds
        return folds

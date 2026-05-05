from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from whar_datasets.config.config import WHARConfig
from whar_datasets.splitting.split import Split
from whar_datasets.splitting.splitter import Splitter


@dataclass(frozen=True)
class MetaLOSOFold:
    identifier: str
    test_subject_id: int
    meta_train_subject_ids: List[int]
    meta_val_subject_ids: List[int]
    pretrain_train_subject_ids: List[int]
    pretrain_val_subject_ids: List[int]
    meta_split: Split
    pretrain_split: Split


class MetaLOSOSplitter(Splitter):
    """LOSO with subject-wise validation from the remaining training subjects.

    For each held-out test subject, validation subjects are sampled from the
    non-test subjects based on ``cfg.val_percentage`` at the subject level.

    Two subject-wise splits are produced per fold:
    - meta split: ``meta_train`` vs ``meta_val`` vs held-out ``test`` subject
    - pretrain split: ``pretrain_train`` vs ``pretrain_val`` vs same ``test``,
      where pretrain train/val are both subsets of ``meta_train`` only.
    """

    def __init__(
        self,
        cfg: WHARConfig,
        subject_ids: List[int] | None = None,
        meta_val_percentage: float | None = None,
        pretrain_val_percentage: float | None = None,
    ):
        super().__init__(cfg)
        self.subject_ids = subject_ids
        self.meta_val_percentage = (
            cfg.val_percentage if meta_val_percentage is None else meta_val_percentage
        )
        self.pretrain_val_percentage = (
            cfg.val_percentage
            if pretrain_val_percentage is None
            else pretrain_val_percentage
        )

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
        rng,
        subject_ids: List[int],
        val_percentage: float,
    ) -> tuple[List[int], List[int]]:
        if not subject_ids:
            return [], []
        if len(subject_ids) == 1:
            return subject_ids.copy(), []

        n_val = int(len(subject_ids) * val_percentage)
        n_val = max(1, min(n_val, len(subject_ids) - 1))
        shuffled = rng.permutation(subject_ids).tolist()
        val_subjects = [int(x) for x in shuffled[:n_val]]
        train_subjects = [int(x) for x in shuffled[n_val:]]
        return train_subjects, val_subjects

    def get_folds(
        self,
        session_df: pd.DataFrame,
        window_df: pd.DataFrame,
    ) -> List[MetaLOSOFold]:
        subject_ids = [int(x) for x in (self.subject_ids or session_df["subject_id"].unique().tolist())]
        folds: List[MetaLOSOFold] = []

        for test_subject_id in subject_ids:
            remaining_subjects = [s for s in subject_ids if s != test_subject_id]
            if not remaining_subjects:
                raise ValueError(
                    "MetaLOSOSplitter requires at least 2 subjects to create folds."
                )

            meta_train_subjects, meta_val_subjects = self._subject_level_partition(
                self.rng, remaining_subjects, self.meta_val_percentage
            )
            pretrain_train_subjects, pretrain_val_subjects = self._subject_level_partition(
                self.rng, meta_train_subjects, self.pretrain_val_percentage
            )

            test_indices = self._indices_for_subjects(
                session_df, window_df, [test_subject_id]
            )
            meta_train_indices = self._indices_for_subjects(
                session_df, window_df, meta_train_subjects
            )
            meta_val_indices = self._indices_for_subjects(
                session_df, window_df, meta_val_subjects
            )
            pretrain_train_indices = self._indices_for_subjects(
                session_df, window_df, pretrain_train_subjects
            )
            pretrain_val_indices = self._indices_for_subjects(
                session_df, window_df, pretrain_val_subjects
            )

            meta_split = Split(
                identifier=f"subject_{test_subject_id}",
                train_indices=meta_train_indices,
                val_indices=meta_val_indices,
                test_indices=test_indices,
            )
            pretrain_split = Split(
                identifier=f"subject_{test_subject_id}",
                train_indices=pretrain_train_indices,
                val_indices=pretrain_val_indices,
                test_indices=test_indices,
            )

            assert not self._check_indices_overlap(
                meta_split.train_indices, meta_split.val_indices, meta_split.test_indices
            ), "Overlap detected in meta split indices!"
            assert not self._check_indices_overlap(
                pretrain_split.train_indices,
                pretrain_split.val_indices,
                pretrain_split.test_indices,
            ), "Overlap detected in pretrain split indices!"

            folds.append(
                MetaLOSOFold(
                    identifier=f"subject_{test_subject_id}",
                    test_subject_id=int(test_subject_id),
                    meta_train_subject_ids=meta_train_subjects,
                    meta_val_subject_ids=meta_val_subjects,
                    pretrain_train_subject_ids=pretrain_train_subjects,
                    pretrain_val_subject_ids=pretrain_val_subjects,
                    meta_split=meta_split,
                    pretrain_split=pretrain_split,
                )
            )

        return folds

    def get_splits(
        self,
        session_df: pd.DataFrame,
        window_df: pd.DataFrame,
    ) -> List[Split]:
        """Return meta-training splits (meta-train/meta-val/test)."""
        return [fold.meta_split for fold in self.get_folds(session_df, window_df)]

    def get_pretrain_splits(
        self,
        session_df: pd.DataFrame,
        window_df: pd.DataFrame,
    ) -> List[Split]:
        """Return backbone pretraining splits from meta-train subjects only."""
        return [fold.pretrain_split for fold in self.get_folds(session_df, window_df)]

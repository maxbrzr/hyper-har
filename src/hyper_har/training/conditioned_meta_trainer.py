from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from whar_datasets import Loader


@dataclass
class ConditionedMetaTrainerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_subjects: int = 4
    support_per_class: int = 4
    support_per_class_choices: Sequence[int] | None = None
    query_per_class: int = 8
    seed: int | None = None
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


@dataclass
class ConditionedMetaTrainerState:
    history: Dict[str, list[float]] = field(
        default_factory=lambda: {
            "meta_train_loss": [],
            "meta_train_macro_f1": [],
        }
    )
    steps: int = 0


class SubjectConditionedMetaTrainer:
    """Episodic trainer for subject-conditioned activation-space adaptation."""

    def __init__(
        self,
        conditioned_model: nn.Module,
        set_encoder: nn.Module,
        baseline_model: nn.Module,
        loader: Loader,
        num_classes: int,
        config: ConditionedMetaTrainerConfig,
        optimizer: torch.optim.Optimizer | None = None,
        class_weights: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
        activity_ids: Sequence[int] | None = None,
        freeze_set_encoder: bool = True,
    ) -> None:
        self.conditioned_model = conditioned_model
        self.set_encoder = set_encoder
        self.baseline_model = baseline_model
        self.loader = loader
        self.num_classes = num_classes
        self.config = config
        self.device = torch.device(config.device)
        self.rng = np.random.default_rng(config.seed)
        self.freeze_set_encoder = bool(freeze_set_encoder)

        self.conditioned_model.to(self.device)
        self.set_encoder.to(self.device)
        self.baseline_model.to(self.device)

        for param in self.baseline_model.parameters():
            param.requires_grad = False
        self.baseline_model.eval()

        if self.freeze_set_encoder:
            for param in self.set_encoder.parameters():
                param.requires_grad = False
            self.set_encoder.eval()

        self.indices = list(indices) if indices is not None else self.loader.filter_indices()
        self.activity_ids = (
            sorted(set(int(x) for x in activity_ids))
            if activity_ids is not None
            else sorted(
                set(
                    int(x)
                    for x in self.loader.session_df["activity_id"].unique().tolist()
                    if 0 <= int(x) < num_classes
                )
            )
        )
        if not self.activity_ids:
            raise ValueError("No activity ids available for episodic sampling.")

        self._indices_by_subject_activity = self._build_subject_activity_index()
        self.eligible_subject_ids = self._build_eligible_subject_ids()
        if len(self.eligible_subject_ids) < self.config.batch_subjects:
            raise ValueError(
                "Not enough eligible subjects for an episode: "
                f"need {self.config.batch_subjects}, found {len(self.eligible_subject_ids)}."
            )

        trainable_params = [
            p for p in self.conditioned_model.parameters() if p.requires_grad
        ]
        if not trainable_params:
            raise ValueError("Conditioned model has no trainable parameters.")
        self.optimizer = optimizer or torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.class_weights = (
            class_weights.to(self.device) if class_weights is not None else None
        )
        self.state = ConditionedMetaTrainerState()

    def _build_subject_activity_index(self) -> dict[tuple[int, int], np.ndarray]:
        subset = self.loader.window_df.loc[self.indices, ["session_id"]].copy()
        subset["window_index"] = subset.index.astype(int)

        session_meta = (
            self.loader.session_df[["session_id", "subject_id", "activity_id"]]
            .drop_duplicates("session_id")
            .copy()
        )
        merged = subset.merge(session_meta, on="session_id", how="left")
        if merged["subject_id"].isna().any() or merged["activity_id"].isna().any():
            raise ValueError("Missing subject/activity metadata while building episodes.")

        grouped = merged.groupby(["subject_id", "activity_id"])["window_index"]
        return {
            (int(subject_id), int(activity_id)): np.asarray(group.tolist(), dtype=np.int64)
            for (subject_id, activity_id), group in grouped
        }

    def _build_eligible_subject_ids(self) -> list[int]:
        max_support_per_class = self.config.support_per_class
        if self.config.support_per_class_choices is not None:
            cleaned_choices = [
                int(k) for k in self.config.support_per_class_choices if int(k) > 0
            ]
            if not cleaned_choices:
                raise ValueError(
                    "support_per_class_choices must contain positive integers."
                )
            max_support_per_class = max(cleaned_choices)
        support_plus_query = max_support_per_class + self.config.query_per_class
        subject_ids = sorted(
            {subject_id for subject_id, _ in self._indices_by_subject_activity.keys()}
        )
        eligible: list[int] = []
        for subject_id in subject_ids:
            enough_all_classes = all(
                len(
                    self._indices_by_subject_activity.get(
                        (subject_id, activity_id), np.empty((0,), dtype=np.int64)
                    )
                )
                >= support_plus_query
                for activity_id in self.activity_ids
            )
            if enough_all_classes:
                eligible.append(subject_id)
        return eligible

    def _sample_support_per_class(self) -> int:
        choices = self.config.support_per_class_choices
        if choices is None:
            return int(self.config.support_per_class)
        cleaned = [int(k) for k in choices if int(k) > 0]
        if not cleaned:
            raise ValueError("support_per_class_choices must contain positive integers.")
        return int(self.rng.choice(np.asarray(cleaned, dtype=np.int64)))

    def _sample_window_array(self, index: int) -> np.ndarray:
        sample = self.loader.get_sample(index)
        if not sample:
            raise ValueError(f"Empty sample for window index {index}.")
        x_np = np.asarray(sample[0])
        if x_np.ndim == 2:
            return x_np
        if x_np.ndim == 3 and x_np.shape[0] == 1:
            return x_np[0]
        raise ValueError(
            f"Expected sample with shape (window, sensors), got {tuple(x_np.shape)}."
        )

    def _sample_episode(
        self,
        support_per_class: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        bsz = self.config.batch_subjects
        k_shot = (
            int(support_per_class)
            if support_per_class is not None
            else self._sample_support_per_class()
        )
        q_query = self.config.query_per_class

        sampled_subjects = self.rng.choice(
            np.asarray(self.eligible_subject_ids, dtype=np.int64),
            size=bsz,
            replace=False,
        ).tolist()

        support_x_all: list[np.ndarray] = []
        support_y_all: list[np.ndarray] = []
        query_x_all: list[np.ndarray] = []
        query_y_all: list[np.ndarray] = []

        for subject_id in sampled_subjects:
            subject_support_x: list[np.ndarray] = []
            subject_support_y: list[int] = []
            subject_query_x: list[np.ndarray] = []
            subject_query_y: list[int] = []

            for activity_id in self.activity_ids:
                candidate = self._indices_by_subject_activity[(subject_id, activity_id)]
                picked = self.rng.choice(candidate, size=k_shot + q_query, replace=False)
                support_inds = picked[:k_shot]
                query_inds = picked[k_shot:]

                for idx in support_inds.tolist():
                    subject_support_x.append(self._sample_window_array(int(idx)))
                    subject_support_y.append(activity_id)
                for idx in query_inds.tolist():
                    subject_query_x.append(self._sample_window_array(int(idx)))
                    subject_query_y.append(activity_id)

            support_perm = self.rng.permutation(len(subject_support_x))
            query_perm = self.rng.permutation(len(subject_query_x))

            support_x_all.append(
                np.stack([subject_support_x[i] for i in support_perm], axis=0)
            )
            support_y_all.append(
                np.asarray([subject_support_y[i] for i in support_perm], dtype=np.int64)
            )
            query_x_all.append(
                np.stack([subject_query_x[i] for i in query_perm], axis=0)
            )
            query_y_all.append(
                np.asarray([subject_query_y[i] for i in query_perm], dtype=np.int64)
            )

        x_support = torch.from_numpy(np.stack(support_x_all, axis=0)).float().unsqueeze(2)
        y_support = torch.from_numpy(np.stack(support_y_all, axis=0)).long()
        x_query = torch.from_numpy(np.stack(query_x_all, axis=0)).float().unsqueeze(2)
        y_query = torch.from_numpy(np.stack(query_y_all, axis=0)).long()

        return x_support, y_support, x_query, y_query, [int(s) for s in sampled_subjects]

    def encode_subject(
        self, x_support: torch.Tensor, y_support: torch.Tensor
    ) -> torch.Tensor:
        if self.freeze_set_encoder:
            with torch.no_grad():
                return self.set_encoder(x_support, y_support)
        return self.set_encoder(x_support, y_support)

    def train_step(self) -> Dict[str, Any]:
        self.conditioned_model.train()
        if self.freeze_set_encoder:
            self.set_encoder.eval()
        else:
            self.set_encoder.train()
        self.baseline_model.eval()

        x_support, y_support, x_query, y_query, subjects = self._sample_episode()
        x_support = x_support.to(self.device)
        y_support = y_support.to(self.device)
        x_query = x_query.to(self.device)
        y_query = y_query.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        c_subject = self.encode_subject(x_support, y_support)
        logits = self.conditioned_model.forward_episode(x_query, c_subject)

        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = y_query.reshape(-1)
        loss = F.cross_entropy(logits_flat, targets_flat, weight=self.class_weights)
        if not torch.isfinite(loss):
            diagnostics = {
                "loss": float(loss.detach().cpu().item()),
                "logits_finite": bool(torch.isfinite(logits).all().item()),
                "c_subject_finite": bool(torch.isfinite(c_subject).all().item()),
                "support_shape": tuple(x_support.shape),
                "query_shape": tuple(x_query.shape),
                "subjects": subjects,
            }
            raise FloatingPointError(
                f"Non-finite conditioned meta-training loss encountered: {diagnostics}"
            )
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            macro_f1 = f1_score(
                targets_flat.detach().cpu().numpy(),
                preds.reshape(-1).detach().cpu().numpy(),
                average="macro",
                zero_division=0,
            )

        self.state.steps += 1
        self.state.history["meta_train_loss"].append(float(loss.item()))
        self.state.history["meta_train_macro_f1"].append(float(macro_f1))

        return {
            "loss": float(loss.item()),
            "macro_f1": float(macro_f1),
            "subjects": subjects,
            "support_per_class": int(x_support.size(1) // max(1, len(self.activity_ids))),
            "support_shape": tuple(x_support.shape),
            "query_shape": tuple(x_query.shape),
            "step": self.state.steps,
        }

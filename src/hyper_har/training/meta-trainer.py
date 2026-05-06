from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.func import functional_call, vmap
from whar_datasets import Loader


@dataclass
class MetaTrainerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_subjects: int = 4
    support_per_class: int = 4
    query_per_class: int = 8
    use_vmap: bool = True
    seed: int | None = None
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


@dataclass
class MetaTrainerState:
    history: Dict[str, list[float]] = field(
        default_factory=lambda: {"meta_train_loss": [], "meta_train_macro_f1": []}
    )
    steps: int = 0


class SetToLoRAMetaTrainer:
    """Episodic meta-trainer for Set-to-LoRA adaptation."""

    def __init__(
        self,
        base_model: nn.Module,
        set_encoder: nn.Module,
        hypernet: nn.Module,
        loader: Loader,
        num_classes: int,
        config: MetaTrainerConfig,
        optimizer: torch.optim.Optimizer | None = None,
        class_weights: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
        activity_ids: Sequence[int] | None = None,
    ) -> None:
        self.base_model = base_model
        self.set_encoder = set_encoder
        self.hypernet = hypernet
        self.loader = loader
        self.num_classes = num_classes
        self.config = config
        self.device = torch.device(config.device)
        self.rng = np.random.default_rng(config.seed)

        self.base_model.to(self.device)
        self.set_encoder.to(self.device)
        self.hypernet.to(self.device)

        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        self._param_names = set(dict(self.base_model.named_parameters()).keys())
        self._buffers = dict(self.base_model.named_buffers())
        self.lora_rank = int(getattr(self.hypernet, "lora_rank", 1))
        default_alpha = float(self.lora_rank)
        self.lora_alpha = float(getattr(self.hypernet, "lora_alpha", default_alpha))
        self.lora_scale = float(self.lora_alpha / max(1, self.lora_rank))
        full_target_param_names: Dict[str, str] = {
            "conv1_pointwise": "conv_blocks.0.conv.0.pointwise.weight",
            "conv_last_pointwise": (
                f"conv_blocks.{len(self.base_model.conv_blocks) - 1}.conv.0.pointwise.weight"
            ),
            "gru_ih_fwd": "gru.weight_ih_l0",
            "gru_ih_rev": "gru.weight_ih_l0_reverse",
            "classifier": "classifier.0.weight",
        }
        target_adapter_names = getattr(self.hypernet, "module_names", None)
        if target_adapter_names is None:
            self.target_param_names = full_target_param_names
        else:
            self.target_param_names = {
                name: full_target_param_names[name]
                for name in target_adapter_names
                if name in full_target_param_names
            }
            if not self.target_param_names:
                raise ValueError("No recognized target adapter names found in hypernet.")
        missing = set(self.target_param_names.values()) - self._param_names
        if missing:
            raise ValueError(
                "One or more target parameters are missing in base model: "
                f"{sorted(missing)}"
            )

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

        self.optimizer = optimizer or torch.optim.Adam(
            list(self.set_encoder.parameters()) + list(self.hypernet.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.class_weights = (
            class_weights.to(self.device) if class_weights is not None else None
        )
        self.state = MetaTrainerState()

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
        support_plus_query = self.config.support_per_class + self.config.query_per_class
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        B = self.config.batch_subjects
        K = self.config.support_per_class
        Q = self.config.query_per_class

        sampled_subjects = self.rng.choice(
            np.asarray(self.eligible_subject_ids, dtype=np.int64),
            size=B,
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
                picked = self.rng.choice(candidate, size=K + Q, replace=False)
                support_inds = picked[:K]
                query_inds = picked[K:]

                for idx in support_inds.tolist():
                    subject_support_x.append(self._sample_window_array(int(idx)))
                    subject_support_y.append(activity_id)
                for idx in query_inds.tolist():
                    subject_query_x.append(self._sample_window_array(int(idx)))
                    subject_query_y.append(activity_id)

            support_perm = self.rng.permutation(len(subject_support_x))
            query_perm = self.rng.permutation(len(subject_query_x))

            support_x_all.append(np.stack([subject_support_x[i] for i in support_perm], axis=0))
            support_y_all.append(
                np.asarray([subject_support_y[i] for i in support_perm], dtype=np.int64)
            )
            query_x_all.append(np.stack([subject_query_x[i] for i in query_perm], axis=0))
            query_y_all.append(
                np.asarray([subject_query_y[i] for i in query_perm], dtype=np.int64)
            )

        x_support = torch.from_numpy(np.stack(support_x_all, axis=0)).float().unsqueeze(2)
        y_support = torch.from_numpy(np.stack(support_y_all, axis=0)).long()
        x_query = torch.from_numpy(np.stack(query_x_all, axis=0)).float().unsqueeze(2)
        y_query = torch.from_numpy(np.stack(query_y_all, axis=0)).long()

        return x_support, y_support, x_query, y_query, [int(s) for s in sampled_subjects]

    @staticmethod
    def _compute_lora_delta(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.dim() == 5 and B.dim() == 5:
            A2 = A.squeeze(-1).squeeze(-1)
            B2 = B.squeeze(-1).squeeze(-1)
            return torch.bmm(B2, A2).unsqueeze(-1).unsqueeze(-1)
        if A.dim() == 3 and B.dim() == 3:
            return torch.bmm(B, A)
        raise ValueError(
            f"Unsupported LoRA tensor shapes A={tuple(A.shape)}, B={tuple(B.shape)}."
        )

    def _build_batched_params(
        self, batch_size: int, lora_weights: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        params = dict(self.base_model.named_parameters())
        batched_params = {
            name: param.unsqueeze(0).expand(batch_size, *param.shape)
            for name, param in params.items()
        }

        for adapter_name, param_name in self.target_param_names.items():
            if adapter_name not in lora_weights:
                raise KeyError(f"Missing generated adapter '{adapter_name}'.")
            A, B = lora_weights[adapter_name]
            delta = self._compute_lora_delta(A, B)
            if delta.shape != batched_params[param_name].shape:
                raise ValueError(
                    f"Delta shape mismatch for {adapter_name}: "
                    f"delta={tuple(delta.shape)}, param={tuple(batched_params[param_name].shape)}"
                )
            batched_params[param_name] = batched_params[param_name] + (
                delta * self.lora_scale
            )

        return batched_params

    def _forward_with_subject_params(
        self, subject_params: Mapping[str, torch.Tensor], x_subject_query: torch.Tensor
    ) -> torch.Tensor:
        merged = {**subject_params, **self._buffers}
        return functional_call(self.base_model, merged, (x_subject_query,), strict=False)

    def _forward_queries_vmap(
        self, batched_params: Mapping[str, torch.Tensor], x_query: torch.Tensor
    ) -> torch.Tensor:
        return vmap(self._forward_with_subject_params, in_dims=(0, 0))(
            batched_params, x_query
        )

    def _forward_queries_loop(
        self, batched_params: Mapping[str, torch.Tensor], x_query: torch.Tensor
    ) -> torch.Tensor:
        B = x_query.size(0)
        logits_per_subject: list[torch.Tensor] = []
        for b in range(B):
            subject_params = {name: param[b] for name, param in batched_params.items()}
            subject_logits = self._forward_with_subject_params(subject_params, x_query[b])
            logits_per_subject.append(subject_logits)
        return torch.stack(logits_per_subject, dim=0)

    def train_step(self, use_vmap: bool | None = None) -> Dict[str, Any]:
        self.set_encoder.train()
        self.hypernet.train()
        self.base_model.eval()

        x_support, y_support, x_query, y_query, subjects = self._sample_episode()
        x_support = x_support.to(self.device)
        y_support = y_support.to(self.device)
        x_query = x_query.to(self.device)
        y_query = y_query.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        c_subject = self.set_encoder(x_support, y_support)
        lora_weights = self.hypernet(c_subject)

        B = x_query.size(0)
        batched_params = self._build_batched_params(B, lora_weights)

        use_vmap_flag = self.config.use_vmap if use_vmap is None else use_vmap
        vmap_error: str | None = None
        used_vmap = bool(use_vmap_flag)
        if use_vmap_flag:
            try:
                logits = self._forward_queries_vmap(batched_params, x_query)
            except RuntimeError as exc:
                # GRU currently has limited vmap support in some PyTorch versions.
                if "aten::gru.input" not in str(exc):
                    raise
                logits = self._forward_queries_loop(batched_params, x_query)
                used_vmap = False
                vmap_error = str(exc)
        else:
            logits = self._forward_queries_loop(batched_params, x_query)
            used_vmap = False

        logits_flat = logits.reshape(-1, logits.size(-1))
        y_query_flat = y_query.reshape(-1)
        loss = F.cross_entropy(logits_flat, y_query_flat, weight=self.class_weights)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            macro_f1 = f1_score(
                y_query_flat.detach().cpu().numpy(),
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
            "support_shape": tuple(x_support.shape),
            "query_shape": tuple(x_query.shape),
            "use_vmap": used_vmap,
            "vmap_error": vmap_error,
            "step": self.state.steps,
        }

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
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
from hyper_har.hypernet.hypernet import MLPResidualBlock
from hyper_har.splitting import MetaLOSOSplitter

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


META_TRAINER_MODULE = _load_module_from_path(
    "hyper_har_meta_trainer_update",
    SRC / "hyper_har" / "training" / "meta-trainer.py",
)
MetaTrainerConfig = META_TRAINER_MODULE.MetaTrainerConfig
SetToLoRAMetaTrainer = META_TRAINER_MODULE.SetToLoRAMetaTrainer


# Run settings
DATASET_ID = WHARDatasetID.WEAR
BACKBONE_DIR = ROOT / "artifacts" / "loso_cv"
OUTPUT_ROOT_DIR = ROOT / "artifacts" / "meta_update_loso_cv"
META_UPDATE_VARIANT_NAME = os.getenv("META_UPDATE_VARIANT_NAME")

# Episodic settings
TRAIN_SUBJECTS_PER_EPISODE = 4
TRAIN_SUPPORT_PER_CLASS_CHOICES = (2, 4, 8, 12, 16, 20)
QUERY_PER_CLASS = 8
EVAL_SUPPORT_PER_CLASS_CHOICES = (2, 4, 8, 12, 16, 20)
EVAL_QUERY_PER_CLASS = 16
TRAIN_EPISODES_PER_EPOCH = 64
EVAL_EPISODES = 128
USE_VMAP = True
ENABLE_EARLY_STOPPING = True

# Learned-update model settings
HIDDEN_DIM = 64
LABEL_EMBED_DIM = 32
NUM_HEADS = 4
LORA_RANK = 4
LORA_ALPHA = 1.0
ADAPTER_DELTA_L2 = 1e-4
META_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.0
ENABLE_CONV1_ADAPTER = False
ENABLE_CONV_LAST_ADAPTER = False
FORCE_BASE_EVAL = True


class SupportErrorSetEncoder(nn.Module):
    """Encode support examples using frozen-base features, logits, and residuals."""

    def __init__(
        self,
        base_model: TinierHAR,
        num_classes: int,
        hidden_dim: int = HIDDEN_DIM,
        label_embed_dim: int = LABEL_EMBED_DIM,
        num_heads: int = NUM_HEADS,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.feature_dim = 2 * base_model.nb_units_gru
        self.output_dim = (num_classes + 1) * hidden_dim

        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        self.label_embedding = nn.Embedding(num_classes, label_embed_dim)
        token_dim = (
            self.feature_dim
            + label_embed_dim
            + num_classes  # logits
            + num_classes  # probabilities
            + num_classes  # one_hot(label) - probabilities
            + 1  # per-item CE
        )
        self.token_mlp = nn.Sequential(
            nn.Linear(token_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim, eps=1e-5),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.class_queries = nn.Parameter(
            torch.randn(num_classes, 1, hidden_dim) * 0.02
        )
        self.class_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        raw_stats_dim = 2 * base_model.input_channels
        error_stats_dim = (3 * num_classes) + 1
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim + raw_stats_dim + error_stats_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim, eps=1e-5),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if FORCE_BASE_EVAL:
            self.base_model.eval()
        return self

    def _base_support_signals(
        self, x_flat: torch.Tensor, y_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features = self.base_model.encode(x_flat)
            logits = self.base_model.classifier(features)
            probs = torch.softmax(logits, dim=-1)
            one_hot = F.one_hot(y_flat, num_classes=self.num_classes).to(probs.dtype)
            residual = one_hot - probs
            ce = F.cross_entropy(logits, y_flat, reduction="none").unsqueeze(-1)
        return features, logits, probs, residual, ce

    def forward(self, x_support: torch.Tensor, y_support: torch.Tensor) -> torch.Tensor:
        # x_support: (B, N, 1, T, S), y_support: (B, N)
        B, N, C_in, T, S = x_support.shape
        x_flat = x_support.reshape(B * N, C_in, T, S)
        y_flat = y_support.reshape(-1)

        features, logits, probs, residual, ce = self._base_support_signals(
            x_flat, y_flat
        )
        labels = self.label_embedding(y_flat)
        token = self.token_mlp(
            torch.cat([features, labels, logits, probs, residual, ce], dim=-1)
        )
        z = token.view(B, N, self.hidden_dim)

        class_contexts: list[torch.Tensor] = []
        for class_id in range(self.num_classes):
            key_padding_mask = y_support != class_id
            query = self.class_queries[class_id].unsqueeze(0).expand(B, -1, -1)
            valid_rows = ~key_padding_mask.all(dim=1)
            class_out = torch.zeros(
                (B, 1, self.hidden_dim), device=z.device, dtype=z.dtype
            )
            if valid_rows.any():
                attn_out, _ = self.class_attention(
                    query[valid_rows],
                    z[valid_rows],
                    z[valid_rows],
                    key_padding_mask=key_padding_mask[valid_rows],
                )
                class_out[valid_rows] = attn_out
            class_contexts.append(class_out.squeeze(1))

        raw = x_support.squeeze(2)
        raw_mean = raw.mean(dim=(1, 2))
        raw_std = raw.std(dim=(1, 2), unbiased=False)
        z_global = z.mean(dim=1)
        logits_b = logits.view(B, N, self.num_classes)
        probs_b = probs.view(B, N, self.num_classes)
        residual_b = residual.view(B, N, self.num_classes)
        ce_b = ce.view(B, N, 1)
        error_stats = torch.cat(
            [
                logits_b.mean(dim=1),
                probs_b.mean(dim=1),
                residual_b.mean(dim=1),
                ce_b.mean(dim=1),
            ],
            dim=-1,
        )
        global_context = self.global_mlp(
            torch.cat([z_global, raw_mean, raw_std, error_stats], dim=-1)
        )

        return torch.cat([*class_contexts, global_context], dim=-1)


class LoRAUpdateNet(nn.Module):
    """Predict a support-conditioned LoRA update.

    This is framed as a learned update from a shared low-rank basis: each target
    module owns a learned A matrix, while the support-conditioned network
    predicts B. This avoids starting both LoRA factors at zero.
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        c_subject_dim: int,
        lora_rank: int = LORA_RANK,
        lora_alpha: float = LORA_ALPHA,
        enable_conv1_adapter: bool = ENABLE_CONV1_ADAPTER,
        enable_conv_last_adapter: bool = ENABLE_CONV_LAST_ADAPTER,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        backbone_cfg = DEFAULT_CONFIG.backbone
        nb_filters = backbone_cfg.nb_filters
        nb_units_gru = backbone_cfg.nb_units_gru
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scale = float(self.lora_alpha / max(1, self.lora_rank))

        gru_in_dim = (2 * nb_filters) * num_channels
        self.target_shapes: dict[str, tuple[int, int]] = {}
        if enable_conv1_adapter:
            self.target_shapes["conv1_pointwise"] = (nb_filters, 1)
        if enable_conv_last_adapter:
            self.target_shapes["conv_last_pointwise"] = (2 * nb_filters, 2 * nb_filters)
        self.target_shapes["gru_ih_fwd"] = (3 * nb_units_gru, gru_in_dim)
        self.target_shapes["gru_ih_rev"] = (3 * nb_units_gru, gru_in_dim)
        self.target_shapes["classifier"] = (num_classes, 2 * nb_units_gru)
        self.module_names = list(self.target_shapes.keys())

        self.context_encoder = nn.Sequential(
            nn.Linear(c_subject_dim, 256),
            nn.LayerNorm(256, eps=1e-5),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.module_embedding = nn.Embedding(len(self.module_names), 32)
        self.mixer = nn.Sequential(
            nn.Linear(160, 128),
            nn.SiLU(),
            MLPResidualBlock(dim=128, hidden_dim=512, dropout=dropout),
            nn.LayerNorm(128, eps=1e-5),
            nn.Linear(128, 256),
            nn.SiLU(),
        )

        self.shared_A = nn.ParameterDict()
        self.b_heads = nn.ModuleDict()
        for name, (out_dim, in_dim) in self.target_shapes.items():
            A = nn.Parameter(torch.empty(self.lora_rank, in_dim))
            nn.init.normal_(A, mean=0.0, std=1.0 / math.sqrt(max(1, in_dim)))
            self.shared_A[name] = A

            head = nn.Linear(256, out_dim * self.lora_rank)
            nn.init.normal_(head.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(head.bias)
            self.b_heads[name] = head

    def forward(
        self, c_subject: torch.Tensor
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        batch_size = c_subject.size(0)
        device = c_subject.device
        c = self.context_encoder(c_subject)
        module_ids = torch.arange(len(self.module_names), device=device)
        e_mod = (
            self.module_embedding(module_ids).unsqueeze(0).expand(batch_size, -1, -1)
        )
        c_mod = c.unsqueeze(1).expand(-1, len(self.module_names), -1)
        mixed = self.mixer(torch.cat([c_mod, e_mod], dim=-1))

        updates: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for module_idx, name in enumerate(self.module_names):
            out_dim, in_dim = self.target_shapes[name]
            A_base = self.shared_A[name].unsqueeze(0).expand(batch_size, -1, -1)
            B = self.b_heads[name](mixed[:, module_idx, :]).view(
                batch_size, out_dim, self.lora_rank
            )
            if "conv" in name:
                A = A_base.view(batch_size, self.lora_rank, in_dim, 1, 1)
                B = B.view(batch_size, out_dim, self.lora_rank, 1, 1)
            else:
                A = A_base
            updates[name] = (A, B)
        return updates


@dataclass
class SplitMetaUpdateResult:
    split_index: int
    subject_id: int
    best_epoch: int
    best_val_loss: float
    train_loss_at_best: float
    train_macro_f1_at_best: float
    val_macro_f1_at_best: float
    val_macro_f1_improvement_at_best: float
    test_loss: float
    test_macro_f1: float
    test_base_macro_f1: float
    test_macro_f1_improvement: float
    checkpoint_path: str


def _slugify_path_component(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "run"


def _resolve_variant_name(train_cfg: Any) -> str:
    if META_UPDATE_VARIANT_NAME and META_UPDATE_VARIANT_NAME.strip():
        return _slugify_path_component(META_UPDATE_VARIANT_NAME)
    parts = [
        "upd",
        f"b{TRAIN_SUBJECTS_PER_EPISODE}",
        f"ktrain{'-'.join(str(k) for k in TRAIN_SUPPORT_PER_CLASS_CHOICES)}",
        f"keval{'-'.join(str(k) for k in EVAL_SUPPORT_PER_CLASS_CHOICES)}",
        f"q{QUERY_PER_CLASS}-{EVAL_QUERY_PER_CLASS}",
        f"r{LORA_RANK}",
        f"a{LORA_ALPHA:g}",
        f"lr{META_LEARNING_RATE:g}",
        f"l2{ADAPTER_DELTA_L2:g}",
        f"conv{int(ENABLE_CONV1_ADAPTER)}{int(ENABLE_CONV_LAST_ADAPTER)}",
        f"epochs{train_cfg.num_epochs}",
    ]
    return _slugify_path_component("_".join(parts))


def _run_config_payload(train_cfg: Any, variant_name: str) -> dict[str, Any]:
    return {
        "variant_name": variant_name,
        "method": "support_error_conditioned_lora_update",
        "dataset_id": str(DATASET_ID),
        "backbone_dir": str(BACKBONE_DIR),
        "train_subjects_per_episode": TRAIN_SUBJECTS_PER_EPISODE,
        "train_support_per_class_choices": list(TRAIN_SUPPORT_PER_CLASS_CHOICES),
        "query_per_class": QUERY_PER_CLASS,
        "eval_support_per_class_choices": list(EVAL_SUPPORT_PER_CLASS_CHOICES),
        "eval_query_per_class": EVAL_QUERY_PER_CLASS,
        "train_episodes_per_epoch": TRAIN_EPISODES_PER_EPOCH,
        "eval_episodes": EVAL_EPISODES,
        "use_vmap": USE_VMAP,
        "early_stopping_metric": "val_macro_f1_improvement",
        "hidden_dim": HIDDEN_DIM,
        "label_embed_dim": LABEL_EMBED_DIM,
        "num_heads": NUM_HEADS,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "adapter_delta_l2": ADAPTER_DELTA_L2,
        "meta_learning_rate": META_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "enable_conv1_adapter": ENABLE_CONV1_ADAPTER,
        "enable_conv_last_adapter": ENABLE_CONV_LAST_ADAPTER,
        "epochs": train_cfg.num_epochs,
        "patience": train_cfg.patience,
    }


def _is_completed_split(split_dir: Path) -> bool:
    return all(
        path.exists()
        for path in [
            split_dir / "best_meta_update_modules.pt",
            split_dir / "meta_update_metrics.json",
            split_dir / "meta_update_history.json",
        ]
    )


def _load_existing_metrics(split_dir: Path) -> dict[str, Any] | None:
    metrics_path = split_dir / "meta_update_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _infer_window_size(loader: Loader, indices: Sequence[int]) -> int:
    sample = loader.get_sample(int(indices[0]))
    if not sample:
        raise ValueError("Could not infer window size: empty sample.")
    x = np.asarray(sample[0])
    if x.ndim == 2:
        return int(x.shape[0])
    if x.ndim == 3 and x.shape[0] == 1:
        return int(x.shape[1])
    raise ValueError(f"Unexpected sample shape for window inference: {tuple(x.shape)}")


def _infer_subject_id(loader: Loader, indices: Sequence[int], fallback: int) -> int:
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    session_meta = (
        loader.session_df[["session_id", "subject_id"]]
        .drop_duplicates("session_id")
        .set_index("session_id")
    )
    merged = subset.join(session_meta, on="session_id", how="left")
    subjects = merged["subject_id"].dropna().astype(int).unique().tolist()
    return int(subjects[0]) if len(subjects) == 1 else fallback


def _subject_ids_for_indices(loader: Loader, indices: Sequence[int]) -> list[int]:
    if len(indices) == 0:
        return []
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    session_meta = (
        loader.session_df[["session_id", "subject_id"]]
        .drop_duplicates("session_id")
        .set_index("session_id")
    )
    merged = subset.join(session_meta, on="session_id", how="left")
    if merged["subject_id"].isna().any():
        raise ValueError("Missing subject_id while inferring split debug info.")
    return sorted(set(int(x) for x in merged["subject_id"].tolist()))


def _activity_support_by_subject(
    loader: Loader, indices: Sequence[int]
) -> tuple[dict[int, dict[int, int]], list[int]]:
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    subset["window_index"] = subset.index.astype(int)
    session_meta = loader.session_df[
        ["session_id", "subject_id", "activity_id"]
    ].drop_duplicates("session_id")
    merged = subset.merge(session_meta, on="session_id", how="left")
    if merged["subject_id"].isna().any() or merged["activity_id"].isna().any():
        raise ValueError("Missing subject/activity metadata.")

    counts = (
        merged.groupby(["subject_id", "activity_id"])
        .size()
        .rename("count")
        .reset_index()
    )
    activity_ids = sorted(int(x) for x in counts["activity_id"].unique().tolist())
    support: dict[int, dict[int, int]] = {}
    for row in counts.itertuples(index=False):
        support.setdefault(int(row.subject_id), {})[int(row.activity_id)] = int(
            row.count
        )
    return support, activity_ids


def _choose_activity_ids(
    loader: Loader,
    indices: Sequence[int],
    needed_per_subject_activity: int,
    min_subjects: int,
) -> list[int]:
    support, activities = _activity_support_by_subject(loader, indices)
    candidate = activities.copy()
    while candidate:
        eligible_subjects = [
            sid
            for sid, per_activity in support.items()
            if all(
                per_activity.get(aid, 0) >= needed_per_subject_activity
                for aid in candidate
            )
        ]
        if len(eligible_subjects) >= min_subjects:
            return candidate

        support_counts = {
            aid: sum(
                1
                for per_activity in support.values()
                if per_activity.get(aid, 0) >= needed_per_subject_activity
            )
            for aid in candidate
        }
        drop_aid = min(candidate, key=lambda aid: (support_counts[aid], aid))
        candidate = [aid for aid in candidate if aid != drop_aid]

    raise ValueError("No activity set has enough support for episodic sampling.")


def _fetch_class_weights(
    loader: object, split: object, num_classes: int
) -> torch.Tensor | None:
    get_weights = getattr(loader, "get_class_weights", None)
    if get_weights is None:
        return None

    weights_obj = None
    try:
        n_params = len(inspect.signature(get_weights).parameters)
    except (TypeError, ValueError):
        n_params = -1

    if n_params in (0, -1):
        try:
            weights_obj = get_weights()
        except TypeError:
            pass

    if weights_obj is None and n_params in (1, -1):
        train_indices = getattr(split, "train_indices", None)
        if train_indices is not None:
            try:
                weights_obj = get_weights(train_indices)
            except TypeError:
                pass

    if weights_obj is None:
        return None

    if isinstance(weights_obj, dict):
        weights = torch.ones(num_classes, dtype=torch.float32)
        for class_id in range(num_classes):
            raw_w = float(weights_obj.get(class_id, 1.0))
            weights[class_id] = 0.0 if raw_w < 0.0 else raw_w
        return weights

    weights = torch.as_tensor(weights_obj, dtype=torch.float32).view(-1)
    if weights.numel() != num_classes:
        raise ValueError(
            f"Class weights length mismatch: expected {num_classes}, got {weights.numel()}."
        )
    return torch.where(weights < 0.0, torch.zeros_like(weights), weights)


def _build_episode_bank(
    trainer: SetToLoRAMetaTrainer,
    episodes: int,
    support_per_class_choices: Sequence[int],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]]:
    choices = [int(k) for k in support_per_class_choices if int(k) > 0]
    bank = []
    for episode_idx in range(episodes):
        bank.append(
            trainer._sample_episode(
                support_per_class=choices[episode_idx % len(choices)]
            )
        )
    return bank


def _run_meta_eval(
    trainer: SetToLoRAMetaTrainer,
    episodes: int,
    use_vmap: bool,
    episode_bank: Sequence[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]
    ]
    | None = None,
) -> dict[str, Any]:
    trainer.base_model.eval()
    trainer.set_encoder.eval()
    trainer.hypernet.eval()

    losses: list[float] = []
    base_losses: list[float] = []
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    base_preds_all: list[torch.Tensor] = []
    delta_norms_by_module: dict[str, list[float]] = {}

    iterator = (
        episode_bank
        if episode_bank is not None
        else (trainer._sample_episode() for _ in range(episodes))
    )
    base_params = dict(trainer.base_model.named_parameters())

    with torch.no_grad():
        for x_support, y_support, x_query, y_query, _subjects in iterator:
            x_support = x_support.to(trainer.device)
            y_support = y_support.to(trainer.device)
            x_query = x_query.to(trainer.device)
            y_query = y_query.to(trainer.device)
            targets_flat = y_query.reshape(-1)

            base_logits = trainer.base_model(
                x_query.reshape(-1, *x_query.shape[2:])
            ).reshape(x_query.size(0), x_query.size(1), -1)
            base_logits_flat = base_logits.reshape(-1, base_logits.size(-1))
            base_loss = F.cross_entropy(
                base_logits_flat, targets_flat, weight=trainer.class_weights
            )

            c_subject = trainer.set_encoder(x_support, y_support)
            lora_weights = trainer.hypernet(c_subject)
            for adapter_name, param_name in trainer.target_param_names.items():
                A, B = lora_weights[adapter_name]
                delta = trainer._compute_lora_delta(A, B) * trainer.lora_scale
                base_norm = base_params[param_name].detach().norm().clamp_min(1e-12)
                rel_norm = delta.flatten(1).norm(dim=1) / base_norm
                delta_norms_by_module.setdefault(adapter_name, []).append(
                    float(rel_norm.mean().item())
                )

            batched_params = trainer._build_batched_params(
                x_query.size(0), lora_weights
            )
            if use_vmap:
                try:
                    logits = trainer._forward_queries_vmap(batched_params, x_query)
                except RuntimeError as exc:
                    if "aten::gru.input" not in str(exc):
                        raise
                    logits = trainer._forward_queries_loop(batched_params, x_query)
            else:
                logits = trainer._forward_queries_loop(batched_params, x_query)

            logits_flat = logits.reshape(-1, logits.size(-1))
            loss = F.cross_entropy(
                logits_flat, targets_flat, weight=trainer.class_weights
            )
            losses.append(float(loss.item()))
            base_losses.append(float(base_loss.item()))
            all_preds.append(logits.argmax(dim=-1).reshape(-1).cpu())
            base_preds_all.append(base_logits.argmax(dim=-1).reshape(-1).cpu())
            all_targets.append(targets_flat.cpu())

    preds_t = torch.cat(all_preds)
    base_preds_t = torch.cat(base_preds_all)
    targets_t = torch.cat(all_targets)
    macro_f1 = f1_score(
        targets_t.numpy(), preds_t.numpy(), average="macro", zero_division=0
    )
    base_macro_f1 = f1_score(
        targets_t.numpy(), base_preds_t.numpy(), average="macro", zero_division=0
    )
    delta_norm_by_module_mean = {
        name: sum(values) / max(1, len(values))
        for name, values in delta_norms_by_module.items()
    }
    delta_norm_mean = sum(delta_norm_by_module_mean.values()) / max(
        1, len(delta_norm_by_module_mean)
    )
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "macro_f1": float(macro_f1),
        "base_loss": sum(base_losses) / max(1, len(base_losses)),
        "base_macro_f1": float(base_macro_f1),
        "macro_f1_improvement": float(macro_f1 - base_macro_f1),
        "lora_relative_delta_norm": float(delta_norm_mean),
        "lora_relative_delta_norm_by_module": delta_norm_by_module_mean,
    }


def main() -> None:
    cfg = get_dataset_cfg(DATASET_ID, datasets_dir=str(ROOT / "datasets"))
    train_cfg = DEFAULT_CONFIG.training
    variant_name = _resolve_variant_name(train_cfg)
    output_dir = OUTPUT_ROOT_DIR / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = _run_config_payload(train_cfg, variant_name)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    pre_pipeline = PreProcessingPipeline(cfg)
    _, session_df, window_df = pre_pipeline.run()
    splitter = MetaLOSOSplitter(cfg)
    folds = splitter.get_folds(session_df, window_df)
    summary_rows: list[dict[str, Any]] = []

    for split_idx, fold in enumerate(folds):
        split = fold.meta_split
        subject_id = int(fold.test_subject_id)
        split_dir = output_dir / f"subject_{subject_id}"
        split_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n=== Meta-update split {split_idx + 1}/{len(folds)} | subject={subject_id} ==="
        )

        if _is_completed_split(split_dir):
            existing = _load_existing_metrics(split_dir)
            if existing is not None:
                summary_rows.append(existing)
                print(f"Found completed artifacts, skipping subject {subject_id}.")
                continue

        post_pipeline = PostProcessingPipeline(
            cfg, pre_pipeline, window_df, split.train_indices
        )
        samples = post_pipeline.run()
        loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)
        inferred_subject = _infer_subject_id(
            loader, split.test_indices, fallback=subject_id
        )
        print(
            "Meta split subjects "
            f"(train/val/test)="
            f"{len(_subject_ids_for_indices(loader, split.train_indices))}/"
            f"{len(_subject_ids_for_indices(loader, split.val_indices))}/"
            f"{len(_subject_ids_for_indices(loader, split.test_indices))}"
        )

        num_channels = cfg.num_of_channels
        num_classes = cfg.num_of_activities
        window_size = _infer_window_size(loader, split.train_indices)
        base_model = TinierHAR(
            num_channels=num_channels,
            num_classes=num_classes,
            window_size=window_size,
            backbone_config=DEFAULT_CONFIG.backbone,
        )
        backbone_ckpt = BACKBONE_DIR / f"subject_{subject_id}" / "best_tinierhar.pt"
        if not backbone_ckpt.exists():
            raise FileNotFoundError(f"Missing pretrained backbone: {backbone_ckpt}")
        map_location = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        base_model.load_state_dict(torch.load(backbone_ckpt, map_location=map_location))
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad = False

        set_encoder = SupportErrorSetEncoder(
            base_model=base_model,
            num_classes=num_classes,
            hidden_dim=HIDDEN_DIM,
            label_embed_dim=LABEL_EMBED_DIM,
            num_heads=NUM_HEADS,
        )
        update_net = LoRAUpdateNet(
            num_channels=num_channels,
            num_classes=num_classes,
            c_subject_dim=set_encoder.output_dim,
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            enable_conv1_adapter=ENABLE_CONV1_ADAPTER,
            enable_conv_last_adapter=ENABLE_CONV_LAST_ADAPTER,
        )
        print(f"Set encoder output dim={set_encoder.output_dim}")
        print(f"Active learned-update adapters={update_net.module_names}")

        train_needed = max(TRAIN_SUPPORT_PER_CLASS_CHOICES) + QUERY_PER_CLASS
        eval_needed = max(EVAL_SUPPORT_PER_CLASS_CHOICES) + EVAL_QUERY_PER_CLASS
        train_activity_ids = _choose_activity_ids(
            loader, split.train_indices, train_needed, TRAIN_SUBJECTS_PER_EPISODE
        )
        val_activity_ids = _choose_activity_ids(
            loader, split.val_indices, eval_needed, 1
        )
        test_activity_ids = _choose_activity_ids(
            loader, split.test_indices, eval_needed, 1
        )
        print(
            "Episode activity counts "
            f"(train/val/test)="
            f"{len(train_activity_ids)}/{len(val_activity_ids)}/{len(test_activity_ids)}"
        )

        train_meta_cfg = MetaTrainerConfig(
            learning_rate=META_LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            adapter_delta_l2=ADAPTER_DELTA_L2,
            batch_subjects=TRAIN_SUBJECTS_PER_EPISODE,
            support_per_class=max(TRAIN_SUPPORT_PER_CLASS_CHOICES),
            support_per_class_choices=TRAIN_SUPPORT_PER_CLASS_CHOICES,
            query_per_class=QUERY_PER_CLASS,
            use_vmap=USE_VMAP,
            seed=split_idx,
        )
        val_meta_cfg = MetaTrainerConfig(
            learning_rate=META_LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            adapter_delta_l2=ADAPTER_DELTA_L2,
            batch_subjects=1,
            support_per_class=max(EVAL_SUPPORT_PER_CLASS_CHOICES),
            support_per_class_choices=EVAL_SUPPORT_PER_CLASS_CHOICES,
            query_per_class=EVAL_QUERY_PER_CLASS,
            use_vmap=USE_VMAP,
            seed=10_000 + split_idx,
            device=train_meta_cfg.device,
        )
        test_meta_cfg = MetaTrainerConfig(
            learning_rate=META_LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            adapter_delta_l2=ADAPTER_DELTA_L2,
            batch_subjects=1,
            support_per_class=max(EVAL_SUPPORT_PER_CLASS_CHOICES),
            support_per_class_choices=EVAL_SUPPORT_PER_CLASS_CHOICES,
            query_per_class=EVAL_QUERY_PER_CLASS,
            use_vmap=USE_VMAP,
            seed=20_000 + split_idx,
            device=train_meta_cfg.device,
        )

        optimizer = torch.optim.AdamW(
            list(set_encoder.parameters()) + list(update_net.parameters()),
            lr=META_LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        class_weights = _fetch_class_weights(loader, split, num_classes)
        if class_weights is not None:
            print(f"Using class weights: shape={tuple(class_weights.shape)}")
        else:
            print("No class weights returned, using unweighted cross-entropy.")
        train_trainer = SetToLoRAMetaTrainer(
            base_model=base_model,
            set_encoder=set_encoder,
            hypernet=update_net,
            loader=loader,
            num_classes=num_classes,
            config=train_meta_cfg,
            optimizer=optimizer,
            class_weights=class_weights,
            indices=split.train_indices,
            activity_ids=train_activity_ids,
        )
        val_trainer = SetToLoRAMetaTrainer(
            base_model=base_model,
            set_encoder=set_encoder,
            hypernet=update_net,
            loader=loader,
            num_classes=num_classes,
            config=val_meta_cfg,
            optimizer=optimizer,
            class_weights=class_weights,
            indices=split.val_indices,
            activity_ids=val_activity_ids,
        )
        test_trainer = SetToLoRAMetaTrainer(
            base_model=base_model,
            set_encoder=set_encoder,
            hypernet=update_net,
            loader=loader,
            num_classes=num_classes,
            config=test_meta_cfg,
            optimizer=optimizer,
            class_weights=class_weights,
            indices=split.test_indices,
            activity_ids=test_activity_ids,
        )
        val_episode_bank = _build_episode_bank(
            val_trainer, EVAL_EPISODES, EVAL_SUPPORT_PER_CLASS_CHOICES
        )
        test_episode_bank = _build_episode_bank(
            test_trainer, EVAL_EPISODES, EVAL_SUPPORT_PER_CLASS_CHOICES
        )

        best_val_loss = float("inf")
        best_epoch = -1
        best_train_loss = float("inf")
        best_train_f1 = 0.0
        best_val_f1 = float("-inf")
        best_val_improvement = float("-inf")
        patience_counter = 0
        best_ckpt_path = split_dir / "best_meta_update_modules.pt"
        val_loss_history: list[float] = []
        val_macro_f1_history: list[float] = []
        val_base_macro_f1_history: list[float] = []
        val_macro_f1_improvement_history: list[float] = []

        for epoch in range(1, train_cfg.num_epochs + 1):
            train_losses: list[float] = []
            train_f1s: list[float] = []
            progress = tqdm(
                range(TRAIN_EPISODES_PER_EPOCH),
                desc=f"MetaUpdate {epoch}/{train_cfg.num_epochs}",
                leave=False,
            )
            for _ in progress:
                step_metrics = train_trainer.train_step(use_vmap=USE_VMAP)
                train_losses.append(float(step_metrics["loss"]))
                train_f1s.append(float(step_metrics["macro_f1"]))
                progress.set_postfix(
                    loss=f"{sum(train_losses) / max(1, len(train_losses)):.4f}",
                    f1=f"{sum(train_f1s) / max(1, len(train_f1s)):.4f}",
                )

            train_loss = sum(train_losses) / max(1, len(train_losses))
            train_macro_f1 = sum(train_f1s) / max(1, len(train_f1s))
            val_metrics = _run_meta_eval(
                val_trainer,
                EVAL_EPISODES,
                use_vmap=USE_VMAP,
                episode_bank=val_episode_bank,
            )
            val_loss = float(val_metrics["loss"])
            val_macro_f1 = float(val_metrics["macro_f1"])
            val_base_macro_f1 = float(val_metrics["base_macro_f1"])
            val_macro_f1_improvement = float(val_metrics["macro_f1_improvement"])
            val_loss_history.append(val_loss)
            val_macro_f1_history.append(val_macro_f1)
            val_base_macro_f1_history.append(val_base_macro_f1)
            val_macro_f1_improvement_history.append(val_macro_f1_improvement)

            improved = bool(
                np.isfinite(val_macro_f1_improvement)
                and (
                    (val_macro_f1_improvement > best_val_improvement)
                    or (
                        np.isclose(val_macro_f1_improvement, best_val_improvement)
                        and (
                            (val_macro_f1 > best_val_f1)
                            or (
                                np.isclose(val_macro_f1, best_val_f1)
                                and np.isfinite(val_loss)
                                and val_loss < best_val_loss
                            )
                        )
                    )
                )
            )
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                best_train_loss = train_loss
                best_train_f1 = train_macro_f1
                best_val_f1 = val_macro_f1
                best_val_improvement = val_macro_f1_improvement
                patience_counter = 0
                torch.save(
                    {
                        "set_encoder": set_encoder.state_dict(),
                        "update_net": update_net.state_dict(),
                        "meta_config": asdict(train_meta_cfg),
                        "run_config": run_config,
                        "train_activity_ids": train_activity_ids,
                        "val_activity_ids": val_activity_ids,
                        "test_activity_ids": test_activity_ids,
                        "backbone_checkpoint": str(backbone_ckpt),
                    },
                    best_ckpt_path,
                )
            else:
                patience_counter += 1

            print(
                f"[MetaUpdate Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f} train_macro_f1={train_macro_f1:.4f} "
                f"val_loss={val_loss:.4f} val_macro_f1={val_macro_f1:.4f} "
                f"val_base_macro_f1={val_base_macro_f1:.4f} "
                f"val_improvement={val_macro_f1_improvement:+.4f} "
                f"best_improvement={best_val_improvement:+.4f} "
                f"patience={patience_counter}/{train_cfg.patience}"
            )
            if ENABLE_EARLY_STOPPING and patience_counter >= train_cfg.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
                break

        if not best_ckpt_path.exists():
            best_epoch = epoch
            best_train_loss = train_loss
            best_train_f1 = train_macro_f1
            best_val_f1 = val_macro_f1
            best_val_loss = val_loss
            best_val_improvement = val_macro_f1_improvement
            torch.save(
                {
                    "set_encoder": set_encoder.state_dict(),
                    "update_net": update_net.state_dict(),
                    "meta_config": asdict(train_meta_cfg),
                    "run_config": run_config,
                    "train_activity_ids": train_activity_ids,
                    "val_activity_ids": val_activity_ids,
                    "test_activity_ids": test_activity_ids,
                    "backbone_checkpoint": str(backbone_ckpt),
                    "checkpoint_reason": "fallback_last_epoch",
                },
                best_ckpt_path,
            )

        ckpt = torch.load(best_ckpt_path, map_location=train_trainer.device)
        set_encoder.load_state_dict(ckpt["set_encoder"])
        update_net.load_state_dict(ckpt["update_net"])
        test_metrics = _run_meta_eval(
            test_trainer,
            EVAL_EPISODES,
            use_vmap=USE_VMAP,
            episode_bank=test_episode_bank,
        )

        result = SplitMetaUpdateResult(
            split_index=split_idx,
            subject_id=inferred_subject,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            train_loss_at_best=best_train_loss,
            train_macro_f1_at_best=best_train_f1,
            val_macro_f1_at_best=best_val_f1,
            val_macro_f1_improvement_at_best=best_val_improvement,
            test_loss=float(test_metrics["loss"]),
            test_macro_f1=float(test_metrics["macro_f1"]),
            test_base_macro_f1=float(test_metrics["base_macro_f1"]),
            test_macro_f1_improvement=float(test_metrics["macro_f1_improvement"]),
            checkpoint_path=str(best_ckpt_path),
        )
        result_dict = asdict(result)
        result_dict["test_subject_id"] = int(fold.test_subject_id)
        result_dict["test_lora_relative_delta_norm"] = float(
            test_metrics["lora_relative_delta_norm"]
        )
        result_dict["test_lora_relative_delta_norm_by_module"] = test_metrics[
            "lora_relative_delta_norm_by_module"
        ]
        result_dict["early_stopping_metric"] = "val_macro_f1_improvement"
        result_dict["train_support_per_class_choices"] = list(
            TRAIN_SUPPORT_PER_CLASS_CHOICES
        )
        result_dict["eval_support_per_class_choices"] = list(
            EVAL_SUPPORT_PER_CLASS_CHOICES
        )
        result_dict["eval_query_per_class"] = EVAL_QUERY_PER_CLASS
        with (split_dir / "meta_update_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)

        history = dict(train_trainer.state.history)
        history["val_loss"] = val_loss_history
        history["val_macro_f1"] = val_macro_f1_history
        history["val_base_macro_f1"] = val_base_macro_f1_history
        history["val_macro_f1_improvement"] = val_macro_f1_improvement_history
        with (split_dir / "meta_update_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        summary_rows.append(result_dict)
        print(
            f"Subject {inferred_subject}: "
            f"base_f1={result.test_base_macro_f1:.4f} "
            f"adapted_f1={result.test_macro_f1:.4f} "
            f"improvement={result.test_macro_f1_improvement:+.4f}"
        )

    mean_macro_f1 = sum(float(r["test_macro_f1"]) for r in summary_rows) / max(
        1, len(summary_rows)
    )
    mean_base_macro_f1 = sum(
        float(r["test_base_macro_f1"]) for r in summary_rows
    ) / max(1, len(summary_rows))
    mean_improvement = sum(
        float(r["test_macro_f1_improvement"]) for r in summary_rows
    ) / max(1, len(summary_rows))
    summary = {
        "num_splits": len(summary_rows),
        "mean_test_macro_f1": mean_macro_f1,
        "mean_test_base_macro_f1": mean_base_macro_f1,
        "mean_test_macro_f1_improvement": mean_improvement,
        "variant_name": variant_name,
        "meta_output_dir": str(output_dir),
        "run_config": run_config,
        "splits": summary_rows,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Meta-update training finished ===")
    print(f"Mean base macro F1: {mean_base_macro_f1:.4f}")
    print(f"Mean adapted macro F1: {mean_macro_f1:.4f}")
    print(f"Mean improvement: {mean_improvement:+.4f}")
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()

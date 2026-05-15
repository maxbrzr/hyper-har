from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import (
    ROOT,
    SRC,
    SharedConfig,
    build_loader,
    build_or_load_loso_folds,
    config_fingerprint,
    infer_window_size,
    k_choices_from_range,
    prepare_cfg,
    sample_window_array,
    set_seed,
    split_indices_for_fold,
)
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm
from whar_datasets import PreProcessingPipeline, WHARDatasetID

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hyper_har.backbone.tinierhar import ConvBlock, TinierHAR
from hyper_har.config import DEFAULT_CONFIG


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AttentionSetEncoder = _load_module_from_path(
    "meta_train_attention_set_encoder",
    SRC / "hyper_har" / "set-encoder" / "attention.py",
).AttentionSetEncoder
PrototypicalSetEncoder = _load_module_from_path(
    "meta_train_simple_set_encoder",
    SRC / "hyper_har" / "set-encoder" / "simple.py",
).PrototypicalSetEncoder


@dataclass(frozen=True)
class Config:
    dataset_id: str = WHARDatasetID.WEAR.value
    datasets_dir: str = str(ROOT / "datasets")
    selected_activities: list[str] | None = None
    window_overlap: float = 0.0
    val_subjects: int = 6
    test_subjects: int = 1
    seed: int = 0

    train_subjects_per_episode: int = 4
    train_min_k_per_class: int = 1
    train_max_k_per_class: int = 32
    query_per_class: int = 8
    eval_query_per_class: int = 16
    train_episodes_per_epoch: int = 128
    eval_episodes_per_k: int = 32

    modulator_hidden_dim: int = 192
    modulator_dropout: float = 0.1
    modulator_use_tanh_gating: bool = False
    modulator_gamma_bound: float = 0.5
    modulator_beta_bound: float = 1.0
    pointwise_block_start: int = 0
    pre_classifier_dropout: float = 0.3

    meta_learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    warmup_ratio: float = 0.05
    weight_decay: float = 1e-3
    epochs: int = 100
    patience: int = 18
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    output_root: str = str(ROOT / "artifacts" / "blackbox_meta_loso")
    max_folds: int | None = None
    force_rerun: bool = False


RUN_CONFIG = Config()


class MultiHeadTinierHARModulator(nn.Module):
    def __init__(
        self,
        subject_embedding_dim: int,
        pointwise_bn_channels: Sequence[int],
        attention_feature_dim: int,
        hidden_dim: int,
        dropout: float,
        use_tanh_gating: bool,
        gamma_bound: float,
        beta_bound: float,
    ) -> None:
        super().__init__()
        self.subject_embedding_dim = int(subject_embedding_dim)
        self.pointwise_bn_channels = [int(ch) for ch in pointwise_bn_channels]
        self.attention_feature_dim = int(attention_feature_dim)
        self.use_tanh_gating = bool(use_tanh_gating)
        self.gamma_bound = float(gamma_bound)
        self.beta_bound = float(beta_bound)
        self.subject_norm = nn.LayerNorm(self.subject_embedding_dim, elementwise_affine=False)
        self.trunk = nn.Sequential(
            nn.Linear(self.subject_embedding_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.pointwise_heads = nn.ModuleList(
            [nn.Linear(int(hidden_dim), 2 * channels) for channels in self.pointwise_bn_channels]
        )
        self.attention_head = nn.Linear(int(hidden_dim), 2 * self.attention_feature_dim)
        self.reset_to_identity()

    def reset_to_identity(self) -> None:
        for head in [*self.pointwise_heads, self.attention_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _guard(self, gamma: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_tanh_gating:
            gamma = torch.tanh(gamma) * self.gamma_bound
            beta = torch.tanh(beta) * self.beta_bound
        return gamma, beta

    def forward(
        self, c_subject: torch.Tensor
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]:
        shared = self.trunk(self.subject_norm(c_subject))
        pointwise: list[tuple[torch.Tensor, torch.Tensor]] = []
        for head in self.pointwise_heads:
            gamma, beta = head(shared).chunk(2, dim=-1)
            pointwise.append(self._guard(gamma, beta))
        attn_gamma, attn_beta = self.attention_head(shared).chunk(2, dim=-1)
        return pointwise, self._guard(attn_gamma, attn_beta)


class CoTrainedCBNAttentionTinierHAR(nn.Module):
    def __init__(
        self,
        base_model: TinierHAR,
        subject_embedding_dim: int,
        modulator_hidden_dim: int,
        modulator_dropout: float,
        modulator_use_tanh_gating: bool,
        modulator_gamma_bound: float,
        modulator_beta_bound: float,
        pointwise_block_start: int,
        pre_classifier_dropout: float,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.subject_embedding_dim = int(subject_embedding_dim)
        self.pointwise_bns = self._collect_pointwise_batchnorms()
        self.pointwise_block_indices = [
            idx for idx in range(len(self.pointwise_bns)) if idx >= int(pointwise_block_start)
        ]
        if not self.pointwise_block_indices:
            raise ValueError("No pointwise BN blocks selected for modulation.")
        self.modulator = MultiHeadTinierHARModulator(
            subject_embedding_dim=subject_embedding_dim,
            pointwise_bn_channels=[self.pointwise_bns[idx].num_features for idx in self.pointwise_block_indices],
            attention_feature_dim=int(self.base_model.attention.in_features),
            hidden_dim=modulator_hidden_dim,
            dropout=modulator_dropout,
            use_tanh_gating=modulator_use_tanh_gating,
            gamma_bound=modulator_gamma_bound,
            beta_bound=modulator_beta_bound,
        )
        self.pre_classifier_dropout = nn.Dropout(float(pre_classifier_dropout))

    def _collect_pointwise_batchnorms(self) -> list[nn.BatchNorm2d]:
        bns: list[nn.BatchNorm2d] = []
        for block in self.base_model.conv_blocks:
            if not isinstance(block, ConvBlock):
                raise TypeError("TinierHAR conv_blocks must contain ConvBlock modules.")
            bn = block.conv[1]
            if not isinstance(bn, nn.BatchNorm2d):
                raise TypeError("Expected ConvBlock pointwise BatchNorm at block.conv[1].")
            bns.append(bn)
        return bns

    @staticmethod
    def _conditional_batch_norm(
        x: torch.Tensor,
        bn: nn.BatchNorm2d,
        delta_gamma: torch.Tensor,
        delta_beta: torch.Tensor,
    ) -> torch.Tensor:
        normalized = F.batch_norm(
            x,
            running_mean=bn.running_mean,
            running_var=bn.running_var,
            weight=None,
            bias=None,
            training=bn.training,
            momentum=bn.momentum,
            eps=bn.eps,
        )
        base_gamma = bn.weight.view(1, -1, 1, 1)
        base_beta = bn.bias.view(1, -1, 1, 1)
        return (
            (base_gamma + delta_gamma.view(x.size(0), -1, 1, 1)) * normalized
            + base_beta
            + delta_beta.view(x.size(0), -1, 1, 1)
        )

    def _forward_block(
        self,
        block: ConvBlock,
        x: torch.Tensor,
        pointwise_params: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        main = block.conv[0](x)
        if pointwise_params is None:
            main = block.conv[1](main)
        else:
            main = self._conditional_batch_norm(main, block.conv[1], *pointwise_params)
        main = block.conv[2](main)
        if block.use_maxpool:
            main = block.conv[3](main)
        if block.shortcut:
            return main + block.f_shortcut(x)
        return main

    def extract_conditioned_sequence(
        self,
        x: torch.Tensor,
        pointwise_params: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        params_by_block = dict(zip(self.pointwise_block_indices, pointwise_params))
        for idx, block in enumerate(self.base_model.conv_blocks):
            x = self._forward_block(block, x, params_by_block.get(idx))
        bsz, _, tlen, _ = x.shape
        x = x.permute(0, 2, 1, 3).reshape(bsz, tlen, -1)
        return self.base_model.dropout(x)

    def encode(self, x: torch.Tensor, c_subject: torch.Tensor) -> torch.Tensor:
        pointwise_params, attention_params = self.modulator(c_subject)
        x_seq = self.extract_conditioned_sequence(x, pointwise_params)
        x_seq, _ = self.base_model.gru(x_seq)
        gamma, beta = attention_params
        attention_input = (1.0 + gamma.unsqueeze(1)) * x_seq + beta.unsqueeze(1)
        scores = self.base_model.attention(attention_input)
        attn_weights = torch.softmax(scores, dim=1)
        return torch.sum(attn_weights * x_seq, dim=1)

    def forward(self, x: torch.Tensor, c_subject: torch.Tensor) -> torch.Tensor:
        features = self.encode(x, c_subject)
        return self.base_model.classifier(self.pre_classifier_dropout(features))

    def forward_unconditioned(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model.encode(x)
        return self.base_model.classifier(self.pre_classifier_dropout(features))

    def forward_episode(self, x_query: torch.Tensor, c_subject: torch.Tensor) -> torch.Tensor:
        bsz, n_query = x_query.shape[:2]
        c_expanded = (
            c_subject.unsqueeze(1)
            .expand(-1, n_query, -1)
            .reshape(bsz * n_query, -1)
        )
        logits = self.forward(x_query.reshape(bsz * n_query, *x_query.shape[2:]), c_expanded)
        return logits.view(bsz, n_query, -1)

    def forward_unconditioned_episode(self, x_query: torch.Tensor) -> torch.Tensor:
        bsz, n_query = x_query.shape[:2]
        logits = self.forward_unconditioned(x_query.reshape(bsz * n_query, *x_query.shape[2:]))
        return logits.view(bsz, n_query, -1)


def initialize_trainable_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
    if hasattr(model, "modulator"):
        model.modulator.reset_to_identity()


def _build_set_encoder(config: dict[str, Any], num_channels: int, num_classes: int, window_size: int) -> nn.Module:
    backbone = TinierHAR(
        num_channels=num_channels,
        num_classes=num_classes,
        window_size=window_size,
        backbone_config=DEFAULT_CONFIG.backbone,
    )
    se_cfg = replace(DEFAULT_CONFIG.set_encoder, include_global_context=False)
    encoder = str(config.get("encoder", "attention"))
    if encoder == "attention":
        set_encoder = AttentionSetEncoder(
            backbone=backbone,
            num_classes=num_classes,
            backbone_train_mode="unfreeze_all",
            force_conv_bn_eval=False,
            set_encoder_config=se_cfg,
        )
    else:
        set_encoder = PrototypicalSetEncoder(
            backbone=backbone,
            num_classes=num_classes,
            backbone_train_mode="unfreeze_all",
            force_conv_bn_eval=False,
            set_encoder_config=se_cfg,
        )
    return set_encoder


def _build_subject_activity_index(loader: Any, indices: Sequence[int]) -> dict[tuple[int, int], np.ndarray]:
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    subset["window_index"] = subset.index.astype(int)
    session_meta = loader.session_df[
        ["session_id", "subject_id", "activity_id"]
    ].drop_duplicates("session_id")
    merged = subset.merge(session_meta, on="session_id", how="left")
    if merged["subject_id"].isna().any() or merged["activity_id"].isna().any():
        raise ValueError("Missing subject/activity metadata while building episodes.")
    grouped = merged.groupby(["subject_id", "activity_id"])["window_index"]
    return {
        (int(subject_id), int(activity_id)): np.asarray(group.tolist(), dtype=np.int64)
        for (subject_id, activity_id), group in grouped
    }


def _choose_activity_ids(
    loader: Any,
    indices: Sequence[int],
    needed_per_subject_activity: int,
    min_subjects: int,
) -> list[int]:
    subject_activity = _build_subject_activity_index(loader, indices)
    subject_ids = sorted({sid for sid, _ in subject_activity.keys()})
    activity_ids = sorted({aid for _, aid in subject_activity.keys()})
    activities_by_subject = {
        int(sid): {
            int(aid)
            for aid in activity_ids
            if len(subject_activity.get((sid, aid), np.empty(0, dtype=np.int64)))
            >= int(needed_per_subject_activity)
        }
        for sid in subject_ids
    }

    candidates: list[tuple[int, list[int], list[int]]] = []
    for anchor_subject_id, anchor_activities in activities_by_subject.items():
        if not anchor_activities:
            continue
        eligible_subjects = [
            int(sid)
            for sid, supported_activities in activities_by_subject.items()
            if anchor_activities.issubset(supported_activities)
        ]
        if len(eligible_subjects) >= int(min_subjects):
            candidates.append(
                (
                    len(anchor_activities),
                    sorted(int(aid) for aid in anchor_activities),
                    sorted(eligible_subjects),
                )
            )

    if not candidates:
        raise ValueError(
            "No common activity subset satisfies episodic requirements: "
            f"needed_per_subject_activity={needed_per_subject_activity}, min_subjects={min_subjects}."
        )
    candidates.sort(key=lambda item: (item[0], len(item[2]), [-aid for aid in item[1]]), reverse=True)
    return candidates[0][1]


class EpisodicSampler:
    def __init__(
        self,
        loader: Any,
        indices: Sequence[int],
        activity_ids: Sequence[int],
        batch_subjects: int,
        support_per_class_choices: Sequence[int],
        query_per_class: int,
        seed: int,
    ) -> None:
        self.loader = loader
        self.activity_ids = [int(aid) for aid in activity_ids]
        self.batch_subjects = int(batch_subjects)
        self.support_per_class_choices = [int(k) for k in support_per_class_choices if int(k) > 0]
        self.query_per_class = int(query_per_class)
        self.rng = np.random.default_rng(int(seed))
        self.subject_activity = _build_subject_activity_index(loader, indices)
        self.eligible_subject_ids = self._eligible_subjects(max(self.support_per_class_choices))
        if len(self.eligible_subject_ids) < self.batch_subjects:
            raise ValueError(
                f"Need {self.batch_subjects} eligible subjects, found {len(self.eligible_subject_ids)}."
            )

    def _eligible_subjects(self, max_k: int) -> list[int]:
        needed = int(max_k) + self.query_per_class
        subject_ids = sorted({sid for sid, _ in self.subject_activity.keys()})
        return [
            int(sid)
            for sid in subject_ids
            if all(
                len(self.subject_activity.get((sid, aid), np.empty(0, dtype=np.int64))) >= needed
                for aid in self.activity_ids
            )
        ]

    def sample(self, support_per_class: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        k_shot = (
            int(support_per_class)
            if support_per_class is not None
            else int(self.rng.choice(np.asarray(self.support_per_class_choices, dtype=np.int64)))
        )
        sampled_subjects = self.rng.choice(
            np.asarray(self.eligible_subject_ids, dtype=np.int64),
            size=self.batch_subjects,
            replace=False,
        ).tolist()
        support_x_all: list[np.ndarray] = []
        support_y_all: list[np.ndarray] = []
        query_x_all: list[np.ndarray] = []
        query_y_all: list[np.ndarray] = []
        for subject_id in sampled_subjects:
            support_x: list[np.ndarray] = []
            support_y: list[int] = []
            query_x: list[np.ndarray] = []
            query_y: list[int] = []
            for activity_id in self.activity_ids:
                candidates = self.subject_activity[(int(subject_id), int(activity_id))]
                picked = self.rng.choice(candidates, size=k_shot + self.query_per_class, replace=False)
                for idx in picked[:k_shot].tolist():
                    support_x.append(sample_window_array(self.loader, int(idx)))
                    support_y.append(int(activity_id))
                for idx in picked[k_shot:].tolist():
                    query_x.append(sample_window_array(self.loader, int(idx)))
                    query_y.append(int(activity_id))
            s_perm = self.rng.permutation(len(support_x))
            q_perm = self.rng.permutation(len(query_x))
            support_x_all.append(np.stack([support_x[i] for i in s_perm], axis=0))
            support_y_all.append(np.asarray([support_y[i] for i in s_perm], dtype=np.int64))
            query_x_all.append(np.stack([query_x[i] for i in q_perm], axis=0))
            query_y_all.append(np.asarray([query_y[i] for i in q_perm], dtype=np.int64))
        x_support = torch.from_numpy(np.stack(support_x_all, axis=0)).float().unsqueeze(2)
        y_support = torch.from_numpy(np.stack(support_y_all, axis=0)).long()
        x_query = torch.from_numpy(np.stack(query_x_all, axis=0)).float().unsqueeze(2)
        y_query = torch.from_numpy(np.stack(query_y_all, axis=0)).long()
        return x_support, y_support, x_query, y_query, [int(sid) for sid in sampled_subjects]


@torch.no_grad()
def _run_eval(
    model: CoTrainedCBNAttentionTinierHAR,
    set_encoder: nn.Module,
    sampler: EpisodicSampler,
    episode_banks_by_k: Mapping[str, Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]]],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    model.eval()
    set_encoder.eval()
    metrics_by_k: dict[str, dict[str, Any]] = {}
    for k_str, episodes in episode_banks_by_k.items():
        losses: list[float] = []
        cond_preds: list[torch.Tensor] = []
        uncond_preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for x_support, y_support, x_query, y_query, _subjects in episodes:
            x_support = x_support.to(device)
            y_support = y_support.to(device)
            x_query = x_query.to(device)
            y_query = y_query.to(device)
            c_subject = set_encoder(x_support, y_support)
            logits = model.forward_episode(x_query, c_subject)
            uncond_logits = model.forward_unconditioned_episode(x_query)
            target_flat = y_query.reshape(-1)
            losses.append(float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_flat).item()))
            cond_preds.append(logits.argmax(dim=-1).reshape(-1).cpu())
            uncond_preds.append(uncond_logits.argmax(dim=-1).reshape(-1).cpu())
            targets.append(target_flat.cpu())
        y_true = torch.cat(targets).numpy() if targets else np.empty(0, dtype=np.int64)
        cond = torch.cat(cond_preds).numpy() if cond_preds else np.empty(0, dtype=np.int64)
        uncond = torch.cat(uncond_preds).numpy() if uncond_preds else np.empty(0, dtype=np.int64)
        cond_f1 = float(f1_score(y_true, cond, average="macro", zero_division=0)) if y_true.size else 0.0
        uncond_f1 = float(f1_score(y_true, uncond, average="macro", zero_division=0)) if y_true.size else 0.0
        metrics_by_k[str(k_str)] = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "macro_f1": cond_f1,
            "unconditioned_macro_f1": uncond_f1,
            "macro_f1_improvement": float(cond_f1 - uncond_f1),
            "episodes": int(len(episodes)),
            "eligible_subjects": int(len(sampler.eligible_subject_ids)),
        }
    mean_metrics = {
        key: float(np.mean([metrics[key] for metrics in metrics_by_k.values()]))
        if metrics_by_k
        else 0.0
        for key in ("loss", "macro_f1", "unconditioned_macro_f1", "macro_f1_improvement")
    }
    return mean_metrics, metrics_by_k


def _episode_banks(
    sampler: EpisodicSampler,
    support_choices: Sequence[int],
    episodes_per_k: int,
) -> dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]]]:
    return {
        str(int(k)): [sampler.sample(support_per_class=int(k)) for _ in range(int(episodes_per_k))]
        for k in support_choices
    }


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)
    output_root = Path(config.output_root)
    stage_dir = output_root / "02_blackbox_meta"
    stage_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = WHARDatasetID(config.dataset_id)
    cfg = prepare_cfg(
        dataset_id=dataset_id,
        datasets_dir=Path(config.datasets_dir),
        selected_activities=config.selected_activities,
        window_overlap=config.window_overlap,
    )
    pre = PreProcessingPipeline(cfg)
    _raw_df, session_df, window_df = pre.run()
    shared_cfg = SharedConfig(
        dataset_id=config.dataset_id,
        datasets_dir=config.datasets_dir,
        selected_activities=config.selected_activities,
        window_overlap=config.window_overlap,
        val_subjects=config.val_subjects,
        test_subjects=config.test_subjects,
        seed=config.seed,
    )
    manifest_path = output_root / "shared_splits" / "loso_subject_folds.json"
    folds = build_or_load_loso_folds(session_df, window_df, shared_cfg, manifest_path)
    if config.max_folds is not None:
        folds = folds[: int(config.max_folds)]

    set_encoder_summary_path = output_root / "01_set_encoder_supcon" / "summary.json"
    if not set_encoder_summary_path.exists():
        raise FileNotFoundError(f"Run set-encoder pretraining first: {set_encoder_summary_path}")
    set_encoder_summary = json.loads(set_encoder_summary_path.read_text(encoding="utf-8"))
    set_encoder_config = set_encoder_summary["config"]
    expected_split_config = {
        "dataset_id": config.dataset_id,
        "datasets_dir": config.datasets_dir,
        "selected_activities": config.selected_activities,
        "window_overlap": config.window_overlap,
        "val_subjects": config.val_subjects,
        "test_subjects": config.test_subjects,
        "seed": config.seed,
    }
    actual_split_config = {
        key: set_encoder_config.get(key) for key in expected_split_config
    }
    if actual_split_config != expected_split_config:
        raise ValueError(
            "Set-encoder pretraining config does not match meta-training split config. "
            f"Expected {expected_split_config}, found {actual_split_config}. "
            "Rerun meta-scripts/01_pretrain_set_encoder_supcon_loso.py."
        )
    set_manifest_path = Path(set_encoder_summary.get("splits_manifest_path", ""))
    if set_manifest_path != manifest_path:
        raise ValueError(
            "Set-encoder pretraining used a different LOSO manifest: "
            f"{set_manifest_path} != {manifest_path}. "
            "Rerun meta-scripts/01_pretrain_set_encoder_supcon_loso.py."
        )

    summary_rows: list[dict[str, Any]] = []
    skipped_folds: list[str] = []
    for split_idx, fold in enumerate(folds):
        split = split_indices_for_fold(session_df, window_df, fold)
        split_dir = stage_dir / fold.fold_id
        split_dir.mkdir(parents=True, exist_ok=True)
        fold_fp = config_fingerprint(
            {
                "stage": "02_blackbox_meta",
                "config": asdict(config),
                "shared_cfg": asdict(shared_cfg),
                "fold": asdict(fold),
            }
        )
        metrics_path = split_dir / "metrics.json"
        ckpt_path = split_dir / "best_blackbox_meta.pt"
        if not config.force_rerun and metrics_path.exists() and ckpt_path.exists():
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if existing.get("config_fingerprint") == fold_fp:
                print(f"[{fold.fold_id}] skipping (already complete)")
                summary_rows.append(existing)
                skipped_folds.append(fold.fold_id)
                continue

        set_ckpt = output_root / "01_set_encoder_supcon" / fold.fold_id / "best_set_encoder_supcon.pt"
        if not set_ckpt.exists():
            raise FileNotFoundError(f"Missing set encoder checkpoint: {set_ckpt}")
        fit_indices = sorted(set(split.train_indices + split.val_indices + split.test_indices))
        loader = build_loader(cfg, session_df, pre, window_df, fit_indices)
        window_size = infer_window_size(loader, fit_indices)
        num_channels = int(cfg.num_of_channels)
        num_classes = int(cfg.num_of_activities)

        set_encoder = _build_set_encoder(set_encoder_config, num_channels, num_classes, window_size)
        payload = torch.load(set_ckpt, map_location=config.device, weights_only=False)
        set_encoder.load_state_dict(payload["set_encoder"])
        set_encoder.to(device).eval()
        for param in set_encoder.parameters():
            param.requires_grad = False

        base_model = TinierHAR(
            num_channels=num_channels,
            num_classes=num_classes,
            window_size=window_size,
            backbone_config=DEFAULT_CONFIG.backbone,
        )
        model = CoTrainedCBNAttentionTinierHAR(
            base_model=base_model,
            subject_embedding_dim=int(getattr(set_encoder, "output_dim")),
            modulator_hidden_dim=config.modulator_hidden_dim,
            modulator_dropout=config.modulator_dropout,
            modulator_use_tanh_gating=config.modulator_use_tanh_gating,
            modulator_gamma_bound=config.modulator_gamma_bound,
            modulator_beta_bound=config.modulator_beta_bound,
            pointwise_block_start=config.pointwise_block_start,
            pre_classifier_dropout=config.pre_classifier_dropout,
        )
        initialize_trainable_weights(model)
        model.to(device)

        train_choices = k_choices_from_range(config.train_min_k_per_class, config.train_max_k_per_class)
        eval_choices = train_choices
        train_needed = max(train_choices) + int(config.query_per_class)
        eval_needed = max(eval_choices) + int(config.eval_query_per_class)
        train_activity_ids = _choose_activity_ids(
            loader, split.train_indices, train_needed, config.train_subjects_per_episode
        )
        val_activity_ids = _choose_activity_ids(loader, split.val_indices, eval_needed, 1)
        test_activity_ids = _choose_activity_ids(loader, split.test_indices, eval_needed, 1)

        train_sampler = EpisodicSampler(
            loader=loader,
            indices=split.train_indices,
            activity_ids=train_activity_ids,
            batch_subjects=config.train_subjects_per_episode,
            support_per_class_choices=train_choices,
            query_per_class=config.query_per_class,
            seed=config.seed + split_idx,
        )
        val_sampler = EpisodicSampler(
            loader=loader,
            indices=split.val_indices,
            activity_ids=val_activity_ids,
            batch_subjects=1,
            support_per_class_choices=eval_choices,
            query_per_class=config.eval_query_per_class,
            seed=config.seed + 10_000 + split_idx,
        )
        test_sampler = EpisodicSampler(
            loader=loader,
            indices=split.test_indices,
            activity_ids=test_activity_ids,
            batch_subjects=1,
            support_per_class_choices=eval_choices,
            query_per_class=config.eval_query_per_class,
            seed=config.seed + 20_000 + split_idx,
        )
        train_time_val_banks = _episode_banks(val_sampler, (max(eval_choices),), config.eval_episodes_per_k)
        final_val_banks = _episode_banks(val_sampler, eval_choices, config.eval_episodes_per_k)
        final_test_banks = _episode_banks(test_sampler, eval_choices, config.eval_episodes_per_k)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.meta_learning_rate, weight_decay=config.weight_decay)
        total_steps = max(1, int(config.epochs) * int(config.train_episodes_per_epoch))
        warmup_steps = max(1, int(total_steps * float(config.warmup_ratio)))
        warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=float(config.min_learning_rate),
        )
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

        best_val_f1 = float("-inf")
        best_val_loss = float("inf")
        best_epoch = -1
        patience_counter = 0
        history: list[dict[str, Any]] = []
        global_step = 0
        for epoch in range(1, int(config.epochs) + 1):
            model.train()
            set_encoder.eval()
            losses: list[float] = []
            train_f1s: list[float] = []
            for _ in tqdm(
                range(int(config.train_episodes_per_epoch)),
                desc=f"{fold.fold_id} blackbox meta {epoch}/{config.epochs}",
                leave=False,
            ):
                x_support, y_support, x_query, y_query, _subjects = train_sampler.sample()
                x_support = x_support.to(device)
                y_support = y_support.to(device)
                x_query = x_query.to(device)
                y_query = y_query.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    c_subject = set_encoder(x_support, y_support)
                logits = model.forward_episode(x_query, c_subject)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_query.reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite meta loss in {fold.fold_id}.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()
                global_step += 1
                losses.append(float(loss.item()))
                preds = logits.argmax(dim=-1).reshape(-1).detach().cpu().numpy()
                train_f1s.append(
                    float(
                        f1_score(
                            y_query.reshape(-1).detach().cpu().numpy(),
                            preds,
                            average="macro",
                            zero_division=0,
                        )
                    )
                )

            val_metrics, val_by_k = _run_eval(model, set_encoder, val_sampler, train_time_val_banks, device)
            row = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "train_macro_f1": float(np.mean(train_f1s)) if train_f1s else 0.0,
                "val_loss": float(val_metrics["loss"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_unconditioned_macro_f1": float(val_metrics["unconditioned_macro_f1"]),
                "val_macro_f1_improvement": float(val_metrics["macro_f1_improvement"]),
                "val_by_k": val_by_k,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            print(
                f"[{fold.fold_id}] epoch={epoch} train_loss={row['train_loss']:.4f} "
                f"train_f1={row['train_macro_f1']:.4f} val_f1={row['val_macro_f1']:.4f} "
                f"uncond={row['val_unconditioned_macro_f1']:.4f} "
                f"gain={row['val_macro_f1_improvement']:+.4f}"
            )
            improved = row["val_macro_f1"] > best_val_f1 or (
                np.isclose(row["val_macro_f1"], best_val_f1) and row["val_loss"] < best_val_loss
            )
            if improved:
                best_val_f1 = float(row["val_macro_f1"])
                best_val_loss = float(row["val_loss"])
                best_epoch = int(epoch)
                patience_counter = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "set_encoder": set_encoder.state_dict(),
                        "best_epoch": best_epoch,
                        "best_val_macro_f1": best_val_f1,
                        "best_val_loss": best_val_loss,
                        "config": asdict(config),
                        "set_encoder_config": set_encoder_config,
                    },
                    ckpt_path,
                )
            else:
                patience_counter += 1
            if patience_counter >= int(config.patience):
                break

        if not ckpt_path.exists():
            torch.save(
                {
                    "model": model.state_dict(),
                    "set_encoder": set_encoder.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_f1,
                    "best_val_loss": best_val_loss,
                    "config": asdict(config),
                    "set_encoder_config": set_encoder_config,
                },
                ckpt_path,
            )
        best_payload = torch.load(ckpt_path, map_location=config.device, weights_only=False)
        model.load_state_dict(best_payload["model"])
        val_final, val_final_by_k = _run_eval(model, set_encoder, val_sampler, final_val_banks, device)
        test_final, test_final_by_k = _run_eval(model, set_encoder, test_sampler, final_test_banks, device)
        fold_result = {
            "config_fingerprint": fold_fp,
            "fold_id": fold.fold_id,
            "train_subject_ids": fold.train_subject_ids,
            "val_subject_ids": fold.val_subject_ids,
            "test_subject_ids": fold.test_subject_ids,
            "best_epoch": int(best_epoch),
            "best_val_macro_f1": float(best_val_f1),
            "val_loss": float(val_final["loss"]),
            "val_macro_f1": float(val_final["macro_f1"]),
            "val_unconditioned_macro_f1": float(val_final["unconditioned_macro_f1"]),
            "val_macro_f1_improvement": float(val_final["macro_f1_improvement"]),
            "test_loss": float(test_final["loss"]),
            "test_macro_f1": float(test_final["macro_f1"]),
            "test_unconditioned_macro_f1": float(test_final["unconditioned_macro_f1"]),
            "test_macro_f1_improvement": float(test_final["macro_f1_improvement"]),
            "eval_k_values": [int(k) for k in eval_choices],
            "eval_episodes_per_k": int(config.eval_episodes_per_k),
            "val_by_k": val_final_by_k,
            "test_by_k": test_final_by_k,
            "checkpoint_path": str(ckpt_path),
        }
        metrics_path.write_text(json.dumps(fold_result, indent=2), encoding="utf-8")
        (split_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        summary_rows.append(fold_result)
        print(
            f"[{fold.fold_id}] final val_f1={fold_result['val_macro_f1']:.4f} "
            f"test_f1={fold_result['test_macro_f1']:.4f} "
            f"test_gain={fold_result['test_macro_f1_improvement']:+.4f}"
        )

    summary = {
        "config": asdict(config),
        "splits_manifest_path": str(manifest_path),
        "num_folds": len(summary_rows),
        "skipped_folds": skipped_folds,
        "mean_val_macro_f1": float(np.mean([row["val_macro_f1"] for row in summary_rows])) if summary_rows else 0.0,
        "mean_test_macro_f1": float(np.mean([row["test_macro_f1"] for row in summary_rows])) if summary_rows else 0.0,
        "mean_test_macro_f1_improvement": float(np.mean([row["test_macro_f1_improvement"] for row in summary_rows])) if summary_rows else 0.0,
        "folds": summary_rows,
    }
    (stage_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

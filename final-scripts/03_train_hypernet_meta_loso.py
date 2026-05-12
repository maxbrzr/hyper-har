from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from common import (
    DEFAULT_TRAIN_MAX_K_PER_CLASS,
    DEFAULT_TRAIN_MIN_K_PER_CLASS,
    ROOT,
    SharedConfig,
    build_or_load_loso_folds,
    config_fingerprint,
    k_choices_from_range,
    prepare_cfg,
    set_seed,
    split_indices_for_fold,
)
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from whar_datasets import (
    Loader,
    PostProcessingPipeline,
    PreProcessingPipeline,
    WHARDatasetID,
)

from hyper_har.backbone.tinierhar import TinierHAR
from hyper_har.config import DEFAULT_CONFIG
from hyper_har.hypernet.hypernet import HyperNet

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


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


SET_ENCODER_ATTENTION_MODULE = _load_module_from_path(
    "final_phase3_attention_set_encoder",
    SRC / "hyper_har" / "set-encoder" / "attention.py",
)
SET_ENCODER_SIMPLE_MODULE = _load_module_from_path(
    "final_phase3_simple_set_encoder",
    SRC / "hyper_har" / "set-encoder" / "simple.py",
)
META_TRAINER_MODULE = _load_module_from_path(
    "final_phase3_meta_trainer_module",
    SRC / "hyper_har" / "training" / "meta-trainer.py",
)
AttentionSetEncoder = SET_ENCODER_ATTENTION_MODULE.AttentionSetEncoder
PrototypicalSetEncoder = SET_ENCODER_SIMPLE_MODULE.PrototypicalSetEncoder
MetaTrainerConfig = META_TRAINER_MODULE.MetaTrainerConfig
SetToLoRAMetaTrainer = META_TRAINER_MODULE.SetToLoRAMetaTrainer


@dataclass(frozen=True)
class Config:
    dataset_id: str = WHARDatasetID.WEAR.value
    datasets_dir: str = str(ROOT / "datasets")
    selected_activities: list[str] | None = None
    window_overlap: float = 0.5
    subjects_per_group: int = 6
    seed: int = 0

    encoder: str = "attention"
    set_encoder_backbone_train_mode: str = "freeze_all"
    force_conv_bn_eval: bool = True

    train_subjects_per_episode: int = 4
    train_min_k_per_class: int = DEFAULT_TRAIN_MIN_K_PER_CLASS
    train_max_k_per_class: int = DEFAULT_TRAIN_MAX_K_PER_CLASS
    query_per_class: int = 8
    eval_query_per_class: int = 16
    train_episodes_per_epoch: int = 64
    eval_episodes: int = 64

    meta_learning_rate: float = 1e-4
    set_encoder_learning_rate: float = 1e-5
    min_learning_rate: float = 1e-6
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    adapter_delta_l2: float = 1e-4
    lora_rank: int = 8
    lora_alpha: float = 8.0
    enable_conv1_adapter: bool = False
    enable_conv_last_adapter: bool = False

    epochs: int = 100
    patience: int = 10
    unfreeze_set_encoder_after_epoch: int | None = None
    use_vmap: bool = False
    device: str = "cpu"
    output_root: str = str(ROOT / "artifacts" / "final_pipeline")
    max_folds: int | None = None
    force_rerun: bool = False


RUN_CONFIG = Config()


def _build_set_encoder(
    cfg: Config,
    base_backbone: TinierHAR,
    num_classes: int,
) -> torch.nn.Module:
    se_cfg = replace(DEFAULT_CONFIG.set_encoder, include_global_context=False)
    if cfg.encoder == "attention":
        return AttentionSetEncoder(
            backbone=base_backbone,
            num_classes=num_classes,
            backbone_train_mode=cfg.set_encoder_backbone_train_mode,
            force_conv_bn_eval=cfg.force_conv_bn_eval,
            set_encoder_config=se_cfg,
        )
    return PrototypicalSetEncoder(
        backbone=base_backbone,
        num_classes=num_classes,
        backbone_train_mode=cfg.set_encoder_backbone_train_mode,
        force_conv_bn_eval=cfg.force_conv_bn_eval,
        set_encoder_config=se_cfg,
    )


def _choose_activity_ids(
    loader: Loader,
    indices: Sequence[int],
    needed_per_subject_activity: int,
    min_subjects: int,
) -> list[int]:
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    subset["window_index"] = subset.index.astype(int)
    session_meta = loader.session_df[
        ["session_id", "subject_id", "activity_id"]
    ].drop_duplicates("session_id")
    merged = subset.merge(session_meta, on="session_id", how="left")
    grouped = (
        merged.groupby(["subject_id", "activity_id"])["window_index"]
        .count()
        .reset_index(name="count")
    )
    support: dict[int, dict[int, int]] = {}
    for row in grouped.itertuples(index=False):
        support.setdefault(int(row.subject_id), {})[int(row.activity_id)] = int(
            row.count
        )
    activities = sorted(set(int(x) for x in merged["activity_id"].dropna().tolist()))

    selected: list[int] = []
    for aid in activities:
        eligible_subjects = [
            sid
            for sid, per_act in support.items()
            if per_act.get(aid, 0) >= needed_per_subject_activity
        ]
        if len(eligible_subjects) >= min_subjects:
            selected.append(int(aid))
    if not selected:
        raise ValueError(
            "No activity ids satisfy episodic requirements: "
            f"needed_per_subject_activity={needed_per_subject_activity}, min_subjects={min_subjects}"
        )
    return selected


def _run_meta_eval(
    trainer: Any,
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

    with torch.no_grad():
        iterator = (
            episode_bank
            if episode_bank is not None
            else (trainer._sample_episode() for _ in range(episodes))
        )
        for x_support, y_support, x_query, y_query, _ in iterator:
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
            base_losses.append(float(base_loss.item()))
            base_preds_all.append(base_logits.argmax(dim=-1).reshape(-1).cpu())

            c_subject = trainer.set_encoder(x_support, y_support)
            lora_weights = trainer.hypernet(c_subject)
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
            all_preds.append(logits.argmax(dim=-1).reshape(-1).cpu())
            all_targets.append(targets_flat.cpu())

    preds_t = torch.cat(all_preds) if all_preds else torch.empty((0,), dtype=torch.long)
    targets_t = (
        torch.cat(all_targets) if all_targets else torch.empty((0,), dtype=torch.long)
    )
    macro_f1 = (
        f1_score(targets_t.numpy(), preds_t.numpy(), average="macro", zero_division=0)
        if preds_t.numel() > 0
        else 0.0
    )
    base_preds_t = (
        torch.cat(base_preds_all)
        if base_preds_all
        else torch.empty((0,), dtype=torch.long)
    )
    base_macro_f1 = (
        f1_score(
            targets_t.numpy(), base_preds_t.numpy(), average="macro", zero_division=0
        )
        if base_preds_t.numel() > 0
        else 0.0
    )
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "macro_f1": float(macro_f1),
        "base_loss": sum(base_losses) / max(1, len(base_losses)),
        "base_macro_f1": float(base_macro_f1),
        "macro_f1_improvement": float(macro_f1 - base_macro_f1),
    }


def _build_episode_bank(
    trainer: Any,
    episodes: int,
    support_per_class_choices: Sequence[int] | None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]]:
    if episodes <= 0:
        return []
    if support_per_class_choices is None:
        return [trainer._sample_episode() for _ in range(episodes)]
    choices = [int(k) for k in support_per_class_choices if int(k) > 0]
    out = []
    for episode_idx in range(episodes):
        k = choices[episode_idx % len(choices)]
        out.append(trainer._sample_episode(support_per_class=k))
    return out


def _build_episode_bank_by_k(
    trainer: Any,
    episodes_per_k: int,
    support_per_class_choices: Sequence[int],
) -> dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]]]:
    banks: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]]] = {}
    if episodes_per_k <= 0:
        return {str(int(k)): [] for k in support_per_class_choices}
    for k in support_per_class_choices:
        banks[str(int(k))] = [
            trainer._sample_episode(support_per_class=int(k))
            for _ in range(int(episodes_per_k))
        ]
    return banks


def _mean_eval_metrics(metrics_by_k: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    if not metrics_by_k:
        return {
            "loss": 0.0,
            "macro_f1": 0.0,
            "base_loss": 0.0,
            "base_macro_f1": 0.0,
            "macro_f1_improvement": 0.0,
        }
    keys = ["loss", "macro_f1", "base_loss", "base_macro_f1", "macro_f1_improvement"]
    return {
        key: float(np.mean([float(metrics[key]) for metrics in metrics_by_k.values()]))
        for key in keys
    }


def _run_meta_eval_by_k(
    trainer: Any,
    episodes_per_k: int,
    use_vmap: bool,
    episode_banks_by_k: Mapping[
        str,
        Sequence[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]
        ],
    ],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metrics_by_k: dict[str, dict[str, Any]] = {}
    for k_str, bank in episode_banks_by_k.items():
        metrics_by_k[str(k_str)] = _run_meta_eval(
            trainer,
            episodes=episodes_per_k,
            use_vmap=use_vmap,
            episode_bank=bank,
        )
    return _mean_eval_metrics(metrics_by_k), metrics_by_k


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    output_root = Path(config.output_root)
    stage_dir = output_root / "03_hypernet_meta"
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
        subjects_per_group=config.subjects_per_group,
        seed=config.seed,
    )
    manifest_path = output_root / "shared_splits" / "group4_subject_folds.json"
    folds = build_or_load_loso_folds(session_df, window_df, shared_cfg, manifest_path)
    if config.max_folds is not None:
        folds = folds[: int(config.max_folds)]

    pretrain_root = output_root / "01_pretrain_base"
    set_encoder_root = output_root / "02_set_encoder_supcon"
    summary_rows: list[dict[str, Any]] = []
    skipped_folds: list[str] = []

    for split_idx, fold in enumerate(folds):
        split = split_indices_for_fold(
            session_df,
            window_df,
            type(
                "Tmp",
                (),
                {
                    "train_subject_ids": fold.meta_train_subject_ids,
                    "val_subject_ids": fold.val_subject_ids,
                    "test_subject_ids": fold.test_subject_ids,
                },
            )(),
        )
        split_dir = stage_dir / fold.fold_id
        split_dir.mkdir(parents=True, exist_ok=True)
        fold_fp = config_fingerprint(
            {
                "stage": "03_hypernet_meta",
                "config": asdict(config),
                "shared_cfg": asdict(shared_cfg),
                "fold": asdict(fold),
            }
        )
        metrics_path = split_dir / "metrics.json"
        ckpt_path = split_dir / "best_hypernet.pt"
        if not config.force_rerun and metrics_path.exists() and ckpt_path.exists():
            try:
                existing = json.loads(metrics_path.read_text(encoding="utf-8"))
                if existing.get("config_fingerprint") == fold_fp:
                    print(
                        f"[{fold.fold_id}] skipping (already complete with same settings)"
                    )
                    summary_rows.append(existing)
                    skipped_folds.append(fold.fold_id)
                    continue
            except Exception:
                pass

        base_ckpt = pretrain_root / fold.fold_id / "best_base_model.pt"
        set_ckpt = set_encoder_root / fold.fold_id / "best_set_encoder_supcon.pt"
        if not base_ckpt.exists():
            raise FileNotFoundError(f"Missing base checkpoint: {base_ckpt}")
        if not set_ckpt.exists():
            raise FileNotFoundError(f"Missing set encoder checkpoint: {set_ckpt}")

        post = PostProcessingPipeline(cfg, pre, window_df, split.train_indices)
        samples = post.run()
        loader = Loader(session_df, window_df, post.samples_dir, samples)

        x_np = np.asarray(loader.get_sample(split.train_indices[0])[0])
        if x_np.ndim == 3 and x_np.shape[0] == 1:
            x_np = x_np[0]
        window_size = int(x_np.shape[0])
        num_channels = int(cfg.num_of_channels)
        num_classes = int(cfg.num_of_activities)

        base_model = TinierHAR(
            num_channels=num_channels,
            num_classes=num_classes,
            window_size=window_size,
            backbone_config=DEFAULT_CONFIG.backbone,
        )
        base_model.load_state_dict(torch.load(base_ckpt, map_location=config.device))
        for p in base_model.parameters():
            p.requires_grad = False
        base_model.eval()

        se_backbone = TinierHAR(
            num_channels=num_channels,
            num_classes=num_classes,
            window_size=window_size,
            backbone_config=DEFAULT_CONFIG.backbone,
        )
        set_encoder = _build_set_encoder(config, se_backbone, num_classes)
        se_payload = torch.load(
            set_ckpt, map_location=config.device, weights_only=False
        )
        set_encoder.load_state_dict(se_payload["set_encoder"])
        set_encoder_frozen = True
        for p in set_encoder.parameters():
            p.requires_grad = False
        set_encoder.eval()

        if config.lora_rank < 2 or config.lora_rank > 16:
            raise ValueError(
                f"lora_rank={config.lora_rank} is outside recommended range [2,16]."
            )
        if config.lora_alpha < float(config.lora_rank) or config.lora_alpha > float(
            2 * config.lora_rank
        ):
            raise ValueError(
                f"lora_alpha={config.lora_alpha} must be in [rank, 2*rank]; "
                f"rank={config.lora_rank}."
            )

        hypernet = HyperNet(
            num_channels=num_channels,
            num_classes=num_classes,
            set_encoder_output_dim=getattr(set_encoder, "output_dim", None),
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            enable_conv1_adapter=config.enable_conv1_adapter,
            enable_conv_last_adapter=config.enable_conv_last_adapter,
            set_encoder_config=DEFAULT_CONFIG.set_encoder,
            backbone_config=DEFAULT_CONFIG.backbone,
            hypernet_config=DEFAULT_CONFIG.hypernet,
        )

        train_support_choices = k_choices_from_range(
            config.train_min_k_per_class, config.train_max_k_per_class
        )
        eval_support_choices = train_support_choices
        train_needed = max(train_support_choices) + config.query_per_class
        eval_needed = max(eval_support_choices) + config.eval_query_per_class
        train_activity_ids = _choose_activity_ids(
            loader, split.train_indices, train_needed, config.train_subjects_per_episode
        )
        val_activity_ids = _choose_activity_ids(
            loader, split.val_indices, eval_needed, 1
        )
        test_activity_ids = _choose_activity_ids(
            loader, split.test_indices, eval_needed, 1
        )

        train_meta_cfg = MetaTrainerConfig(
            learning_rate=config.meta_learning_rate,
            weight_decay=config.weight_decay,
            adapter_delta_l2=config.adapter_delta_l2,
            batch_subjects=config.train_subjects_per_episode,
            support_per_class=max(train_support_choices),
            support_per_class_choices=train_support_choices,
            query_per_class=config.query_per_class,
            use_vmap=config.use_vmap,
            seed=split_idx,
            device=config.device,
        )
        val_meta_cfg = MetaTrainerConfig(
            learning_rate=config.meta_learning_rate,
            weight_decay=config.weight_decay,
            adapter_delta_l2=config.adapter_delta_l2,
            batch_subjects=1,
            support_per_class=max(eval_support_choices),
            support_per_class_choices=eval_support_choices,
            query_per_class=config.eval_query_per_class,
            use_vmap=config.use_vmap,
            seed=10_000 + split_idx,
            device=config.device,
        )
        test_meta_cfg = MetaTrainerConfig(
            learning_rate=config.meta_learning_rate,
            weight_decay=config.weight_decay,
            adapter_delta_l2=config.adapter_delta_l2,
            batch_subjects=1,
            support_per_class=max(eval_support_choices),
            support_per_class_choices=eval_support_choices,
            query_per_class=config.eval_query_per_class,
            use_vmap=config.use_vmap,
            seed=20_000 + split_idx,
            device=config.device,
        )

        optimizer = torch.optim.AdamW(
            list(hypernet.parameters()),
            lr=config.meta_learning_rate,
            weight_decay=config.weight_decay,
        )
        steps_per_epoch = int(config.train_episodes_per_epoch)
        total_steps = max(1, int(config.epochs) * steps_per_epoch)
        warmup_steps = max(1, int(total_steps * float(config.warmup_ratio)))
        decay_steps = max(1, total_steps - warmup_steps)
        warmup = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=float(config.min_learning_rate),
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

        train_trainer = SetToLoRAMetaTrainer(
            base_model=base_model,
            set_encoder=set_encoder,
            hypernet=hypernet,
            loader=loader,
            num_classes=num_classes,
            config=train_meta_cfg,
            optimizer=optimizer,
            class_weights=None,
            indices=split.train_indices,
            activity_ids=train_activity_ids,
        )
        val_trainer = SetToLoRAMetaTrainer(
            base_model=base_model,
            set_encoder=set_encoder,
            hypernet=hypernet,
            loader=loader,
            num_classes=num_classes,
            config=val_meta_cfg,
            optimizer=optimizer,
            class_weights=None,
            indices=split.val_indices,
            activity_ids=val_activity_ids,
        )
        test_trainer = SetToLoRAMetaTrainer(
            base_model=base_model,
            set_encoder=set_encoder,
            hypernet=hypernet,
            loader=loader,
            num_classes=num_classes,
            config=test_meta_cfg,
            optimizer=optimizer,
            class_weights=None,
            indices=split.test_indices,
            activity_ids=test_activity_ids,
        )

        val_episode_banks_by_k = _build_episode_bank_by_k(
            val_trainer, config.eval_episodes, eval_support_choices
        )
        test_episode_banks_by_k = _build_episode_bank_by_k(
            test_trainer, config.eval_episodes, eval_support_choices
        )

        best_val_improvement = float("-inf")
        best_val_loss = float("inf")
        best_epoch = -1
        patience_counter = 0
        ckpt_path = split_dir / "best_hypernet.pt"
        history_rows: list[dict[str, float | int]] = []
        global_step = 0
        for epoch in range(1, config.epochs + 1):
            if (
                set_encoder_frozen
                and config.unfreeze_set_encoder_after_epoch is not None
                and epoch > int(config.unfreeze_set_encoder_after_epoch)
            ):
                for p in set_encoder.parameters():
                    p.requires_grad = True
                optimizer.add_param_group(
                    {
                        "params": [
                            p for p in set_encoder.parameters() if p.requires_grad
                        ],
                        "lr": float(config.set_encoder_learning_rate),
                        "weight_decay": float(config.weight_decay),
                    }
                )
                set_encoder.train()
                set_encoder_frozen = False
                print(
                    f"[{fold.fold_id}] unfroze set encoder at epoch {epoch} "
                    f"(set_encoder_lr={config.set_encoder_learning_rate:.2e})"
                )

            step_losses: list[float] = []
            step_f1: list[float] = []
            for _ in tqdm(
                range(config.train_episodes_per_epoch),
                desc=f"{fold.fold_id} meta {epoch}/{config.epochs}",
                leave=False,
            ):
                metrics = train_trainer.train_step(use_vmap=config.use_vmap)
                step_losses.append(float(metrics["loss"]))
                step_f1.append(float(metrics["macro_f1"]))
                scheduler.step()
                global_step += 1

            val_metrics, val_metrics_by_k = _run_meta_eval_by_k(
                val_trainer,
                use_vmap=config.use_vmap,
                episodes_per_k=config.eval_episodes,
                episode_banks_by_k=val_episode_banks_by_k,
            )
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(step_losses)),
                "train_macro_f1": float(np.mean(step_f1)),
                "val_loss": float(val_metrics["loss"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_base_macro_f1": float(val_metrics["base_macro_f1"]),
                "val_macro_f1_improvement": float(val_metrics["macro_f1_improvement"]),
                "val_macro_f1_improvement_by_k": {
                    k: float(metrics["macro_f1_improvement"])
                    for k, metrics in val_metrics_by_k.items()
                },
                "set_encoder_trainable": int(not set_encoder_frozen),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "global_step": int(global_step),
            }
            history_rows.append(row)
            print(
                f"[{fold.fold_id}] epoch={epoch} train_loss={row['train_loss']:.4f} "
                f"train_f1={row['train_macro_f1']:.4f} "
                f"val_improvement={row['val_macro_f1_improvement']:+.4f} "
                f"val_f1={row['val_macro_f1']:.4f} base_val_f1={row['val_base_macro_f1']:.4f}"
            )

            improved = row["val_macro_f1_improvement"] > best_val_improvement or (
                np.isclose(row["val_macro_f1_improvement"], best_val_improvement)
                and row["val_loss"] < best_val_loss
            )
            if improved:
                best_val_improvement = float(row["val_macro_f1_improvement"])
                best_val_loss = float(row["val_loss"])
                best_epoch = int(epoch)
                patience_counter = 0
                torch.save(
                    {
                        "hypernet": hypernet.state_dict(),
                        "set_encoder": set_encoder.state_dict(),
                        "best_epoch": best_epoch,
                        "best_val_improvement": best_val_improvement,
                        "best_val_loss": best_val_loss,
                    },
                    ckpt_path,
                )
            else:
                patience_counter += 1
            if patience_counter >= config.patience:
                break

        if not ckpt_path.exists():
            torch.save(
                {
                    "hypernet": hypernet.state_dict(),
                    "set_encoder": set_encoder.state_dict(),
                },
                ckpt_path,
            )
        best_payload = torch.load(ckpt_path, map_location=config.device)
        hypernet.load_state_dict(best_payload["hypernet"])
        if "set_encoder" in best_payload:
            set_encoder.load_state_dict(best_payload["set_encoder"])

        val_final, val_final_by_k = _run_meta_eval_by_k(
            val_trainer,
            use_vmap=config.use_vmap,
            episodes_per_k=config.eval_episodes,
            episode_banks_by_k=val_episode_banks_by_k,
        )
        test_final, test_final_by_k = _run_meta_eval_by_k(
            test_trainer,
            use_vmap=config.use_vmap,
            episodes_per_k=config.eval_episodes,
            episode_banks_by_k=test_episode_banks_by_k,
        )
        fold_result = {
            "config_fingerprint": fold_fp,
            "fold_id": fold.fold_id,
            "base_train_subject_ids": fold.base_train_subject_ids,
            "meta_train_subject_ids": fold.meta_train_subject_ids,
            "val_subject_ids": fold.val_subject_ids,
            "test_subject_ids": fold.test_subject_ids,
            "best_epoch": int(best_epoch),
            "best_val_macro_f1_improvement": float(best_val_improvement),
            "val_loss": float(val_final["loss"]),
            "val_macro_f1": float(val_final["macro_f1"]),
            "val_base_macro_f1": float(val_final["base_macro_f1"]),
            "val_macro_f1_improvement": float(val_final["macro_f1_improvement"]),
            "test_loss": float(test_final["loss"]),
            "test_macro_f1": float(test_final["macro_f1"]),
            "test_base_macro_f1": float(test_final["base_macro_f1"]),
            "test_macro_f1_improvement": float(test_final["macro_f1_improvement"]),
            "eval_k_values": [int(k) for k in eval_support_choices],
            "eval_episodes_per_k": int(config.eval_episodes),
            "val_by_k": val_final_by_k,
            "test_by_k": test_final_by_k,
        }
        (split_dir / "metrics.json").write_text(
            json.dumps(fold_result, indent=2), encoding="utf-8"
        )
        (split_dir / "history.json").write_text(
            json.dumps(history_rows, indent=2), encoding="utf-8"
        )
        summary_rows.append(fold_result)
        print(
            f"[{fold.fold_id}] val improvement={fold_result['val_macro_f1_improvement']:+.4f} "
            f"test improvement={fold_result['test_macro_f1_improvement']:+.4f}"
        )

    summary = {
        "config": asdict(config),
        "splits_manifest_path": str(manifest_path),
        "num_folds": len(summary_rows),
        "skipped_folds": skipped_folds,
        "subject_splits": [
            {
                "fold_id": r["fold_id"],
                "base_train_subject_ids": r["base_train_subject_ids"],
                "meta_train_subject_ids": r["meta_train_subject_ids"],
                "val_subject_ids": r["val_subject_ids"],
                "test_subject_ids": r["test_subject_ids"],
            }
            for r in summary_rows
        ],
        "mean_val_macro_f1_improvement": float(
            sum(r["val_macro_f1_improvement"] for r in summary_rows)
            / max(1, len(summary_rows))
        ),
        "mean_test_macro_f1_improvement": float(
            sum(r["test_macro_f1_improvement"] for r in summary_rows)
            / max(1, len(summary_rows))
        ),
        "folds": summary_rows,
    }
    (stage_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

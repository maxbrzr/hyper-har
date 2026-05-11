from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

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

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hyper_har.backbone.conditioned_tinierhar import ConditionedTinierHAR
from hyper_har.backbone.tinierhar import TinierHAR
from hyper_har.config import DEFAULT_CONFIG
from hyper_har.training.conditioned_meta_trainer import (
    ConditionedMetaTrainerConfig,
    SubjectConditionedMetaTrainer,
)


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SET_ENCODER_ATTENTION_MODULE = _load_module_from_path(
    "final_phase3_conditioned_attention_set_encoder",
    SRC / "hyper_har" / "set-encoder" / "attention.py",
)
SET_ENCODER_SIMPLE_MODULE = _load_module_from_path(
    "final_phase3_conditioned_simple_set_encoder",
    SRC / "hyper_har" / "set-encoder" / "simple.py",
)
AttentionSetEncoder = SET_ENCODER_ATTENTION_MODULE.AttentionSetEncoder
PrototypicalSetEncoder = SET_ENCODER_SIMPLE_MODULE.PrototypicalSetEncoder


@dataclass(frozen=True)
class Config:
    dataset_id: str = WHARDatasetID.WEAR.value
    datasets_dir: str = str(ROOT / "datasets")
    selected_activities: list[str] | None = None
    window_overlap: float = 0.0
    subjects_per_group: int = 6
    seed: int = 0

    encoder: str = "attention"
    set_encoder_backbone_train_mode: str = "freeze_all"
    force_conv_bn_eval: bool = True

    fusion_mode: str = "temporal_tiling"
    subject_condition_dim: int = 64
    freeze_conv_blocks: bool = True
    train_subject_projector: bool = True
    train_gru: bool = True
    train_attention: bool = True
    train_classifier: bool = True

    train_subjects_per_episode: int = 4
    train_min_k_per_class: int = DEFAULT_TRAIN_MIN_K_PER_CLASS
    train_max_k_per_class: int = DEFAULT_TRAIN_MAX_K_PER_CLASS
    query_per_class: int = 8
    eval_min_k_per_class: int = DEFAULT_TRAIN_MIN_K_PER_CLASS
    eval_max_k_per_class: int = DEFAULT_TRAIN_MAX_K_PER_CLASS
    eval_query_per_class: int = 16
    train_episodes_per_epoch: int = 64
    eval_episodes: int = 64

    meta_learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    epochs: int = 100
    patience: int = 10
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
            f"needed_per_subject_activity={needed_per_subject_activity}, "
            f"min_subjects={min_subjects}"
        )
    return selected


@torch.no_grad()
def _run_meta_eval(
    trainer: SubjectConditionedMetaTrainer,
    episodes: int,
    episode_bank: Sequence[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]
    ]
    | None = None,
) -> dict[str, Any]:
    trainer.baseline_model.eval()
    trainer.set_encoder.eval()
    trainer.conditioned_model.eval()

    losses: list[float] = []
    base_losses: list[float] = []
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    base_preds_all: list[torch.Tensor] = []

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
        base_logits = trainer.baseline_model(
            x_query.reshape(-1, *x_query.shape[2:])
        ).reshape(x_query.size(0), x_query.size(1), -1)
        base_logits_flat = base_logits.reshape(-1, base_logits.size(-1))
        base_loss = F.cross_entropy(
            base_logits_flat, targets_flat, weight=trainer.class_weights
        )
        base_losses.append(float(base_loss.item()))
        base_preds_all.append(base_logits.argmax(dim=-1).reshape(-1).cpu())

        c_subject = trainer.set_encoder(x_support, y_support)
        logits = trainer.conditioned_model.forward_episode(x_query, c_subject)
        logits_flat = logits.reshape(-1, logits.size(-1))
        loss = F.cross_entropy(logits_flat, targets_flat, weight=trainer.class_weights)
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
    trainer: SubjectConditionedMetaTrainer,
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
        k_shot = choices[episode_idx % len(choices)]
        out.append(trainer._sample_episode(support_per_class=k_shot))
    return out


def _load_base_model(
    ckpt_path: Path,
    num_channels: int,
    num_classes: int,
    window_size: int,
    device: str,
) -> TinierHAR:
    model = TinierHAR(
        num_channels=num_channels,
        num_classes=num_classes,
        window_size=window_size,
        backbone_config=DEFAULT_CONFIG.backbone,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    output_root = Path(config.output_root)
    stage_dir = output_root / "03_conditioned_meta"
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
                "stage": "03_conditioned_meta",
                "config": asdict(config),
                "shared_cfg": asdict(shared_cfg),
                "fold": asdict(fold),
            }
        )
        metrics_path = split_dir / "metrics.json"
        ckpt_path = split_dir / "best_conditioned_tinierhar.pt"
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

        baseline_model = _load_base_model(
            base_ckpt, num_channels, num_classes, window_size, config.device
        )
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
        for param in set_encoder.parameters():
            param.requires_grad = False
        set_encoder.eval()

        subject_embedding_dim = int(getattr(set_encoder, "output_dim"))
        conditioned_model = ConditionedTinierHAR(
            num_channels=num_channels,
            num_classes=num_classes,
            window_size=window_size,
            subject_embedding_dim=subject_embedding_dim,
            condition_dim=config.subject_condition_dim,
            fusion_mode=config.fusion_mode,  # type: ignore[arg-type]
            backbone_config=DEFAULT_CONFIG.backbone,
        )
        conditioned_model.load_tinierhar_state_dict(
            torch.load(base_ckpt, map_location=config.device)
        )
        conditioned_model.configure_for_meta_training(
            freeze_conv_blocks=config.freeze_conv_blocks,
            train_subject_projector=config.train_subject_projector,
            train_gru=config.train_gru,
            train_attention=config.train_attention,
            train_classifier=config.train_classifier,
        )

        trainable_params = [
            param for param in conditioned_model.parameters() if param.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_params,
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

        train_support_choices = k_choices_from_range(
            config.train_min_k_per_class, config.train_max_k_per_class
        )
        eval_support_choices = k_choices_from_range(
            config.eval_min_k_per_class, config.eval_max_k_per_class
        )
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

        train_meta_cfg = ConditionedMetaTrainerConfig(
            learning_rate=config.meta_learning_rate,
            weight_decay=config.weight_decay,
            batch_subjects=config.train_subjects_per_episode,
            support_per_class=max(train_support_choices),
            support_per_class_choices=train_support_choices,
            query_per_class=config.query_per_class,
            seed=split_idx,
            device=config.device,
        )
        val_meta_cfg = ConditionedMetaTrainerConfig(
            learning_rate=config.meta_learning_rate,
            weight_decay=config.weight_decay,
            batch_subjects=1,
            support_per_class=max(eval_support_choices),
            support_per_class_choices=eval_support_choices,
            query_per_class=config.eval_query_per_class,
            seed=10_000 + split_idx,
            device=config.device,
        )
        test_meta_cfg = ConditionedMetaTrainerConfig(
            learning_rate=config.meta_learning_rate,
            weight_decay=config.weight_decay,
            batch_subjects=1,
            support_per_class=max(eval_support_choices),
            support_per_class_choices=eval_support_choices,
            query_per_class=config.eval_query_per_class,
            seed=20_000 + split_idx,
            device=config.device,
        )

        train_trainer = SubjectConditionedMetaTrainer(
            conditioned_model=conditioned_model,
            set_encoder=set_encoder,
            baseline_model=baseline_model,
            loader=loader,
            num_classes=num_classes,
            config=train_meta_cfg,
            optimizer=optimizer,
            class_weights=None,
            indices=split.train_indices,
            activity_ids=train_activity_ids,
            freeze_set_encoder=True,
        )
        val_trainer = SubjectConditionedMetaTrainer(
            conditioned_model=conditioned_model,
            set_encoder=set_encoder,
            baseline_model=baseline_model,
            loader=loader,
            num_classes=num_classes,
            config=val_meta_cfg,
            optimizer=optimizer,
            class_weights=None,
            indices=split.val_indices,
            activity_ids=val_activity_ids,
            freeze_set_encoder=True,
        )
        test_trainer = SubjectConditionedMetaTrainer(
            conditioned_model=conditioned_model,
            set_encoder=set_encoder,
            baseline_model=baseline_model,
            loader=loader,
            num_classes=num_classes,
            config=test_meta_cfg,
            optimizer=optimizer,
            class_weights=None,
            indices=split.test_indices,
            activity_ids=test_activity_ids,
            freeze_set_encoder=True,
        )

        val_episode_bank = _build_episode_bank(
            val_trainer, config.eval_episodes, eval_support_choices
        )
        test_episode_bank = _build_episode_bank(
            test_trainer, config.eval_episodes, eval_support_choices
        )

        best_val_improvement = float("-inf")
        best_val_loss = float("inf")
        best_epoch = -1
        patience_counter = 0
        history_rows: list[dict[str, float | int]] = []
        global_step = 0

        for epoch in range(1, config.epochs + 1):
            step_losses: list[float] = []
            step_f1: list[float] = []
            for _ in tqdm(
                range(config.train_episodes_per_epoch),
                desc=f"{fold.fold_id} conditioned meta {epoch}/{config.epochs}",
                leave=False,
            ):
                metrics = train_trainer.train_step()
                step_losses.append(float(metrics["loss"]))
                step_f1.append(float(metrics["macro_f1"]))
                scheduler.step()
                global_step += 1

            val_metrics = _run_meta_eval(
                val_trainer,
                episodes=config.eval_episodes,
                episode_bank=val_episode_bank,
            )
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(step_losses)),
                "train_macro_f1": float(np.mean(step_f1)),
                "val_loss": float(val_metrics["loss"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_base_macro_f1": float(val_metrics["base_macro_f1"]),
                "val_macro_f1_improvement": float(val_metrics["macro_f1_improvement"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "global_step": int(global_step),
            }
            history_rows.append(row)
            print(
                f"[{fold.fold_id}] epoch={epoch} train_loss={row['train_loss']:.4f} "
                f"train_f1={row['train_macro_f1']:.4f} "
                f"val_improvement={row['val_macro_f1_improvement']:+.4f} "
                f"val_f1={row['val_macro_f1']:.4f} "
                f"base_val_f1={row['val_base_macro_f1']:.4f}"
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
                        "conditioned_model": conditioned_model.state_dict(),
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
                    "conditioned_model": conditioned_model.state_dict(),
                    "set_encoder": set_encoder.state_dict(),
                },
                ckpt_path,
            )
        best_payload = torch.load(
            ckpt_path, map_location=config.device, weights_only=False
        )
        conditioned_model.load_state_dict(best_payload["conditioned_model"])
        if "set_encoder" in best_payload:
            set_encoder.load_state_dict(best_payload["set_encoder"])

        val_final = _run_meta_eval(
            val_trainer,
            episodes=config.eval_episodes,
            episode_bank=val_episode_bank,
        )
        test_final = _run_meta_eval(
            test_trainer,
            episodes=config.eval_episodes,
            episode_bank=test_episode_bank,
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

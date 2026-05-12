from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm
from whar_datasets import (
    Loader,
    PostProcessingPipeline,
    PreProcessingPipeline,
    WHARDatasetID,
)

from hyper_har.backbone.film_tinierhar import FiLMTinierHAR
from hyper_har.backbone.tinierhar import TinierHAR
from hyper_har.config import DEFAULT_CONFIG
from hyper_har.training.conditioned_meta_trainer import (
    ConditionedMetaTrainerConfig,
    SubjectConditionedMetaTrainer,
)

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


CONDITIONED_HELPERS = _load_module_from_path(
    "final_phase3_conditioned_helpers",
    THIS_DIR / "03_train_conditioned_meta_loso.py",
)


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

    film_hidden_dim: int = 128
    film_dropout: float = 0.0  # 0.1  # 0.0
    film_use_explosion_guard: bool = True  # False  # False
    film_gamma_bound: float = 0.5
    film_beta_bound: float = 1.0
    film_enable_conv1: bool = False  # True  # False
    film_modulation_mode: str = "dynamic_time"  # "static"
    film_condition_gru_h0: bool = True  # False
    film_stage_name: str | None = None

    train_subjects_per_episode: int = 4
    train_min_k_per_class: int = DEFAULT_TRAIN_MIN_K_PER_CLASS
    train_max_k_per_class: int = DEFAULT_TRAIN_MAX_K_PER_CLASS
    query_per_class: int = 8
    eval_query_per_class: int = 16
    train_episodes_per_epoch: int = 64
    eval_episodes: int = 32

    meta_learning_rate: float = 1e-5
    min_learning_rate: float = 1e-6
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    epochs: int = 100
    patience: int = 20  # 10  # 10
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    output_root: str = str(ROOT / "artifacts" / "final_pipeline")
    max_folds: int | None = None
    force_rerun: bool = False


RUN_CONFIG = Config()


def _path_float(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _resolve_stage_name(config: Config) -> str:
    if config.film_stage_name is not None:
        return str(config.film_stage_name)

    stage_name = (
        "03_film_meta_guarded" if config.film_use_explosion_guard else "03_film_meta"
    )
    suffix_parts: list[str] = []
    if int(config.film_hidden_dim) != 128:
        suffix_parts.append(f"hidden-{int(config.film_hidden_dim)}")
    if config.film_modulation_mode.strip().lower() != "static":
        suffix_parts.append(
            config.film_modulation_mode.strip().lower().replace("_", "-")
        )
    if config.film_enable_conv1:
        suffix_parts.append("conv1")
    if config.film_condition_gru_h0:
        suffix_parts.append("gru-h0")
    if not np.isclose(float(config.film_dropout), 0.0):
        suffix_parts.append(f"dropout-{_path_float(float(config.film_dropout))}")
    if config.film_use_explosion_guard:
        if not np.isclose(float(config.film_gamma_bound), 0.5):
            suffix_parts.append(f"gamma-{_path_float(float(config.film_gamma_bound))}")
        if not np.isclose(float(config.film_beta_bound), 1.0):
            suffix_parts.append(f"beta-{_path_float(float(config.film_beta_bound))}")

    if suffix_parts:
        stage_name = f"{stage_name}__{'__'.join(suffix_parts)}"
    return stage_name


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
    stage_name = _resolve_stage_name(config)
    stage_dir = output_root / stage_name
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
                "stage": "03_film_meta",
                "stage_dir": stage_name,
                "config": asdict(config),
                "shared_cfg": asdict(shared_cfg),
                "fold": asdict(fold),
            }
        )
        metrics_path = split_dir / "metrics.json"
        ckpt_path = split_dir / "best_film_tinierhar.pt"
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
        film_base_model = _load_base_model(
            base_ckpt, num_channels, num_classes, window_size, config.device
        )
        se_backbone = TinierHAR(
            num_channels=num_channels,
            num_classes=num_classes,
            window_size=window_size,
            backbone_config=DEFAULT_CONFIG.backbone,
        )
        set_encoder = CONDITIONED_HELPERS._build_set_encoder(
            config, se_backbone, num_classes
        )
        se_payload = torch.load(
            set_ckpt, map_location=config.device, weights_only=False
        )
        set_encoder.load_state_dict(se_payload["set_encoder"])
        for param in set_encoder.parameters():
            param.requires_grad = False
        set_encoder.eval()

        subject_embedding_dim = int(getattr(set_encoder, "output_dim"))
        film_model = FiLMTinierHAR(
            base_model=film_base_model,
            subject_embedding_dim=subject_embedding_dim,
            film_hidden_dim=config.film_hidden_dim,
            film_dropout=config.film_dropout,
            film_use_explosion_guard=config.film_use_explosion_guard,
            film_gamma_bound=config.film_gamma_bound,
            film_beta_bound=config.film_beta_bound,
            film_enable_conv1=config.film_enable_conv1,
            film_modulation_mode=config.film_modulation_mode,
            film_condition_gru_h0=config.film_condition_gru_h0,
        )
        trainable_params = [
            param for param in film_model.parameters() if param.requires_grad
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
        eval_support_choices = train_support_choices
        train_time_eval_support_choices = (max(train_support_choices),)
        train_needed = max(train_support_choices) + config.query_per_class
        eval_needed = max(eval_support_choices) + config.eval_query_per_class
        train_activity_ids = CONDITIONED_HELPERS._choose_activity_ids(
            loader, split.train_indices, train_needed, config.train_subjects_per_episode
        )
        val_activity_ids = CONDITIONED_HELPERS._choose_activity_ids(
            loader, split.val_indices, eval_needed, 1
        )
        test_activity_ids = CONDITIONED_HELPERS._choose_activity_ids(
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
            conditioned_model=film_model,
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
            conditioned_model=film_model,
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
            conditioned_model=film_model,
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

        train_time_val_episode_banks_by_k = (
            CONDITIONED_HELPERS._build_episode_bank_by_k(
                val_trainer, config.eval_episodes, train_time_eval_support_choices
            )
        )
        final_val_episode_banks_by_k = CONDITIONED_HELPERS._build_episode_bank_by_k(
            val_trainer, config.eval_episodes, eval_support_choices
        )
        final_test_episode_banks_by_k = CONDITIONED_HELPERS._build_episode_bank_by_k(
            test_trainer, config.eval_episodes, eval_support_choices
        )

        best_val_improvement = float("-inf")
        best_val_loss = float("inf")
        best_epoch = -1
        patience_counter = 0
        history_rows: list[dict] = []
        global_step = 0

        for epoch in range(1, config.epochs + 1):
            step_losses: list[float] = []
            step_f1: list[float] = []
            for _ in tqdm(
                range(config.train_episodes_per_epoch),
                desc=f"{fold.fold_id} FiLM meta {epoch}/{config.epochs}",
                leave=False,
            ):
                metrics = train_trainer.train_step()
                step_losses.append(float(metrics["loss"]))
                step_f1.append(float(metrics["macro_f1"]))
                scheduler.step()
                global_step += 1

            val_metrics, val_metrics_by_k = CONDITIONED_HELPERS._run_meta_eval_by_k(
                val_trainer,
                episodes_per_k=config.eval_episodes,
                episode_banks_by_k=train_time_val_episode_banks_by_k,
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
                        "film_model": film_model.state_dict(),
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
                    "film_model": film_model.state_dict(),
                    "set_encoder": set_encoder.state_dict(),
                },
                ckpt_path,
            )
        best_payload = torch.load(
            ckpt_path, map_location=config.device, weights_only=False
        )
        film_model.load_state_dict(best_payload["film_model"])
        if "set_encoder" in best_payload:
            set_encoder.load_state_dict(best_payload["set_encoder"])

        val_final, val_final_by_k = CONDITIONED_HELPERS._run_meta_eval_by_k(
            val_trainer,
            episodes_per_k=config.eval_episodes,
            episode_banks_by_k=final_val_episode_banks_by_k,
        )
        test_final, test_final_by_k = CONDITIONED_HELPERS._run_meta_eval_by_k(
            test_trainer,
            episodes_per_k=config.eval_episodes,
            episode_banks_by_k=final_test_episode_banks_by_k,
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
            "train_time_eval_k_values": [
                int(k) for k in train_time_eval_support_choices
            ],
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

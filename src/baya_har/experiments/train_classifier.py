import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from whar_datasets import PreProcessingPipeline, WHARDatasetID

from baya_har.training.trainer import TinierHARTrainer, TrainerConfig

from .common import (
    DEFAULT_DATASET_ID,
    DEFAULT_DATASETS_DIR,
    DEFAULT_SEED,
    DEFAULT_SELECTED_ACTIVITIES,
    DEFAULT_SPLIT_STRATEGY,
    DEFAULT_TEST_SUBJECTS,
    DEFAULT_VAL_PERCENTAGE,
    DEFAULT_VAL_SUBJECTS,
    SharedConfig,
    WindowDataset,
    build_loader,
    build_or_load_loso_folds,
    build_tinierhar,
    class_names,
    config_fingerprint,
    infer_window_size,
    prepare_cfg,
    reconcile_activity_config,
    resolve_output_root,
    set_seed,
    split_indices_for_fold,
)


@dataclass(frozen=True)
class Config:
    dataset_id: str = DEFAULT_DATASET_ID
    datasets_dir: str = DEFAULT_DATASETS_DIR
    selected_activities: list[str] | None = DEFAULT_SELECTED_ACTIVITIES
    window_overlap: float = 0.5
    val_subjects: int = DEFAULT_VAL_SUBJECTS
    test_subjects: int = DEFAULT_TEST_SUBJECTS
    seed: int = DEFAULT_SEED
    split_strategy: str = DEFAULT_SPLIT_STRATEGY
    val_percentage: float = DEFAULT_VAL_PERCENTAGE

    batch_size: int = 64
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 100
    patience: int = 10
    min_delta: float = 0.0
    early_stopping_metric: str = "val_macro_f1"
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_root: str | None = None
    stage_name: str = "classifier"
    max_folds: int | None = None
    force_rerun: bool = False


RUN_CONFIG = Config()


class TrainerWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, base: WindowDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        sample = self.base[item]
        return sample["x"], sample["y"]


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    output_root = resolve_output_root(config.output_root, config.dataset_id)
    stage_dir = output_root / config.stage_name
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
    reconcile_activity_config(cfg, session_df)

    shared_cfg = SharedConfig(
        dataset_id=config.dataset_id,
        datasets_dir=config.datasets_dir,
        selected_activities=config.selected_activities,
        window_overlap=config.window_overlap,
        val_subjects=config.val_subjects,
        test_subjects=config.test_subjects,
        seed=config.seed,
        split_strategy=config.split_strategy,
        val_percentage=config.val_percentage,
    )
    manifest_path = output_root / "shared_splits" / "loso_subject_folds.json"
    folds = build_or_load_loso_folds(session_df, window_df, shared_cfg, manifest_path)
    if config.max_folds is not None:
        folds = folds[: int(config.max_folds)]

    summary_rows: list[dict[str, Any]] = []
    skipped_folds: list[str] = []
    for fold in folds:
        split = split_indices_for_fold(session_df, window_df, fold)
        split_dir = stage_dir / fold.fold_id
        split_dir.mkdir(parents=True, exist_ok=True)
        fold_fp = config_fingerprint(
            {
                "stage": config.stage_name,
                "config": asdict(config),
                "shared_cfg": asdict(shared_cfg),
                "fold": asdict(fold),
                "num_classes": int(cfg.num_of_activities),
                "class_names": class_names(cfg),
            }
        )
        metrics_path = split_dir / "metrics.json"
        ckpt_path = split_dir / "best_model.pt"
        if not config.force_rerun and metrics_path.exists() and ckpt_path.exists():
            try:
                existing = json.loads(metrics_path.read_text(encoding="utf-8"))
                if existing.get("config_fingerprint") == fold_fp:
                    print(f"[{fold.fold_id}] skipping (already complete)")
                    summary_rows.append(existing)
                    skipped_folds.append(fold.fold_id)
                    continue
            except Exception:
                pass

        loader = build_loader(cfg, session_df, pre, window_df, split.train_indices)
        train_ds = WindowDataset(loader, split.train_indices)
        val_ds = WindowDataset(loader, split.val_indices)
        test_ds = WindowDataset(loader, split.test_indices)
        train_loader = DataLoader(
            TrainerWindowDataset(train_ds),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        val_loader = DataLoader(
            TrainerWindowDataset(val_ds),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        test_loader = DataLoader(
            TrainerWindowDataset(test_ds),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        window_size = infer_window_size(loader, split.train_indices)
        model = build_tinierhar(
            num_channels=int(cfg.num_of_channels),
            num_classes=int(cfg.num_of_activities),
            window_size=window_size,
        )
        trainer_cfg = TrainerConfig(
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            patience=config.patience,
            min_delta=config.min_delta,
            device=config.device,
            checkpoint_path=str(ckpt_path),
            early_stopping_metric=config.early_stopping_metric,
        )
        trainer = TinierHARTrainer(
            model=model,
            num_classes=int(cfg.num_of_activities),
            config=trainer_cfg,
        )
        history = trainer.fit(train_loader, val_loader)
        test_metrics = trainer.evaluate(test_loader, desc=f"{fold.fold_id} CE test")

        result = {
            "config_fingerprint": fold_fp,
            "fold_id": fold.fold_id,
            "train_subject_ids": fold.train_subject_ids,
            "val_subject_ids": fold.val_subject_ids,
            "test_subject_ids": fold.test_subject_ids,
            "window_size": int(window_size),
            "num_channels": int(cfg.num_of_channels),
            "num_classes": int(cfg.num_of_activities),
            "class_names": class_names(cfg),
            "test_loss": float(test_metrics["loss"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "best_val_loss": float(trainer.state.best_val_loss),
            "best_val_macro_f1": float(trainer.state.best_val_macro_f1),
            "best_epoch": int(trainer.state.best_epoch),
        }
        metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        (split_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "model_meta": {
                    "window_size": int(window_size),
                    "num_channels": int(cfg.num_of_channels),
                    "num_classes": int(cfg.num_of_activities),
                    "class_names": class_names(cfg),
                },
                "fold": asdict(fold),
            },
            split_dir / "best_model_with_meta.pt",
        )
        summary_rows.append(result)
        print(
            f"[{fold.fold_id}] CE test_macro_f1={result['test_macro_f1']:.4f} "
            f"test_loss={result['test_loss']:.4f}"
        )

    mean_f1 = sum(r["test_macro_f1"] for r in summary_rows) / max(1, len(summary_rows))
    summary = {
        "config": asdict(config),
        "splits_manifest_path": str(manifest_path),
        "num_folds": len(summary_rows),
        "skipped_folds": skipped_folds,
        "mean_test_macro_f1": float(mean_f1),
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

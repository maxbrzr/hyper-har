from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from whar_datasets import (
    Loader,
    PostProcessingPipeline,
    PreProcessingPipeline,
    TorchAdapter,
    WHARDatasetID,
)

from hyper_har.backbone.tinierhar import TinierHAR
from hyper_har.config import DEFAULT_CONFIG
from hyper_har.training.trainer import TinierHARTrainer, TrainerConfig

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (
    ROOT,
    SharedConfig,
    build_or_load_loso_folds,
    config_fingerprint,
    prepare_cfg,
    set_seed,
    split_indices_for_fold,
)


@dataclass(frozen=True)
class Config:
    dataset_id: str = WHARDatasetID.WEAR.value
    datasets_dir: str = str(ROOT / "datasets")
    selected_activities: list[str] | None = None
    window_overlap: float = 0.0
    subjects_per_group: int = 6
    seed: int = 0

    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 100
    patience: int = 10
    min_delta: float = 0.0
    device: str = (
        "mps"
        if __import__("torch").backends.mps.is_available()
        else "cuda"
        if __import__("torch").cuda.is_available()
        else "cpu"
    )
    output_root: str = str(ROOT / "artifacts" / "final_pipeline")
    max_folds: int | None = None
    force_rerun: bool = False


RUN_CONFIG = Config()


@dataclass(frozen=True)
class AdapterSplit:
    identifier: str
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]


def _infer_window_size_from_batch(batch: Any) -> int:
    import torch

    if isinstance(batch, dict):
        x = batch.get("x") or batch.get("features") or batch.get("inputs")
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        a, b = batch[0], batch[1]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.dim() <= 1 and b.dim() >= 2:
                x = b
            elif b.dim() <= 1 and a.dim() >= 2:
                x = a
            else:
                x = a
        else:
            x = a
    else:
        raise ValueError("Unsupported dataloader batch format.")
    if x is None:
        raise ValueError("Could not infer input tensor from batch.")
    if x.dim() == 3:
        return int(x.shape[1])
    if x.dim() == 4:
        return int(x.shape[2])
    raise ValueError(f"Unexpected input shape {tuple(x.shape)}")


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    output_root = Path(config.output_root)
    stage_dir = output_root / "01_pretrain_base"
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

    summary_rows: list[dict[str, Any]] = []
    skipped_folds: list[str] = []
    for fold in folds:
        base_split = split_indices_for_fold(
            session_df,
            window_df,
            type("Tmp", (), {
                "train_subject_ids": fold.base_train_subject_ids,
                "val_subject_ids": fold.val_subject_ids,
                "test_subject_ids": fold.test_subject_ids,
            })(),
        )
        split_dir = stage_dir / fold.fold_id
        split_dir.mkdir(parents=True, exist_ok=True)
        fold_fp = config_fingerprint(
            {
                "stage": "01_pretrain_base",
                "config": asdict(config),
                "shared_cfg": asdict(shared_cfg),
                "fold": asdict(fold),
            }
        )
        metrics_path = split_dir / "metrics.json"
        ckpt_path = split_dir / "best_base_model.pt"
        if not config.force_rerun and metrics_path.exists() and ckpt_path.exists():
            try:
                existing = json.loads(metrics_path.read_text(encoding="utf-8"))
                if existing.get("config_fingerprint") == fold_fp:
                    print(f"[{fold.fold_id}] skipping (already complete with same settings)")
                    summary_rows.append(existing)
                    skipped_folds.append(fold.fold_id)
                    continue
            except Exception:
                pass

        adapter_split = AdapterSplit(
            identifier=fold.fold_id,
            train_indices=base_split.train_indices,
            val_indices=base_split.val_indices,
            test_indices=base_split.test_indices,
        )
        post = PostProcessingPipeline(cfg, pre, window_df, base_split.train_indices)
        samples = post.run()
        loader = Loader(session_df, window_df, post.samples_dir, samples)
        adapter = TorchAdapter(cfg, loader, adapter_split)
        dataloaders = adapter.get_dataloaders(batch_size=config.batch_size)
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        test_loader = dataloaders["test"]

        window_size = _infer_window_size_from_batch(next(iter(train_loader)))
        model = TinierHAR(
            num_channels=cfg.num_of_channels,
            num_classes=cfg.num_of_activities,
            window_size=window_size,
            backbone_config=DEFAULT_CONFIG.backbone,
        )

        trainer_cfg = TrainerConfig(
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            patience=config.patience,
            min_delta=config.min_delta,
            device=config.device,
            checkpoint_path=str(split_dir / "best_base_model.pt"),
            early_stopping_metric="val_macro_f1",
        )
        trainer = TinierHARTrainer(
            model=model,
            num_classes=cfg.num_of_activities,
            config=trainer_cfg,
        )
        history = trainer.fit(train_loader, val_loader)
        test_metrics = trainer.evaluate(test_loader, desc=f"Pretrain Test {fold.fold_id}")
        result = {
            "config_fingerprint": fold_fp,
            "fold_id": fold.fold_id,
            "base_train_subject_ids": fold.base_train_subject_ids,
            "meta_train_subject_ids": fold.meta_train_subject_ids,
            "val_subject_ids": fold.val_subject_ids,
            "test_subject_ids": fold.test_subject_ids,
            "test_loss": float(test_metrics["loss"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "best_val_loss": float(trainer.state.best_val_loss),
            "best_val_macro_f1": float(trainer.state.best_val_macro_f1),
            "best_epoch": int(trainer.state.best_epoch),
        }
        (split_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (split_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        summary_rows.append(result)
        print(
            f"[{fold.fold_id}] pretrain test_macro_f1={result['test_macro_f1']:.4f} "
            f"test_loss={result['test_loss']:.4f}"
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
        "mean_test_macro_f1": float(sum(r["test_macro_f1"] for r in summary_rows) / max(1, len(summary_rows))),
        "mean_test_loss": float(sum(r["test_loss"] for r in summary_rows) / max(1, len(summary_rows))),
        "folds": summary_rows,
    }
    (stage_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from whar_datasets import (
    Loader,
    PostProcessingPipeline,
    PreProcessingPipeline,
    TorchAdapter,
    WHARDatasetID,
    get_dataset_cfg,
)

from hyper_har.backbone.tinierhar import TinierHAR
from hyper_har.config import DEFAULT_CONFIG
from hyper_har.splitting import StrictAdaptationSplitter
from hyper_har.training.trainer import TinierHARTrainer, TrainerConfig

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DATASET_ID = WHARDatasetID.WEAR
OUTPUT_DIR = ROOT / "artifacts" / "strict_adaptation" / "pretrain"


def _is_completed_split(split_dir: Path) -> bool:
    return all(
        path.exists()
        for path in [
            split_dir / "best_tinierhar.pt",
            split_dir / "metrics.json",
            split_dir / "history.json",
            split_dir / "confusion_matrix.pt",
            split_dir / "confusion_matrix.png",
        ]
    )


def _load_existing_metrics(split_dir: Path) -> dict[str, Any] | None:
    metrics_path = split_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _infer_window_size_from_batch(batch: object) -> int:
    if isinstance(batch, dict):
        x = None
        for key in ("x", "features", "inputs"):
            if key in batch:
                x = batch[key]
                break
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
    if not isinstance(x, torch.Tensor):
        raise ValueError("Could not infer input tensor from batch.")
    if x.dim() == 3:
        return int(x.shape[1])
    if x.dim() == 4:
        return int(x.shape[2])
    raise ValueError(f"Expected 3D or 4D input tensors, got shape {tuple(x.shape)}")


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


def main() -> None:
    cfg = get_dataset_cfg(DATASET_ID, datasets_dir=str(ROOT / "datasets"))
    train_cfg = DEFAULT_CONFIG.training
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pre_pipeline = PreProcessingPipeline(cfg)
    _, session_df, window_df = pre_pipeline.run()
    splitter = StrictAdaptationSplitter(cfg)
    folds = splitter.get_folds(session_df, window_df)
    summary_rows: list[dict[str, Any]] = []

    for split_idx, fold in enumerate(folds):
        split = fold.pretrain_split
        subject_id = int(fold.test_subject_id)
        split_dir = OUTPUT_DIR / f"subject_{subject_id}"
        split_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n=== STRICT PRETRAIN split {split_idx + 1}/{len(folds)} | test_subject={subject_id} ==="
        )
        print(
            "Subject roles "
            f"base={fold.base_pretrain_subject_ids} "
            f"meta_train={fold.meta_train_subject_ids} "
            f"meta_val={fold.meta_val_subject_ids} "
            f"test={subject_id} "
            f"pretrain_val_split={fold.pretrain_val_split_level}"
        )
        if _is_completed_split(split_dir):
            existing = _load_existing_metrics(split_dir)
            if (
                existing is not None
                and existing.get("pretrain_val_split_level")
                == fold.pretrain_val_split_level
            ):
                summary_rows.append(existing)
                print(f"Found completed artifacts, skipping subject {subject_id}.")
                continue
            if existing is not None:
                print(
                    "Found completed artifacts with different split metadata; "
                    f"retraining subject {subject_id}."
                )

        post_pipeline = PostProcessingPipeline(
            cfg, pre_pipeline, window_df, split.train_indices
        )
        samples = post_pipeline.run()
        loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)
        print(
            "Pretrain split subjects "
            f"(train/val/test)="
            f"{_subject_ids_for_indices(loader, split.train_indices)}/"
            f"{_subject_ids_for_indices(loader, split.val_indices)}/"
            f"{_subject_ids_for_indices(loader, split.test_indices)}"
        )

        adapter = TorchAdapter(cfg, loader, split)
        dataloaders = adapter.get_dataloaders(batch_size=train_cfg.batch_size)
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        test_loader = dataloaders["test"]
        window_size = _infer_window_size_from_batch(next(iter(train_loader)))
        num_channels = cfg.num_of_channels
        num_classes = cfg.num_of_activities
        model = TinierHAR(
            num_channels=num_channels,
            num_classes=num_classes,
            window_size=window_size,
            backbone_config=DEFAULT_CONFIG.backbone,
        )
        checkpoint_path = split_dir / "best_tinierhar.pt"
        trainer_cfg = TrainerConfig(
            epochs=train_cfg.num_epochs,
            patience=train_cfg.patience,
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            checkpoint_path=str(checkpoint_path),
        )
        class_weights = _fetch_class_weights(loader, split, num_classes)
        trainer = TinierHARTrainer(
            model=model,
            num_classes=num_classes,
            config=trainer_cfg,
            class_weights=class_weights,
        )
        history = trainer.fit(train_loader, val_loader)
        test_metrics = trainer.evaluate(test_loader, desc=f"Test subject {subject_id}")

        torch.save(test_metrics["confusion_matrix"], split_dir / "confusion_matrix.pt")
        trainer.save_confusion_matrix_plot(
            test_metrics["confusion_matrix"], str(split_dir / "confusion_matrix.png")
        )
        result = {
            "split_index": split_idx,
            "test_subject_id": int(subject_id),
            "base_pretrain_subject_ids": sorted(
                int(x) for x in fold.base_pretrain_subject_ids
            ),
            "pretrain_train_subject_ids": sorted(
                int(x) for x in fold.pretrain_train_subject_ids
            ),
            "pretrain_val_subject_ids": sorted(
                int(x) for x in fold.pretrain_val_subject_ids
            ),
            "pretrain_val_split_level": fold.pretrain_val_split_level,
            "meta_train_subject_ids": sorted(
                int(x) for x in fold.meta_train_subject_ids
            ),
            "meta_val_subject_ids": sorted(int(x) for x in fold.meta_val_subject_ids),
            "test_loss": float(test_metrics["loss"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "best_val_loss": float(trainer.state.best_val_loss),
            "best_val_macro_f1": float(
                max(history["val_macro_f1"]) if history["val_macro_f1"] else 0.0
            ),
            "best_epoch": int(trainer.state.best_epoch),
            "checkpoint_path": str(checkpoint_path),
        }
        with (split_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        with (split_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        summary_rows.append(result)
        print(
            f"Subject {subject_id}: test_loss={result['test_loss']:.4f}, "
            f"test_macro_f1={result['test_macro_f1']:.4f}"
        )

    summary = {
        "num_splits": len(summary_rows),
        "mean_test_macro_f1": sum(float(r["test_macro_f1"]) for r in summary_rows)
        / max(1, len(summary_rows)),
        "mean_test_loss": sum(float(r["test_loss"]) for r in summary_rows)
        / max(1, len(summary_rows)),
        "splits": summary_rows,
        "output_dir": str(OUTPUT_DIR),
        "splitter": "StrictAdaptationSplitter",
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Strict adaptation pretraining finished ===")
    print(f"Mean test macro F1: {summary['mean_test_macro_f1']:.4f}")
    print(f"Saved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import torch
from whar_datasets import (
    Loader,
    LOSOSplitter,
    PostProcessingPipeline,
    PreProcessingPipeline,
    TorchAdapter,
    WHARDatasetID,
    get_dataset_cfg,
)

from hyper_har.backbone.tinierhar import TinierHAR
from hyper_har.config import DEFAULT_CONFIG
from hyper_har.training.trainer import TinierHARTrainer, TrainerConfig

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Script-level run settings
DATASET_ID = WHARDatasetID.WEAR
OUTPUT_DIR = ROOT / "artifacts" / "loso_cv"


def _infer_window_size_from_batch(batch: object) -> int:
    if isinstance(batch, dict):
        x = batch.get("x") or batch.get("features") or batch.get("inputs")
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        a, b = batch[0], batch[1]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            # Support both (x, y) and (y, x), e.g. TorchAdapter returns (y, x).
            if a.dim() <= 1 and b.dim() >= 2:
                x = b
            elif b.dim() <= 1 and a.dim() >= 2:
                x = a
            else:
                x = a
        else:
            x = a
    elif isinstance(batch, (tuple, list)) and len(batch) >= 1:
        x = batch[0]
    else:
        raise ValueError("Unsupported dataloader batch format.")

    if x is None or not isinstance(x, torch.Tensor):
        raise ValueError("Could not infer input tensor from batch.")

    if x.dim() == 3:
        # (B, T, S)
        return int(x.shape[1])
    if x.dim() == 4:
        # (B, C, T, S)
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
            # whar_datasets uses -1 for classes absent in windows -> ignore via zero weight.
            weights[class_id] = 0.0 if raw_w < 0.0 else raw_w
        return weights

    weights = torch.as_tensor(weights_obj, dtype=torch.float32).view(-1)
    if weights.numel() != num_classes:
        raise ValueError(
            f"Class weights length mismatch: expected {num_classes}, got {weights.numel()}."
        )
    weights = torch.where(weights < 0.0, torch.zeros_like(weights), weights)
    return weights


def main() -> None:
    cfg = get_dataset_cfg(DATASET_ID, datasets_dir=str(ROOT / "datasets"))
    train_cfg = DEFAULT_CONFIG.training
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pre_pipeline = PreProcessingPipeline(cfg)
    _, session_df, window_df = pre_pipeline.run()

    splitter = LOSOSplitter(cfg)
    splits = splitter.get_splits(session_df, window_df)
    summary_rows: list[dict[str, float | int | str]] = []

    for split_idx, split in enumerate(splits):
        subject_id = getattr(split, "test_subject_id", split_idx)
        split_dir = OUTPUT_DIR / f"subject_{subject_id}"
        split_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n=== LOSO split {split_idx + 1}/{len(splits)} | subject={subject_id} ==="
        )

        post_pipeline = PostProcessingPipeline(
            cfg, pre_pipeline, window_df, split.train_indices
        )
        samples = post_pipeline.run()

        loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)
        adapter = TorchAdapter(cfg, loader, split)
        dataloaders = adapter.get_dataloaders(batch_size=train_cfg.batch_size)

        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        test_loader = dataloaders["test"]

        train_batch = next(iter(train_loader))
        window_size = _infer_window_size_from_batch(train_batch)

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
        if class_weights is not None:
            print(
                f"Using class weights from loader.get_class_weights: shape={tuple(class_weights.shape)}"
            )
        else:
            print("No class weights returned, using unweighted cross-entropy.")

        trainer = TinierHARTrainer(
            model=model,
            num_classes=num_classes,
            config=trainer_cfg,
            class_weights=class_weights,
        )

        history = trainer.fit(train_loader, val_loader)
        test_metrics = trainer.evaluate(test_loader, desc=f"Test subject {subject_id}")

        cm_tensor_path = split_dir / "confusion_matrix.pt"
        cm_plot_path = split_dir / "confusion_matrix.png"
        torch.save(test_metrics["confusion_matrix"], cm_tensor_path)
        trainer.save_confusion_matrix_plot(
            test_metrics["confusion_matrix"], str(cm_plot_path)
        )

        result = {
            "split_index": split_idx,
            "subject_id": int(subject_id)
            if isinstance(subject_id, int)
            else str(subject_id),
            "test_loss": float(test_metrics["loss"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "best_val_loss": float(trainer.state.best_val_loss),
            "best_epoch": int(trainer.state.best_epoch),
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

    mean_macro_f1 = sum(float(r["test_macro_f1"]) for r in summary_rows) / max(
        1, len(summary_rows)
    )
    mean_loss = sum(float(r["test_loss"]) for r in summary_rows) / max(
        1, len(summary_rows)
    )
    summary = {
        "num_splits": len(summary_rows),
        "mean_test_macro_f1": mean_macro_f1,
        "mean_test_loss": mean_loss,
        "splits": summary_rows,
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LOSO cross-validation finished ===")
    print(f"Mean test macro F1: {mean_macro_f1:.4f}")
    print(f"Mean test loss: {mean_loss:.4f}")
    print(f"Saved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

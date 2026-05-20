import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from common import (
    ROOT,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASETS_DIR,
    DEFAULT_SELECTED_ACTIVITIES,
    DEFAULT_SEED,
    DEFAULT_TEST_SUBJECTS,
    DEFAULT_VAL_SUBJECTS,
    DEFAULT_WINDOW_OVERLAP,
    DEFAULT_SPLIT_STRATEGY,
    DEFAULT_VAL_PERCENTAGE,
    SharedConfig,
    WindowDataset,
    build_loader,
    build_or_load_loso_folds,
    class_names,
    classification_metrics,
    config_fingerprint,
    build_supcon_projection_head,
    load_supcon_backbone,
    prepare_cfg,
    prepare_inputs,
    reconcile_activity_config,
    resolve_output_root,
    save_confusion_matrix_plot,
    set_seed,
    split_indices_for_fold,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from whar_datasets import PreProcessingPipeline, WHARDatasetID

from hyper_har.backbone.tinierhar import TinierHAR


@dataclass(frozen=True)
class Config:
    dataset_id: str = DEFAULT_DATASET_ID
    datasets_dir: str = DEFAULT_DATASETS_DIR
    selected_activities: list[str] | None = DEFAULT_SELECTED_ACTIVITIES
    window_overlap: float = DEFAULT_WINDOW_OVERLAP
    val_subjects: int = DEFAULT_VAL_SUBJECTS
    test_subjects: int = DEFAULT_TEST_SUBJECTS
    seed: int = DEFAULT_SEED
    split_strategy: str = DEFAULT_SPLIT_STRATEGY
    val_percentage: float = DEFAULT_VAL_PERCENTAGE

    batch_size: int = 128
    num_workers: int = 0
    learning_rate: float = 1e-3
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 80
    patience: int = 10
    min_delta: float = 0.0
    finetune_backbone: bool = False
    finetune_projection_head: bool = False
    embedding_space: str = "projected"  # "projected" or "backbone"
    grad_clip_norm: float | None = 5.0
    skip_missing_folds: bool = False
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_root: str | None = None
    supcon_stage_name: str = "02_tinierhar_supcon_loso"
    eval_stage_name: str = "03_supcon_linear_head_loso"
    max_folds: int | None = None
    force_rerun: bool = False


RUN_CONFIG = Config()


class LinearProbe(nn.Module):
    def __init__(
        self,
        backbone: TinierHAR,
        projection_head: nn.Module | None,
        num_classes: int,
        finetune_backbone: bool,
        finetune_projection_head: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head
        self.finetune_backbone = bool(finetune_backbone)
        self.finetune_projection_head = bool(finetune_projection_head)
        feature_dim = (
            int(projection_head[-1].out_features)
            if projection_head is not None
            else int(2 * backbone.nb_units_gru)
        )
        self.head = nn.Linear(feature_dim, int(num_classes))
        for param in self.backbone.parameters():
            param.requires_grad = self.finetune_backbone
        if self.projection_head is not None:
            for param in self.projection_head.parameters():
                param.requires_grad = self.finetune_projection_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        train_feature_extractor = self.finetune_backbone or self.finetune_projection_head
        if train_feature_extractor:
            features = self.backbone.encode(x)
            if self.projection_head is not None:
                features = self.projection_head(features)
        else:
            with torch.no_grad():
                features = self.backbone.encode(x)
                if self.projection_head is not None:
                    features = self.projection_head(features)
        return self.head(features)


def _run_epoch(
    model: LinearProbe,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float | None,
    num_classes: int,
    desc: str,
) -> dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    running_loss = 0.0
    running_total = 0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in tqdm(dataloader, desc=desc, leave=False):
        x = prepare_inputs(batch["x"]).to(device).float()
        y = batch["y"].to(device).long().view(-1)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(grad_clip_norm)
                    )
                optimizer.step()
        running_loss += float(loss.item()) * int(y.size(0))
        running_total += int(y.size(0))
        preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
    y_true = np.concatenate(targets, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    metrics = classification_metrics(y_true, y_pred, num_classes)
    metrics["loss"] = running_loss / max(1, running_total)
    return metrics


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)
    output_root = resolve_output_root(config.output_root, config.dataset_id)
    supcon_stage_dir = output_root / config.supcon_stage_name
    eval_dir = output_root / config.eval_stage_name
    eval_dir.mkdir(parents=True, exist_ok=True)

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

    rows: list[dict[str, Any]] = []
    skipped_folds: list[str] = []
    for fold in folds:
        ckpt_path = supcon_stage_dir / fold.fold_id / "best_supcon_backbone.pt"
        if not ckpt_path.exists():
            if config.skip_missing_folds:
                skipped_folds.append(fold.fold_id)
                print(f"[{fold.fold_id}] skipping missing checkpoint: {ckpt_path}")
                continue
            raise FileNotFoundError(f"Missing SupCon checkpoint: {ckpt_path}")

        split = split_indices_for_fold(session_df, window_df, fold)
        split_dir = eval_dir / fold.fold_id
        split_dir.mkdir(parents=True, exist_ok=True)
        fold_fp = config_fingerprint(
            {
                "stage": config.eval_stage_name,
                "config": asdict(config),
                "shared_cfg": asdict(shared_cfg),
                "fold": asdict(fold),
                "supcon_checkpoint": str(ckpt_path),
                "num_classes": int(cfg.num_of_activities),
                "class_names": class_names(cfg),
            }
        )
        metrics_path = split_dir / "metrics.json"
        if not config.force_rerun and metrics_path.exists():
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if existing.get("config_fingerprint") == fold_fp:
                print(f"[{fold.fold_id}] skipping (already complete)")
                rows.append(existing)
                skipped_folds.append(fold.fold_id)
                continue

        loader = build_loader(cfg, session_df, pre, window_df, split.train_indices)
        train_loader = DataLoader(
            WindowDataset(loader, split.train_indices),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        val_loader = DataLoader(
            WindowDataset(loader, split.val_indices),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        test_loader = DataLoader(
            WindowDataset(loader, split.test_indices),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        if config.embedding_space not in {"projected", "backbone"}:
            raise ValueError("embedding_space must be 'projected' or 'backbone'.")
        backbone, checkpoint = load_supcon_backbone(ckpt_path, device)
        projection_head = (
            build_supcon_projection_head(checkpoint, device)
            if config.embedding_space == "projected"
            else None
        )
        model = LinearProbe(
            backbone=backbone,
            projection_head=projection_head,
            num_classes=int(cfg.num_of_activities),
            finetune_backbone=config.finetune_backbone,
            finetune_projection_head=config.finetune_projection_head,
        ).to(device)
        if config.finetune_backbone:
            param_groups = [
                {
                    "params": model.backbone.parameters(),
                    "lr": config.backbone_learning_rate,
                },
                {"params": model.head.parameters(), "lr": config.learning_rate},
            ]
            if model.projection_head is not None and config.finetune_projection_head:
                param_groups.insert(
                    1,
                    {
                        "params": model.projection_head.parameters(),
                        "lr": config.backbone_learning_rate,
                    },
                )
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=config.weight_decay,
            )
        elif model.projection_head is not None and config.finetune_projection_head:
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": model.projection_head.parameters(),
                        "lr": config.backbone_learning_rate,
                    },
                    {"params": model.head.parameters(), "lr": config.learning_rate},
                ],
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                model.head.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        criterion = nn.CrossEntropyLoss()
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_macro_f1": [],
            "val_loss": [],
            "val_macro_f1": [],
        }
        best_val_f1 = float("-inf")
        best_epoch = -1
        patience_counter = 0
        best_path = split_dir / "best_linear_probe.pt"
        for epoch in range(1, int(config.epochs) + 1):
            train_metrics = _run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                config.grad_clip_norm,
                int(cfg.num_of_activities),
                desc=f"{fold.fold_id} linear train {epoch}/{config.epochs}",
            )
            val_metrics = _run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                grad_clip_norm=None,
                num_classes=int(cfg.num_of_activities),
                desc=f"{fold.fold_id} linear val {epoch}/{config.epochs}",
            )
            history["train_loss"].append(float(train_metrics["loss"]))
            history["train_macro_f1"].append(float(train_metrics["macro_f1"]))
            history["val_loss"].append(float(val_metrics["loss"]))
            history["val_macro_f1"].append(float(val_metrics["macro_f1"]))
            improved = val_metrics["macro_f1"] > best_val_f1 + float(config.min_delta)
            if improved:
                best_val_f1 = float(val_metrics["macro_f1"])
                best_epoch = int(epoch)
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": asdict(config),
                        "supcon_checkpoint": str(ckpt_path),
                        "supcon_model_meta": checkpoint.get("model_meta", {}),
                        "embedding_space": config.embedding_space,
                        "fold": asdict(fold),
                    },
                    best_path,
                )
            else:
                patience_counter += 1
            print(
                f"[{fold.fold_id} epoch {epoch:03d}] "
                f"train_f1={train_metrics['macro_f1']:.4f} "
                f"val_f1={val_metrics['macro_f1']:.4f} "
                f"best_val_f1={best_val_f1:.4f} "
                f"patience={patience_counter}/{config.patience}"
            )
            if patience_counter >= int(config.patience):
                break

        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=False)[
                "model_state_dict"
            ]
        )
        test_metrics = _run_epoch(
            model,
            test_loader,
            criterion,
            device,
            optimizer=None,
            grad_clip_norm=None,
            num_classes=int(cfg.num_of_activities),
            desc=f"{fold.fold_id} linear test",
        )
        cm_path = split_dir / "confusion_matrix.png"
        save_confusion_matrix_plot(
            test_metrics["confusion_matrix"],
            class_names(cfg),
            cm_path,
            title=f"{fold.fold_id} Linear Probe",
        )
        result = {
            "config_fingerprint": fold_fp,
            "fold_id": fold.fold_id,
            "train_subject_ids": fold.train_subject_ids,
            "val_subject_ids": fold.val_subject_ids,
            "test_subject_ids": fold.test_subject_ids,
            "best_epoch": int(best_epoch),
            "best_val_macro_f1": float(best_val_f1),
            "test_loss": float(test_metrics["loss"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "test_weighted_f1": float(test_metrics["weighted_f1"]),
            "embedding_space": config.embedding_space,
            "checkpoint_path": str(best_path),
            "confusion_matrix_path": str(cm_path),
        }
        metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        (split_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        rows.append(result)
        print(f"[{fold.fold_id}] linear test_macro_f1={result['test_macro_f1']:.4f}")

    summary = {
        "config": asdict(config),
        "splits_manifest_path": str(manifest_path),
        "supcon_stage_dir": str(supcon_stage_dir),
        "num_folds": len(rows),
        "skipped_folds": skipped_folds,
        "mean_test_macro_f1": float(
            sum(r["test_macro_f1"] for r in rows) / max(1, len(rows))
        ),
        "folds": rows,
    }
    (eval_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

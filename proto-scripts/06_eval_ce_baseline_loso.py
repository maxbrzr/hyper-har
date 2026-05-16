import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from common import (
    ROOT,
    SharedConfig,
    WindowDataset,
    build_loader,
    build_or_load_loso_folds,
    class_names,
    classification_metrics,
    config_fingerprint,
    load_ce_backbone,
    prepare_cfg,
    prepare_inputs,
    save_confusion_matrix_plot,
    set_seed,
    split_indices_for_fold,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from whar_datasets import PreProcessingPipeline, WHARDatasetID


@dataclass(frozen=True)
class Config:
    dataset_id: str = WHARDatasetID.WEAR.value
    datasets_dir: str = str("datasets")
    selected_activities: list[str] | None = None
    window_overlap: float = 0.5
    val_subjects: int = 3
    test_subjects: int = 1
    seed: int = 0

    batch_size: int = 256
    num_workers: int = 0
    skip_missing_folds: bool = False
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_root: str = str(ROOT / "artifacts" / "proto_pipeline")
    ce_stage_name: str = "01_tinierhar_ce_loso"
    eval_stage_name: str = "06_ce_baseline_eval_loso"
    max_folds: int | None = None
    force_rerun: bool = False


RUN_CONFIG = Config()


def _loader(dataset: WindowDataset, config: Config) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )


@torch.no_grad()
def _evaluate_classifier(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true: list[torch.Tensor] = []
    y_pred: list[torch.Tensor] = []

    for batch in tqdm(dataloader, desc="CE eval", leave=False):
        x = prepare_inputs(batch["x"]).to(device).float()
        y = batch["y"].long().view(-1).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)

        batch_size = int(y.numel())
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        y_true.append(y.cpu())
        y_pred.append(pred.cpu())

    if total_count == 0:
        raise ValueError("Cannot evaluate an empty dataloader.")

    true = torch.cat(y_true).numpy()
    pred = torch.cat(y_pred).numpy()
    metrics = classification_metrics(true, pred, num_classes)
    metrics["loss"] = float(total_loss / total_count)
    return metrics


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)
    output_root = Path(config.output_root)
    ce_stage_dir = output_root / config.ce_stage_name
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

    rows: list[dict[str, Any]] = []
    skipped_folds: list[str] = []
    for fold in folds:
        ckpt_path = ce_stage_dir / fold.fold_id / "best_model_with_meta.pt"
        if not ckpt_path.exists():
            if config.skip_missing_folds:
                skipped_folds.append(fold.fold_id)
                print(f"[{fold.fold_id}] skipping missing checkpoint: {ckpt_path}")
                continue
            raise FileNotFoundError(
                f"Missing CE checkpoint: {ckpt_path}. "
                "Run 01_train_tinierhar_ce_loso.py first."
            )

        split = split_indices_for_fold(session_df, window_df, fold)
        split_dir = eval_dir / fold.fold_id
        split_dir.mkdir(parents=True, exist_ok=True)
        fold_fp = config_fingerprint(
            {
                "stage": config.eval_stage_name,
                "config": asdict(config),
                "shared_cfg": asdict(shared_cfg),
                "fold": asdict(fold),
                "ce_checkpoint": str(ckpt_path),
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
        val_ds = WindowDataset(loader, split.val_indices)
        test_ds = WindowDataset(loader, split.test_indices)
        model, checkpoint = load_ce_backbone(ckpt_path, device)
        model_meta = dict(checkpoint.get("model_meta", {}))

        split_metrics: dict[str, Any] = {}
        for split_name, dataset in (("val", val_ds), ("test", test_ds)):
            metrics = _evaluate_classifier(
                model,
                _loader(dataset, config),
                device,
                num_classes=int(cfg.num_of_activities),
            )
            split_metrics[split_name] = metrics
            cm_path = split_dir / f"{split_name}_confusion_matrix.png"
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                class_names(cfg),
                cm_path,
                title=f"{fold.fold_id} CE Baseline {split_name}",
            )
            split_metrics[split_name]["confusion_matrix_path"] = str(cm_path)

        result = {
            "config_fingerprint": fold_fp,
            "fold_id": fold.fold_id,
            "train_subject_ids": fold.train_subject_ids,
            "val_subject_ids": fold.val_subject_ids,
            "test_subject_ids": fold.test_subject_ids,
            "ce_checkpoint": str(ckpt_path),
            "window_size": int(model_meta.get("window_size", -1)),
            "num_channels": int(model_meta.get("num_channels", -1)),
            "num_classes": int(model_meta.get("num_classes", cfg.num_of_activities)),
            "class_names": class_names(cfg),
            "val_loss": float(split_metrics["val"]["loss"]),
            "val_accuracy": float(split_metrics["val"]["accuracy"]),
            "val_macro_f1": float(split_metrics["val"]["macro_f1"]),
            "val_weighted_f1": float(split_metrics["val"]["weighted_f1"]),
            "test_loss": float(split_metrics["test"]["loss"]),
            "test_accuracy": float(split_metrics["test"]["accuracy"]),
            "test_macro_f1": float(split_metrics["test"]["macro_f1"]),
            "test_weighted_f1": float(split_metrics["test"]["weighted_f1"]),
            "val_confusion_matrix_path": split_metrics["val"]["confusion_matrix_path"],
            "test_confusion_matrix_path": split_metrics["test"][
                "confusion_matrix_path"
            ],
        }
        metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        rows.append(result)
        print(
            f"[{fold.fold_id}] CE eval test_macro_f1="
            f"{result['test_macro_f1']:.4f} test_loss={result['test_loss']:.4f}"
        )

    summary = {
        "config": asdict(config),
        "splits_manifest_path": str(manifest_path),
        "ce_stage_dir": str(ce_stage_dir),
        "num_folds": len(rows),
        "skipped_folds": skipped_folds,
        "mean_test_macro_f1": float(
            sum(r["test_macro_f1"] for r in rows) / max(1, len(rows))
        ),
        "mean_test_accuracy": float(
            sum(r["test_accuracy"] for r in rows) / max(1, len(rows))
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

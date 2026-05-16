import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from common import (
    ROOT,
    SharedConfig,
    WindowDataset,
    build_loader,
    build_or_load_loso_folds,
    class_names,
    classification_metrics,
    config_fingerprint,
    cosine_logits,
    build_supcon_projection_head,
    extract_supcon_embeddings,
    load_ce_backbone,
    load_supcon_backbone,
    make_class_prototypes,
    prepare_cfg,
    save_confusion_matrix_plot,
    set_seed,
    split_indices_for_fold,
)
from torch.utils.data import DataLoader
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
    cosine_temperature: float = 0.1
    normalize_embeddings: bool = True
    embedding_space: str = "projected"  # "projected" or "backbone"
    backbone_source: str = "supcon"  # "supcon" or "ce"
    skip_missing_folds: bool = False
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_root: str = str(ROOT / "artifacts" / "proto_pipeline")
    supcon_stage_name: str = "02_tinierhar_supcon_loso"
    ce_stage_name: str = "01_tinierhar_ce_loso"
    eval_stage_name: str = "04_supcon_fixed_prototypes_loso"
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


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)
    output_root = Path(config.output_root)
    supcon_stage_dir = output_root / config.supcon_stage_name
    ce_stage_dir = output_root / config.ce_stage_name
    eval_dir = output_root / config.eval_stage_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    if config.backbone_source not in {"supcon", "ce"}:
        raise ValueError("backbone_source must be 'supcon' or 'ce'.")

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
        ckpt_path = (
            supcon_stage_dir / fold.fold_id / "best_supcon_backbone.pt"
            if config.backbone_source == "supcon"
            else ce_stage_dir / fold.fold_id / "best_model_with_meta.pt"
        )
        if not ckpt_path.exists():
            if config.skip_missing_folds:
                skipped_folds.append(fold.fold_id)
                print(f"[{fold.fold_id}] skipping missing checkpoint: {ckpt_path}")
                continue
            raise FileNotFoundError(
                f"Missing {config.backbone_source} checkpoint: {ckpt_path}"
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
                "backbone_checkpoint": str(ckpt_path),
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
        train_ds = WindowDataset(loader, split.train_indices)
        val_ds = WindowDataset(loader, split.val_indices)
        test_ds = WindowDataset(loader, split.test_indices)
        if config.embedding_space not in {"projected", "backbone"}:
            raise ValueError("embedding_space must be 'projected' or 'backbone'.")
        if config.backbone_source == "supcon":
            backbone, checkpoint = load_supcon_backbone(ckpt_path, device)
            projection_head = (
                build_supcon_projection_head(checkpoint, device)
                if config.embedding_space == "projected"
                else None
            )
            effective_embedding_space = config.embedding_space
        else:
            backbone, checkpoint = load_ce_backbone(ckpt_path, device)
            projection_head = None
            effective_embedding_space = "backbone"
            if config.embedding_space == "projected":
                print(
                    f"[{fold.fold_id}] CE checkpoint has no projection head; "
                    "using backbone embeddings."
                )
        train_emb, train_y, _ = extract_supcon_embeddings(
            backbone,
            projection_head,
            _loader(train_ds, config),
            device,
            normalize=config.normalize_embeddings,
        )
        prototypes = make_class_prototypes(
            train_emb,
            train_y,
            num_classes=int(cfg.num_of_activities),
            normalize=True,
        )
        split_metrics: dict[str, Any] = {}
        for split_name, dataset in (("val", val_ds), ("test", test_ds)):
            emb, y, _ = extract_supcon_embeddings(
                backbone,
                projection_head,
                _loader(dataset, config),
                device,
                normalize=config.normalize_embeddings,
            )
            logits = cosine_logits(emb, prototypes, config.cosine_temperature)
            pred = logits.argmax(dim=1).numpy()
            metrics = classification_metrics(
                y.numpy(), pred, int(cfg.num_of_activities)
            )
            split_metrics[split_name] = metrics
            cm_path = split_dir / f"{split_name}_confusion_matrix.png"
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                class_names(cfg),
                cm_path,
                title=f"{fold.fold_id} Fixed Prototypes {split_name}",
            )
            split_metrics[split_name]["confusion_matrix_path"] = str(cm_path)

        torch.save(
            {
                "prototypes": prototypes,
                "class_labels": torch.arange(
                    int(cfg.num_of_activities), dtype=torch.long
                ),
                "config": asdict(config),
                "backbone_source": config.backbone_source,
                "backbone_checkpoint": str(ckpt_path),
                "embedding_space": config.embedding_space,
                "effective_embedding_space": effective_embedding_space,
                "fold": asdict(fold),
            },
            split_dir / "fixed_prototypes.pt",
        )
        result = {
            "config_fingerprint": fold_fp,
            "fold_id": fold.fold_id,
            "train_subject_ids": fold.train_subject_ids,
            "val_subject_ids": fold.val_subject_ids,
            "test_subject_ids": fold.test_subject_ids,
            "val_accuracy": float(split_metrics["val"]["accuracy"]),
            "val_macro_f1": float(split_metrics["val"]["macro_f1"]),
            "test_accuracy": float(split_metrics["test"]["accuracy"]),
            "test_macro_f1": float(split_metrics["test"]["macro_f1"]),
            "test_weighted_f1": float(split_metrics["test"]["weighted_f1"]),
            "backbone_source": config.backbone_source,
            "backbone_checkpoint": str(ckpt_path),
            "embedding_space": config.embedding_space,
            "effective_embedding_space": effective_embedding_space,
            "val_confusion_matrix_path": split_metrics["val"]["confusion_matrix_path"],
            "test_confusion_matrix_path": split_metrics["test"][
                "confusion_matrix_path"
            ],
            "prototype_path": str(split_dir / "fixed_prototypes.pt"),
        }
        metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        rows.append(result)
        print(
            f"[{fold.fold_id}] fixed-proto test_macro_f1={result['test_macro_f1']:.4f}"
        )

    summary = {
        "config": asdict(config),
        "splits_manifest_path": str(manifest_path),
        "supcon_stage_dir": str(supcon_stage_dir),
        "ce_stage_dir": str(ce_stage_dir),
        "backbone_source": config.backbone_source,
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

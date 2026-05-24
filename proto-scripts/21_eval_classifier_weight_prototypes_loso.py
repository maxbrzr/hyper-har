import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from common import (
    DEFAULT_DATASET_ID,
    DEFAULT_DATASETS_DIR,
    DEFAULT_SEED,
    DEFAULT_SELECTED_ACTIVITIES,
    DEFAULT_SPLIT_STRATEGY,
    DEFAULT_TEST_SUBJECTS,
    DEFAULT_VAL_PERCENTAGE,
    DEFAULT_VAL_SUBJECTS,
    DEFAULT_WINDOW_OVERLAP,
    SharedConfig,
    WindowDataset,
    build_loader,
    build_or_load_loso_folds,
    class_names,
    classification_metrics,
    config_fingerprint,
    extract_supcon_embeddings,
    load_ce_backbone,
    prepare_cfg,
    prototype_predictions,
    reconcile_activity_config,
    resolve_output_root,
    save_confusion_matrix_plot,
    set_seed,
    split_indices_for_fold,
)
from torch.utils.data import DataLoader
from whar_datasets import PreProcessingPipeline, WHARDatasetID


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

    batch_size: int = 256
    num_workers: int = 0
    cosine_temperature: float = 0.1
    normalize_embeddings: bool = True
    normalize_classifier_weights: bool = True
    distance_metric: str = "cosine"  # "cosine" or "euclidean"
    skip_missing_folds: bool = False
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_root: str | None = None
    ce_stage_name: str = "01_tinierhar_ce_loso"
    eval_stage_name: str = "21_classifier_weight_prototypes_loso"
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


def _classifier_weight_prototypes(
    model: torch.nn.Module,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, torch.nn.Sequential) or not classifier:
        raise TypeError("Expected TinierHAR classifier to be a non-empty Sequential.")
    linear = classifier[0]
    if not isinstance(linear, torch.nn.Linear):
        raise TypeError("Expected TinierHAR classifier[0] to be nn.Linear.")
    prototypes = linear.weight.detach().cpu().clone()
    if normalize:
        prototypes = F.normalize(prototypes, p=2, dim=1)
    labels = torch.arange(prototypes.shape[0], dtype=torch.long)
    bias = None if linear.bias is None else linear.bias.detach().cpu().clone()
    return prototypes, labels, bias


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)
    output_root = resolve_output_root(config.output_root, config.dataset_id)
    ce_stage_dir = output_root / config.ce_stage_name
    if config.distance_metric not in {"cosine", "euclidean"}:
        raise ValueError("distance_metric must be 'cosine' or 'euclidean'.")
    effective_normalize_embeddings = (
        bool(config.normalize_embeddings) and config.distance_metric == "cosine"
    )
    effective_normalize_weights = (
        bool(config.normalize_classifier_weights) and config.distance_metric == "cosine"
    )

    eval_stage_parts = [
        config.eval_stage_name,
        "ce_backbone",
        config.distance_metric,
        "normalized_weights" if effective_normalize_weights else "raw_weights",
    ]
    if effective_normalize_embeddings:
        eval_stage_parts.append("normalized_embeddings")
    eval_stage_name = "_".join(eval_stage_parts)
    eval_dir = output_root / eval_stage_name
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
        ckpt_path = ce_stage_dir / fold.fold_id / "best_model_with_meta.pt"
        if not ckpt_path.exists():
            if config.skip_missing_folds:
                skipped_folds.append(fold.fold_id)
                print(f"[{fold.fold_id}] skipping missing checkpoint: {ckpt_path}")
                continue
            raise FileNotFoundError(f"Missing CE checkpoint: {ckpt_path}")

        split = split_indices_for_fold(session_df, window_df, fold)
        split_dir = eval_dir / fold.fold_id
        split_dir.mkdir(parents=True, exist_ok=True)
        fold_fp = config_fingerprint(
            {
                "stage": config.eval_stage_name,
                "resolved_eval_stage": eval_stage_name,
                "config": asdict(config),
                "shared_cfg": asdict(shared_cfg),
                "fold": asdict(fold),
                "backbone_checkpoint": str(ckpt_path),
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
        val_ds = WindowDataset(loader, split.val_indices)
        test_ds = WindowDataset(loader, split.test_indices)
        backbone, _checkpoint = load_ce_backbone(ckpt_path, device)
        prototypes, prototype_class_labels, classifier_bias = (
            _classifier_weight_prototypes(backbone, effective_normalize_weights)
        )

        split_metrics: dict[str, Any] = {}
        for split_name, dataset in (("val", val_ds), ("test", test_ds)):
            emb, y, _ = extract_supcon_embeddings(
                backbone,
                None,
                _loader(dataset, config),
                device,
                normalize=effective_normalize_embeddings,
            )
            pred = prototype_predictions(
                emb,
                prototypes,
                prototype_class_labels,
                config.cosine_temperature,
                config.distance_metric,
            )
            metrics = classification_metrics(
                y.numpy(),
                pred,
                int(cfg.num_of_activities),
            )
            split_metrics[split_name] = metrics
            cm_path = split_dir / f"{split_name}_confusion_matrix.png"
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                class_names(cfg),
                cm_path,
                title=f"{fold.fold_id} Classifier-Weight Prototypes {split_name}",
            )
            split_metrics[split_name]["confusion_matrix_path"] = str(cm_path)

        torch.save(
            {
                "prototypes": prototypes,
                "class_labels": prototype_class_labels,
                "classifier_bias": classifier_bias,
                "config": asdict(config),
                "backbone_checkpoint": str(ckpt_path),
                "distance_metric": config.distance_metric,
                "normalize_embeddings": bool(config.normalize_embeddings),
                "effective_normalize_embeddings": effective_normalize_embeddings,
                "normalize_classifier_weights": bool(
                    config.normalize_classifier_weights
                ),
                "effective_normalize_classifier_weights": effective_normalize_weights,
                "fold": asdict(fold),
            },
            split_dir / "classifier_weight_prototypes.pt",
        )
        prototype_norms = prototypes.norm(p=2, dim=1)
        bias_norm = 0.0 if classifier_bias is None else float(classifier_bias.norm().item())
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
            "backbone_source": "ce",
            "backbone_checkpoint": str(ckpt_path),
            "distance_metric": config.distance_metric,
            "normalize_embeddings": bool(config.normalize_embeddings),
            "effective_normalize_embeddings": effective_normalize_embeddings,
            "normalize_classifier_weights": bool(config.normalize_classifier_weights),
            "effective_normalize_classifier_weights": effective_normalize_weights,
            "prototype_class_labels": prototype_class_labels.tolist(),
            "prototype_norm_mean": float(prototype_norms.mean().item()),
            "prototype_norm_std": float(prototype_norms.std(unbiased=False).item()),
            "classifier_bias_norm": bias_norm,
            "val_confusion_matrix_path": split_metrics["val"]["confusion_matrix_path"],
            "test_confusion_matrix_path": split_metrics["test"][
                "confusion_matrix_path"
            ],
            "prototype_path": str(split_dir / "classifier_weight_prototypes.pt"),
        }
        metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        rows.append(result)
        print(
            f"[{fold.fold_id}] classifier-weight-proto "
            f"test_macro_f1={result['test_macro_f1']:.4f}"
        )

    summary = {
        "config": asdict(config),
        "baseline": "frozen CE classifier weights as embedding-space prototypes",
        "splits_manifest_path": str(manifest_path),
        "eval_stage_name": eval_stage_name,
        "eval_dir": str(eval_dir),
        "ce_stage_dir": str(ce_stage_dir),
        "backbone_source": "ce",
        "distance_metric": config.distance_metric,
        "effective_normalize_embeddings": effective_normalize_embeddings,
        "effective_normalize_classifier_weights": effective_normalize_weights,
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
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

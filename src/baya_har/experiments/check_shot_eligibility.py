"""Audit whether every class present for a LOSO subject supports each K-shot run."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from whar_datasets import PreProcessingPipeline, WHARDatasetID

from .common import (
    ARTIFACTS_ROOT,
    DEFAULT_DATASETS_DIR,
    DEFAULT_SEED,
    DEFAULT_SELECTED_ACTIVITIES,
    DEFAULT_SPLIT_STRATEGY,
    DEFAULT_TEST_SUBJECTS,
    DEFAULT_VAL_PERCENTAGE,
    DEFAULT_VAL_SUBJECTS,
    DEFAULT_WINDOW_OVERLAP,
    SharedConfig,
    build_or_load_loso_folds,
    class_names,
    prepare_cfg,
    reconcile_activity_config,
    repo_relative_path,
    split_indices_for_fold,
)


@dataclass(frozen=True)
class Config:
    dataset_ids: tuple[str, ...] = ("hhar", "wear", "harth", "hapt")
    datasets_dir: str = DEFAULT_DATASETS_DIR
    artifacts_dir: str = str(ARTIFACTS_ROOT)
    k_values: tuple[int, ...] = tuple(range(17))
    min_query_per_class: int = 1
    selected_activities: list[str] | None = DEFAULT_SELECTED_ACTIVITIES
    window_overlap: float = DEFAULT_WINDOW_OVERLAP
    val_subjects: int = DEFAULT_VAL_SUBJECTS
    test_subjects: int = DEFAULT_TEST_SUBJECTS
    seed: int = DEFAULT_SEED
    split_strategy: str = DEFAULT_SPLIT_STRATEGY
    val_percentage: float = DEFAULT_VAL_PERCENTAGE
    max_folds: int | None = None


RUN_CONFIG = Config()


def assess_class_counts(
    class_counts: Mapping[int, int],
    k_values: Sequence[int],
    min_query_per_class: int = 1,
) -> list[dict[str, Any]]:
    """Return one eligibility row per K without dropping any present class."""
    if not class_counts:
        raise ValueError("A subject must have at least one present class.")
    if int(min_query_per_class) < 1:
        raise ValueError("min_query_per_class must be at least 1.")

    rows: list[dict[str, Any]] = []
    normalized_counts = {
        int(class_id): int(count) for class_id, count in class_counts.items()
    }
    for raw_k in k_values:
        k = int(raw_k)
        if k < 0:
            raise ValueError(f"k_values must be non-negative, got {k}.")
        required = k + int(min_query_per_class)
        insufficient = {
            class_id: count
            for class_id, count in sorted(normalized_counts.items())
            if count < required
        }
        rows.append(
            {
                "k": k,
                "valid": not insufficient,
                "required_windows_per_class": required,
                "insufficient_class_counts": insufficient,
            }
        )
    return rows


def _counts_for_indices(
    session_df: pd.DataFrame,
    window_df: pd.DataFrame,
    indices: Sequence[int],
) -> dict[int, int]:
    subset = window_df.loc[list(indices), ["session_id"]].copy()
    session_meta = session_df[["session_id", "activity_id"]].drop_duplicates(
        "session_id"
    )
    merged = subset.merge(session_meta, on="session_id", how="left")
    if merged["activity_id"].isna().any():
        raise ValueError("Missing activity metadata in LOSO test split.")
    counts = merged["activity_id"].value_counts().sort_index()
    return {int(class_id): int(count) for class_id, count in counts.items()}


def _format_shots(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def run(config: Config = RUN_CONFIG) -> dict[str, Any]:
    k_values = tuple(dict.fromkeys(int(k) for k in config.k_values))
    if not k_values or any(k < 0 for k in k_values):
        raise ValueError("k_values must contain non-negative values.")

    detail_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    artifacts_dir = Path(config.artifacts_dir)
    for dataset_id_text in config.dataset_ids:
        dataset_id = WHARDatasetID(dataset_id_text)
        cfg = prepare_cfg(
            dataset_id=dataset_id,
            datasets_dir=Path(config.datasets_dir),
            selected_activities=config.selected_activities,
            window_overlap=config.window_overlap,
        )
        pre = PreProcessingPipeline(cfg)
        _raw_df, session_df, window_df = pre.run()
        reconcile_activity_config(cfg, session_df)
        names = class_names(cfg)

        shared_cfg = SharedConfig(
            dataset_id=dataset_id_text,
            datasets_dir=config.datasets_dir,
            selected_activities=config.selected_activities,
            window_overlap=config.window_overlap,
            val_subjects=config.val_subjects,
            test_subjects=config.test_subjects,
            seed=config.seed,
            split_strategy=config.split_strategy,
            val_percentage=config.val_percentage,
        )
        manifest_path = (
            artifacts_dir
            / "datasets"
            / dataset_id_text
            / "shared_splits"
            / "loso_subject_folds.json"
        )
        folds = build_or_load_loso_folds(
            session_df, window_df, shared_cfg, manifest_path
        )
        if config.max_folds is not None:
            folds = folds[: int(config.max_folds)]

        for fold in folds:
            split = split_indices_for_fold(session_df, window_df, fold)
            counts = _counts_for_indices(session_df, window_df, split.test_indices)
            subject_id = int(fold.test_subject_ids[0])
            assessed = assess_class_counts(
                counts, k_values, config.min_query_per_class
            )
            valid_shots = [int(row["k"]) for row in assessed if row["valid"]]
            invalid_shots = [int(row["k"]) for row in assessed if not row["valid"]]
            min_count = min(counts.values())
            max_supported_k = min_count - int(config.min_query_per_class)
            limiting_class_ids = sorted(
                class_id for class_id, count in counts.items() if count == min_count
            )

            subject_rows.append(
                {
                    "dataset_id": dataset_id_text,
                    "fold_id": fold.fold_id,
                    "subject_id": subject_id,
                    "num_present_classes": len(counts),
                    "min_class_windows": min_count,
                    "max_supported_k": max_supported_k,
                    "limiting_classes": json.dumps(
                        {
                            str(class_id): {
                                "name": names[class_id],
                                "windows": counts[class_id],
                            }
                            for class_id in limiting_class_ids
                        },
                        sort_keys=True,
                    ),
                    "valid_requested_shots": _format_shots(valid_shots),
                    "invalid_requested_shots": _format_shots(invalid_shots),
                }
            )

            for class_id, count in sorted(counts.items()):
                class_rows.append(
                    {
                        "dataset_id": dataset_id_text,
                        "fold_id": fold.fold_id,
                        "subject_id": subject_id,
                        "class_id": class_id,
                        "class_name": names[class_id],
                        "num_windows": count,
                        "max_supported_k": count
                        - int(config.min_query_per_class),
                    }
                )

            for row in assessed:
                insufficient = dict(row["insufficient_class_counts"])
                detail_rows.append(
                    {
                        "dataset_id": dataset_id_text,
                        "fold_id": fold.fold_id,
                        "subject_id": subject_id,
                        "k": int(row["k"]),
                        "valid": bool(row["valid"]),
                        "num_present_classes": len(counts),
                        "required_windows_per_class": int(
                            row["required_windows_per_class"]
                        ),
                        "insufficient_classes": json.dumps(
                            {
                                str(class_id): {
                                    "name": names[class_id],
                                    "available_windows": count,
                                }
                                for class_id, count in insufficient.items()
                            },
                            sort_keys=True,
                        ),
                    }
                )

    output_dir = artifacts_dir / "tables" / "shot_eligibility"
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "subject_by_k.csv"
    subject_path = output_dir / "subjects.csv"
    class_path = output_dir / "class_counts.csv"
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
    pd.DataFrame(subject_rows).to_csv(subject_path, index=False)
    pd.DataFrame(class_rows).to_csv(class_path, index=False)

    invalid_rows = [row for row in detail_rows if not bool(row["valid"])]
    config_payload = asdict(config)
    config_payload["datasets_dir"] = repo_relative_path(config.datasets_dir)
    config_payload["artifacts_dir"] = repo_relative_path(config.artifacts_dir)
    summary = {
        "config": config_payload,
        "num_subjects": len(subject_rows),
        "num_subject_k_checks": len(detail_rows),
        "num_invalid_subject_k_checks": len(invalid_rows),
        "all_requested_shots_valid": not invalid_rows,
        "subject_report": repo_relative_path(subject_path),
        "subject_by_k_report": repo_relative_path(detail_path),
        "class_count_report": repo_relative_path(class_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("dataset subject max-k valid-shots invalid-shots")
    for row in subject_rows:
        print(
            f"{row['dataset_id']:7} {row['subject_id']:>7} "
            f"{row['max_supported_k']:>5} "
            f"{row['valid_requested_shots'] or '-':>20} "
            f"{row['invalid_requested_shots'] or '-'}"
        )
    print(f"Reports written to {output_dir}")
    return summary

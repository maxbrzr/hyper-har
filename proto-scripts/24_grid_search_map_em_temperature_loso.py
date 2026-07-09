from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTO_SCRIPTS = ROOT / "proto-scripts"
if str(PROTO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PROTO_SCRIPTS))

MAP_EM_SCRIPT = PROTO_SCRIPTS / "09_eval_unlabeled_map_em_support_prototypes_loso.py"

DATASETS: dict[str, str] = {
    "HARTH": "harth",
    "HAPT": "hapt",
    "WEAR": "wear",
    "HHAR": "hhar",
}

DEFAULT_K_VALUES = tuple(range(1, 17))
DEFAULT_EM_TEMPERATURES = (0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
GRID_STAGE_NAME = "24_map_em_temperature_grid_search_loso"


def _parse_csv(values: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in values.split(",") if part.strip())


def _parse_k_values(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in _parse_csv(raw):
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(part))
    unique_values = tuple(dict.fromkeys(values))
    if any(k < 1 for k in unique_values):
        raise argparse.ArgumentTypeError("k values must be >= 1 for this sweep")
    return unique_values


def _parse_temperatures(raw: str) -> tuple[float, ...]:
    values = tuple(dict.fromkeys(float(part) for part in _parse_csv(raw)))
    if not values:
        raise argparse.ArgumentTypeError("at least one em_temperature is required")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("em_temperature values must be > 0")
    return values


def _artifact_safe_name(value: str) -> str:
    return value.replace(".", "p").replace("-", "m").replace("+", "p").replace(" ", "_")


def _load_module(script_path: Path) -> ModuleType:
    module_name = "_proto_eval_map_em_temperature_grid"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _selected_datasets(raw_names: Iterable[str]) -> tuple[tuple[str, str], ...]:
    selected: list[tuple[str, str]] = []
    for raw_name in raw_names:
        name = raw_name.upper()
        if name not in DATASETS:
            choices = ", ".join(DATASETS)
            raise ValueError(f"Unknown dataset {raw_name!r}; expected one of: {choices}")
        selected.append((name, DATASETS[name]))
    return tuple(selected)


def _cleanup_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _temperature_stage_name(base_stage_name: str, em_temperature: float) -> str:
    temp_name = _artifact_safe_name(f"{float(em_temperature):g}")
    return f"{base_stage_name}_emtemp_{temp_name}"


def _eval_dir_for_config(module: ModuleType, config: Any) -> Path:
    output_root = Path(module.resolve_output_root(config.output_root, config.dataset_id))
    active_transform = module._active_embedding_transform_name(  # noqa: SLF001
        config, config.backbone_source
    )
    effective_distance_metric = module.resolve_distance_metric(
        config.distance_metric, config.backbone_source
    )
    effective_normalize = (
        module._resolve_optional_bool(config.normalize_embeddings, False)  # noqa: SLF001
        and False
    )
    em_likelihood_variance_source = module._validate_em_likelihood_variance_source(  # noqa: SLF001
        config.em_likelihood_variance_source
    )
    em_responsibility_variance_source = (
        module._validate_em_responsibility_variance_source(  # noqa: SLF001
            config.em_responsibility_variance_source
        )
    )

    parts = [config.eval_stage_name]
    if config.separate_backbone_source_dir:
        parts.append(f"{config.backbone_source}_backbone")
    parts.append(effective_distance_metric)
    if active_transform != "none":
        parts.append(_artifact_safe_name(active_transform))
        parts.append("l2norm" if effective_normalize else "rawstats")
    if config.center_train_support_query:
        parts.append("centered")
    if em_likelihood_variance_source == "responsibility":
        parts.append("respvar")
        parts.append(
            f"varfloor_{_artifact_safe_name(f'{float(config.em_support_variance_floor):g}')}"
        )
    if em_responsibility_variance_source == "support":
        parts.append("diaglike")
    return output_root / "_".join(parts)


def _csv_k_values(csv_path: Path) -> set[int]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(csv_path, usecols=["k"])
    return {int(k) for k in df["k"].dropna().astype(int).tolist()}


def _completion_status(
    eval_dir: Path,
    k_values: tuple[int, ...],
    em_temperature: float,
) -> tuple[bool, str]:
    summary_path = eval_dir / "summary.json"
    overall_path = eval_dir / "overall_by_k_results.csv"
    if not summary_path.exists() or not overall_path.exists():
        return False, "missing summary or overall_by_k_results.csv"
    if summary_path.stat().st_size == 0 or overall_path.stat().st_size == 0:
        return False, "empty summary or overall_by_k_results.csv"

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"unreadable summary.json: {exc}"
    observed_temperature = float(summary.get("em_temperature", "nan"))
    if observed_temperature != float(em_temperature):
        return False, (
            f"summary has em_temperature={observed_temperature:g}, "
            f"expected {float(em_temperature):g}"
        )

    missing_k = sorted(set(k_values) - _csv_k_values(overall_path))
    if missing_k:
        preview = ",".join(str(k) for k in missing_k[:10])
        suffix = "..." if len(missing_k) > 10 else ""
        return False, f"missing k values: {preview}{suffix}"
    return True, "complete"


def _read_overall_rows(
    dataset_label: str,
    em_temperature: float,
    eval_dir: Path,
) -> pd.DataFrame:
    overall_path = eval_dir / "overall_by_k_results.csv"
    df = pd.read_csv(overall_path)
    df.insert(0, "dataset", dataset_label)
    df.insert(1, "em_temperature", float(em_temperature))
    df.insert(2, "eval_dir", str(eval_dir))
    return df


def _mean_std(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _summarize_grid(
    result_rows: list[pd.DataFrame],
    out_dir: Path,
    expected_datasets: tuple[tuple[str, str], ...],
    expected_k_values: tuple[int, ...],
) -> None:
    if not result_rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.concat(result_rows, ignore_index=True)
    for column in (
        "k",
        "macro_f1_mean",
        "macro_f1_trial_mean",
        "accuracy_mean",
        "accuracy_trial_mean",
        "num_trials",
        "num_subjects",
    ):
        if column in results_df.columns:
            results_df[column] = pd.to_numeric(results_df[column], errors="coerce")
    results_df["score_macro_f1"] = results_df.get(
        "macro_f1_mean", results_df["macro_f1_trial_mean"]
    )
    results_df["score_accuracy"] = results_df.get(
        "accuracy_mean", results_df["accuracy_trial_mean"]
    )
    results_df = results_df.sort_values(["em_temperature", "dataset", "k"])
    results_csv = out_dir / "temperature_grid_results_by_dataset_k.csv"
    results_df.to_csv(results_csv, index=False)

    per_dataset_rows: list[dict[str, Any]] = []
    for (temperature, dataset), group in results_df.groupby(
        ["em_temperature", "dataset"], sort=True
    ):
        macro_mean, macro_std = _mean_std(group["score_macro_f1"])
        acc_mean, acc_std = _mean_std(group["score_accuracy"])
        observed_k = sorted(int(k) for k in group["k"].dropna().astype(int).unique())
        per_dataset_rows.append(
            {
                "em_temperature": float(temperature),
                "dataset": str(dataset),
                "num_k_values": int(len(observed_k)),
                "k_values": json.dumps(observed_k),
                "macro_f1_mean_over_k": macro_mean,
                "macro_f1_std_over_k": macro_std,
                "accuracy_mean_over_k": acc_mean,
                "accuracy_std_over_k": acc_std,
                "num_trials": int(group.get("num_trials", pd.Series(dtype=float)).sum()),
                "num_subject_rows": int(
                    group.get("num_subjects", pd.Series(dtype=float)).sum()
                ),
            }
        )
    per_dataset_df = pd.DataFrame(per_dataset_rows).sort_values(
        ["em_temperature", "dataset"]
    )
    per_dataset_csv = out_dir / "temperature_grid_results_by_dataset.csv"
    per_dataset_df.to_csv(per_dataset_csv, index=False)

    expected_dataset_count = len(expected_datasets)
    expected_cell_count = expected_dataset_count * len(expected_k_values)
    summary_rows: list[dict[str, Any]] = []
    for temperature, group in per_dataset_df.groupby("em_temperature", sort=True):
        cell_group = results_df[results_df["em_temperature"] == float(temperature)]
        dataset_macro_mean, dataset_macro_std = _mean_std(
            group["macro_f1_mean_over_k"]
        )
        dataset_acc_mean, dataset_acc_std = _mean_std(group["accuracy_mean_over_k"])
        cell_macro_mean, cell_macro_std = _mean_std(cell_group["score_macro_f1"])
        observed_datasets = sorted(str(x) for x in group["dataset"].unique())
        summary_rows.append(
            {
                "em_temperature": float(temperature),
                "rank_metric": "macro_f1_mean_over_datasets_after_mean_over_k",
                "macro_f1_mean": dataset_macro_mean,
                "macro_f1_std_across_datasets": dataset_macro_std,
                "macro_f1_mean_over_dataset_k_cells": cell_macro_mean,
                "macro_f1_std_over_dataset_k_cells": cell_macro_std,
                "accuracy_mean": dataset_acc_mean,
                "accuracy_std_across_datasets": dataset_acc_std,
                "num_datasets": int(len(observed_datasets)),
                "datasets": json.dumps(observed_datasets),
                "num_dataset_k_cells": int(cell_group.shape[0]),
                "expected_dataset_k_cells": int(expected_cell_count),
                "is_complete": bool(
                    len(observed_datasets) == expected_dataset_count
                    and int(cell_group.shape[0]) == expected_cell_count
                ),
                "num_trials": int(cell_group.get("num_trials", pd.Series(dtype=float)).sum()),
                "num_subject_rows": int(
                    cell_group.get("num_subjects", pd.Series(dtype=float)).sum()
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["is_complete", "macro_f1_mean", "em_temperature"],
        ascending=[False, False, True],
    )
    summary_df.insert(0, "rank", range(1, len(summary_df) + 1))
    summary_csv = out_dir / "temperature_grid_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    complete_df = summary_df[summary_df["is_complete"]]
    best_pool = complete_df if not complete_df.empty else summary_df
    best_row = best_pool.iloc[0].to_dict()
    best_json = out_dir / "best_em_temperature.json"
    best_json.write_text(
        json.dumps(
            {
                "best_em_temperature": float(best_row["em_temperature"]),
                "selection_metric": best_row["rank_metric"],
                "macro_f1_mean": float(best_row["macro_f1_mean"]),
                "is_complete": bool(best_row["is_complete"]),
                "summary_csv": str(summary_csv),
                "per_dataset_csv": str(per_dataset_csv),
                "results_csv": str(results_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Grid search em_temperature for 09_eval_unlabeled_map_em_support_"
            "prototypes_loso.py across HARTH, HAPT, WEAR, HHAR and k=1..16."
        )
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated dataset names. Default: HARTH,HAPT,WEAR,HHAR.",
    )
    parser.add_argument(
        "--k-values",
        type=_parse_k_values,
        default=DEFAULT_K_VALUES,
        help="Comma-separated k values and/or ranges. Default: 1-16.",
    )
    parser.add_argument(
        "--em-temperatures",
        type=_parse_temperatures,
        default=DEFAULT_EM_TEMPERATURES,
        help=(
            "Comma-separated positive em_temperature values. "
            "Default: 0.1,0.2,0.5,0.75,1,1.5,2,3,5."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override the eval config device, e.g. cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--episodes-per-k",
        type=int,
        default=None,
        help="Override episodes_per_k for smoke tests or final sweeps.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Limit folds per run for smoke tests.",
    )
    parser.add_argument(
        "--skip-missing-folds",
        action="store_true",
        help="Skip folds whose CE checkpoints are missing.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Recompute jobs even when completed outputs already exist.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip completed dataset/temperature outputs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining jobs after a failure and report failures at the end.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned jobs without executing them.",
    )
    parser.add_argument(
        "--summary-dir",
        default=str(ROOT / "artifacts" / "proto_pipeline" / GRID_STAGE_NAME),
        help="Directory for aggregate grid-search CSV/JSON summaries.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    datasets = _selected_datasets(_parse_csv(args.datasets))
    k_values = tuple(int(k) for k in args.k_values)
    em_temperatures = tuple(float(value) for value in args.em_temperatures)
    summary_dir = Path(args.summary_dir)
    os.chdir(ROOT)

    module = _load_module(MAP_EM_SCRIPT)
    base_eval_stage_name = module.RUN_CONFIG.eval_stage_name
    total_jobs = len(datasets) * len(em_temperatures)
    print(
        f"Planned EM-temperature grid: {len(datasets)} datasets x "
        f"{len(em_temperatures)} temperatures x {len(k_values)} k-values "
        f"({k_values[0]}-{k_values[-1]})"
    )
    print(f"Temperatures: {em_temperatures}")

    failures: list[tuple[str, float, BaseException]] = []
    result_rows: list[pd.DataFrame] = []
    job_idx = 0

    for dataset_label, dataset_id in datasets:
        for em_temperature in em_temperatures:
            job_idx += 1
            stage_name = _temperature_stage_name(base_eval_stage_name, em_temperature)
            print(f"\n[{job_idx}/{total_jobs}] {dataset_label} :: em_temperature={em_temperature:g}")
            started_at = time.perf_counter()
            try:
                config_overrides: dict[str, Any] = {
                    "dataset_id": dataset_id,
                    "k_values": k_values,
                    "em_temperature": float(em_temperature),
                    "eval_stage_name": stage_name,
                }
                if args.device is not None:
                    config_overrides["device"] = args.device
                if args.episodes_per_k is not None:
                    config_overrides["episodes_per_k"] = int(args.episodes_per_k)
                if args.max_folds is not None:
                    config_overrides["max_folds"] = int(args.max_folds)
                if args.skip_missing_folds:
                    config_overrides["skip_missing_folds"] = True
                if args.force_rerun:
                    config_overrides["force_rerun"] = True

                config = replace(module.RUN_CONFIG, **config_overrides)
                eval_dir = _eval_dir_for_config(module, config)
                is_complete, completion_reason = _completion_status(
                    eval_dir, k_values, em_temperature
                )
                resume_enabled = not args.no_resume and not args.force_rerun
                print(f"  output: {eval_dir.relative_to(ROOT)}")
                print(f"  status: {completion_reason}")
                print(f"  em_iterations: {config.em_iterations}")
                print(
                    f"  unchanged config except: "
                    f"{json.dumps({key: value for key, value in config_overrides.items() if key != 'eval_stage_name'}, sort_keys=True)}"
                )
                if args.dry_run:
                    continue
                if is_complete and resume_enabled:
                    print("  action: skip completed output")
                else:
                    print("  action: run")
                    module.run(config)
                result_rows.append(
                    _read_overall_rows(dataset_label, em_temperature, eval_dir)
                )
                elapsed = time.perf_counter() - started_at
                print(f"  completed in {elapsed / 60.0:.1f} min")
            except BaseException as exc:
                failures.append((dataset_label, em_temperature, exc))
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                if not args.continue_on_error:
                    raise
            finally:
                _cleanup_accelerator_cache()

    if not args.dry_run:
        _summarize_grid(result_rows, summary_dir, datasets, k_values)
        config_path = summary_dir / "grid_search_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "base_script": str(MAP_EM_SCRIPT),
                    "base_run_config": asdict(module.RUN_CONFIG),
                    "datasets": [label for label, _dataset_id in datasets],
                    "dataset_ids": {label: dataset_id for label, dataset_id in datasets},
                    "k_values": list(k_values),
                    "em_temperatures": list(em_temperatures),
                    "summary_dir": str(summary_dir),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if failures:
        print("\nFailures:")
        for dataset_label, em_temperature, exc in failures:
            print(
                f"  - {dataset_label} :: em_temperature={em_temperature:g}: "
                f"{type(exc).__name__}: {exc}"
            )
        raise SystemExit(1)

    if args.dry_run:
        print("\nDry run completed; no jobs were executed.")
    else:
        print(f"\nGrid search summary written to: {summary_dir}")


if __name__ == "__main__":
    main()

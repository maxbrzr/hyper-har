from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
PROTO_SCRIPTS = ROOT / "proto-scripts"
if str(PROTO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PROTO_SCRIPTS))

DATASETS: dict[str, str] = {
    "HARTH": "harth",
    "HAPT": "hapt",
    "HHAR": "hhar",
    "WEAR": "wear",
}

DEFAULT_K_VALUES = tuple(range(33))

METHODS: tuple[dict[str, Any], ...] = (
    {
        "name": "supcon_support_prototypes",
        "script": "05_eval_supcon_support_prototypes_loso.py",
        "overrides": {},
    },
    {
        "name": "bayesian_support_prototypes",
        "script": "07_eval_bayesian_support_prototypes_loso.py",
        "overrides": {},
    },
    {
        "name": "pda_support_prototypes",
        "script": "16_eval_pda_support_prototypes_loso.py",
        "overrides": {},
    },
    {
        "name": "oftta_support",
        "script": "18_eval_oftta_support_loso.py",
        "overrides": {},
    },
    {
        "name": "logistic_linear_head_support",
        "script": "20_eval_logistic_linear_head_support_loso.py",
        "overrides": {},
    },
    {
        "name": "unlabeled_map_em_support_prototypes_centered",
        "script": "09_eval_unlabeled_map_em_support_prototypes_loso.py",
        "overrides": {"center_train_support_query": True},
    },
    {
        "name": "unlabeled_map_em_support_prototypes_uncentered",
        "script": "09_eval_unlabeled_map_em_support_prototypes_loso.py",
        "overrides": {"center_train_support_query": False},
    },
)


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
    if any(k < 0 for k in unique_values):
        raise argparse.ArgumentTypeError("k values must be non-negative")
    return unique_values


def _load_module(script_path: Path, module_key: str) -> ModuleType:
    module_name = f"_proto_eval_{module_key}"
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


def _selected_methods(raw_names: Iterable[str]) -> tuple[dict[str, Any], ...]:
    by_name = {method["name"]: method for method in METHODS}
    selected: list[dict[str, Any]] = []
    for name in raw_names:
        if name not in by_name:
            choices = ", ".join(by_name)
            raise ValueError(f"Unknown method {name!r}; expected one of: {choices}")
        selected.append(by_name[name])
    return tuple(selected)


def _cleanup_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        # Cache cleanup is best-effort only; the next run should not fail because of it.
        pass


def _artifact_safe_name(value: str) -> str:
    return value.replace(".", "p").replace("-", "m").replace("+", "p").replace(" ", "_")


def _required_final_files(eval_dir: Path) -> tuple[Path, ...]:
    return (
        eval_dir / "summary.json",
        eval_dir / "trial_results.csv",
        eval_dir / "subject_by_k_results.csv",
        eval_dir / "subject_episode_summary_by_k.csv",
        eval_dir / "overall_by_k_results.csv",
    )


def _csv_k_values(csv_path: Path) -> set[int]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "k" not in reader.fieldnames:
            return set()
        values: set[int] = set()
        for row in reader:
            raw_k = row.get("k")
            if raw_k is not None and raw_k != "":
                values.add(int(float(raw_k)))
        return values


def _summary_k_values(summary_path: Path) -> set[int]:
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return set()
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    values: set[int] = set()
    for row in summary.get("summary_by_k", []):
        if isinstance(row, dict) and row.get("k") is not None:
            values.add(int(row["k"]))
    return values


def _completion_status(eval_dir: Path, k_values: tuple[int, ...]) -> tuple[bool, str]:
    required_files = _required_final_files(eval_dir)
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        return False, f"missing final files: {', '.join(missing_files)}"

    empty_files = [path.name for path in required_files if path.stat().st_size == 0]
    if empty_files:
        return False, f"empty final files: {', '.join(empty_files)}"

    required_k = set(k_values)
    summary_k = _summary_k_values(eval_dir / "summary.json")
    overall_k = _csv_k_values(eval_dir / "overall_by_k_results.csv")
    available_k = summary_k & overall_k
    missing_k = sorted(required_k - available_k)
    if missing_k:
        preview = ",".join(str(k) for k in missing_k[:10])
        suffix = "..." if len(missing_k) > 10 else ""
        return False, f"missing k values: {preview}{suffix}"
    return True, "complete"


def _missing_k_values(eval_dir: Path, k_values: tuple[int, ...]) -> tuple[int, ...]:
    required_k = set(k_values)
    summary_k = _summary_k_values(eval_dir / "summary.json")
    overall_k = _csv_k_values(eval_dir / "overall_by_k_results.csv")
    return tuple(sorted(required_k - (summary_k & overall_k)))


def _has_mergeable_final_outputs(eval_dir: Path) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in _required_final_files(eval_dir))


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _row_k(row: dict[str, str]) -> int | None:
    raw_k = row.get("k")
    if raw_k is None or raw_k == "":
        return None
    return int(float(raw_k))


def _sort_rows(path: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    preferred = {
        "trial_results.csv": ("fold_id", "test_subject_id", "k", "episode"),
        "subject_by_k_results.csv": ("fold_id", "test_subject_id", "k"),
        "subject_episode_summary_by_k.csv": ("test_subject_id", "k"),
        "overall_by_k_results.csv": ("k",),
    }.get(path.name, ("k",))

    def key(row: dict[str, str]) -> tuple[Any, ...]:
        values: list[Any] = []
        for column in preferred:
            value = row.get(column, "")
            if column in {"test_subject_id", "k", "episode"} and value != "":
                values.append(int(float(value)))
            else:
                values.append(value)
        return tuple(values)

    return sorted(rows, key=key)


def _snapshot_mergeable_outputs(eval_dir: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"csv": {}, "summary": None}
    for path in _required_final_files(eval_dir):
        if path.suffix == ".csv":
            snapshot["csv"][path.name] = _read_csv_rows(path)
    summary_path = eval_dir / "summary.json"
    try:
        snapshot["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        snapshot["summary"] = None
    return snapshot


def _merge_fieldnames(*fieldname_sets: list[str]) -> list[str]:
    merged: list[str] = []
    for fieldnames in fieldname_sets:
        for fieldname in fieldnames:
            if fieldname not in merged:
                merged.append(fieldname)
    return merged


def _merge_csv_outputs(
    eval_dir: Path,
    snapshot: dict[str, Any],
    rerun_k_values: tuple[int, ...],
) -> None:
    rerun_k_set = set(rerun_k_values)
    for filename, (old_fields, old_rows) in snapshot.get("csv", {}).items():
        path = eval_dir / filename
        new_fields, new_rows = _read_csv_rows(path)
        merged_rows = [
            row for row in old_rows if (_row_k(row) is None or _row_k(row) not in rerun_k_set)
        ]
        merged_rows.extend(new_rows)
        merged_fields = _merge_fieldnames(old_fields, new_fields)
        _write_csv_rows(path, merged_fields, _sort_rows(path, merged_rows))


def _plot_merged_k_curve(eval_dir: Path) -> None:
    overall_csv = eval_dir / "overall_by_k_results.csv"
    _fields, rows = _read_csv_rows(overall_csv)
    rows = [row for row in rows if _row_k(row) is not None]
    if not rows:
        return
    rows = _sort_rows(overall_csv, rows)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    x_values = [int(float(row["k"])) for row in rows]
    y_column = "macro_f1_mean" if "macro_f1_mean" in rows[0] else "macro_f1_trial_mean"
    y_values = [float(row[y_column]) for row in rows]
    yerr_column = (
        "macro_f1_subject_ci95"
        if "macro_f1_subject_ci95" in rows[0]
        else "macro_f1_trial_ci95"
        if "macro_f1_trial_ci95" in rows[0]
        else None
    )
    yerr = [float(row[yerr_column]) for row in rows] if yerr_column else None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(x_values, y_values, yerr=yerr, marker="o", linewidth=2, capsize=4)
    ax.set_xlabel("K shots per activity")
    ax.set_ylabel("Macro F1")
    ax.set_title("Support Evaluation")
    ax.set_xticks(x_values)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(eval_dir / "k_shot_curve.png", dpi=180)
    plt.close(fig)


def _summary_rows_from_overall(overall_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for row in overall_rows:
        if _row_k(row) is None:
            continue
        summary_rows.append(
            {
                "k": int(float(row["k"])),
                "macro_f1_mean": float(row.get("macro_f1_trial_mean", row.get("macro_f1_mean", "nan"))),
                "macro_f1_std": float(row.get("macro_f1_trial_std", row.get("macro_f1_subject_std", "nan"))),
                "weighted_f1_mean": float(row.get("weighted_f1_trial_mean", "nan")),
                "weighted_f1_std": float(row.get("weighted_f1_trial_std", "nan")),
                "accuracy_mean": float(row.get("accuracy_trial_mean", row.get("accuracy_mean", "nan"))),
                "accuracy_std": float(row.get("accuracy_trial_std", row.get("accuracy_subject_std", "nan"))),
                "num_trials": int(float(row.get("num_trials", "0"))),
                "num_subjects": int(float(row.get("num_subjects", "0"))),
            }
        )
    return summary_rows


def _merge_summary_json(
    eval_dir: Path,
    snapshot: dict[str, Any],
    requested_k_values: tuple[int, ...],
) -> None:
    summary_path = eval_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        summary = snapshot.get("summary") or {}
    if not isinstance(summary, dict):
        summary = {}

    _overall_fields, overall_rows = _read_csv_rows(eval_dir / "overall_by_k_results.csv")
    _subject_fields, subject_rows = _read_csv_rows(eval_dir / "subject_by_k_results.csv")
    _subject_summary_fields, subject_summary_rows = _read_csv_rows(
        eval_dir / "subject_episode_summary_by_k.csv"
    )
    _trial_fields, trial_rows = _read_csv_rows(eval_dir / "trial_results.csv")

    summary["overall_by_k"] = [
        row for row in _sort_rows(eval_dir / "overall_by_k_results.csv", overall_rows)
    ]
    summary["summary_by_k"] = _summary_rows_from_overall(
        _sort_rows(eval_dir / "overall_by_k_results.csv", overall_rows)
    )
    summary["summary_by_subject"] = [
        row
        for row in _sort_rows(
            eval_dir / "subject_episode_summary_by_k.csv", subject_summary_rows
        )
    ]
    summary["num_trial_rows"] = len(trial_rows)
    summary["num_subject_k_rows"] = len(subject_rows)
    summary["num_subject_summary_rows"] = len(subject_summary_rows)
    summary.setdefault("resume_merge", {})
    summary["resume_merge"] = {
        "merged_by_runner": True,
        "requested_k_values": list(requested_k_values),
        "available_k_values": sorted(_csv_k_values(eval_dir / "overall_by_k_results.csv")),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _merge_partial_run_outputs(
    eval_dir: Path,
    snapshot: dict[str, Any],
    rerun_k_values: tuple[int, ...],
    requested_k_values: tuple[int, ...],
) -> None:
    _merge_csv_outputs(eval_dir, snapshot, rerun_k_values)
    _plot_merged_k_curve(eval_dir)
    _merge_summary_json(eval_dir, snapshot, requested_k_values)


def _eval_dir_for_config(module: ModuleType, method_name: str, config: Any) -> Path:
    output_root = Path(module.resolve_output_root(config.output_root, config.dataset_id))

    if method_name == "supcon_support_prototypes":
        effective_distance_metric = module.resolve_distance_metric(
            config.distance_metric, config.backbone_source
        )
        parts = [config.eval_stage_name]
        if config.separate_backbone_source_dir:
            parts.append(f"{config.backbone_source}_backbone")
        parts.append(effective_distance_metric)
        return output_root / "_".join(parts)

    if method_name == "bayesian_support_prototypes":
        active_transform = module._active_embedding_transform_name(  # noqa: SLF001
            config, config.backbone_source
        )
        effective_distance_metric = module.resolve_distance_metric(
            config.distance_metric, config.backbone_source
        )
        spherical_geometry = effective_distance_metric == "cosine"
        effective_normalize = (
            module._resolve_optional_bool(  # noqa: SLF001
                config.normalize_embeddings, spherical_geometry
            )
            and spherical_geometry
        )
        parts = [config.eval_stage_name]
        if config.separate_backbone_source_dir:
            parts.append(f"{config.backbone_source}_backbone")
        parts.append(effective_distance_metric)
        if active_transform != "none":
            parts.append(_artifact_safe_name(active_transform))
            parts.append("l2norm" if effective_normalize else "rawstats")
        return output_root / "_".join(parts)

    if method_name in {
        "unlabeled_map_em_support_prototypes_centered",
        "unlabeled_map_em_support_prototypes_uncentered",
    }:
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

    if method_name == "pda_support_prototypes":
        active_transform = module._active_embedding_transform_name(config)  # noqa: SLF001
        parts = [config.eval_stage_name, "ce_backbone", "cosine"]
        if active_transform != "none":
            parts.append(_artifact_safe_name(active_transform))
        parts.append("l2norm" if config.normalize_embeddings else "raw")
        parts.append(config.pseudo_labeler)
        if config.confidence_weighting:
            parts.append("confidence_weighted")
        if config.center_support_query:
            parts.append("centered")
        return output_root / "_".join(parts)

    if method_name == "oftta_support":
        parts = [
            config.eval_stage_name,
            "ce_backbone",
            f"edtn_d{float(config.edtn_decay):g}".replace(".", "p"),
            f"cap{int(config.support_capacity_per_class)}",
        ]
        if config.support_query_session_disjoint:
            parts.append("session_disjoint")
        return output_root / "_".join(parts)

    if method_name == "logistic_linear_head_support":
        parts = [
            config.eval_stage_name,
            "ce_backbone",
            f"C_{float(config.inverse_regularization):g}".replace(".", "p"),
        ]
        if config.standardize_features:
            parts.append("standardized")
        if config.support_query_session_disjoint:
            parts.append("session_disjoint")
        return output_root / "_".join(parts)

    raise ValueError(f"No eval-dir resolver for method {method_name!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the support-evaluation LOSO grid across datasets, methods, and k values."
        )
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated dataset names. Default: HARTH,HAPT,HHAR,WEAR.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(method["name"] for method in METHODS),
        help="Comma-separated method names. Use --list-methods to see valid names.",
    )
    parser.add_argument(
        "--k-values",
        type=_parse_k_values,
        default=DEFAULT_K_VALUES,
        help="Comma-separated k values and/or ranges. Default: 0-32.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override each eval config device, e.g. cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--episodes-per-k",
        type=int,
        default=None,
        help="Override episodes_per_k for every method.",
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
        help="Skip folds whose trained checkpoints are missing.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Recompute even when completed final outputs already exist.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip completed dataset/method outputs.",
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
        "--list-methods",
        action="store_true",
        help="Print valid method names and exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_methods:
        for method in METHODS:
            print(method["name"])
        return

    datasets = _selected_datasets(_parse_csv(args.datasets))
    methods = _selected_methods(_parse_csv(args.methods))
    k_values = tuple(int(k) for k in args.k_values)
    os.chdir(ROOT)

    total_jobs = len(datasets) * len(methods)
    print(
        f"Planned grid: {len(datasets)} datasets x {len(methods)} methods "
        f"x {len(k_values)} k-values ({k_values[0]}-{k_values[-1]})"
    )

    failures: list[tuple[str, str, BaseException]] = []
    job_idx = 0
    loaded_modules: dict[tuple[str, str], ModuleType] = {}

    for dataset_label, dataset_id in datasets:
        for method in methods:
            job_idx += 1
            method_name = str(method["name"])
            script_name = str(method["script"])
            overrides = dict(method["overrides"])
            print(f"\n[{job_idx}/{total_jobs}] {dataset_label} :: {method_name}")
            print(f"  script: proto-scripts/{script_name}")
            print(f"  k_values: {k_values}")
            if overrides:
                print(f"  overrides: {overrides}")

            started_at = time.perf_counter()
            try:
                module_key = (script_name, method_name)
                module = loaded_modules.get(module_key)
                if module is None:
                    module = _load_module(PROTO_SCRIPTS / script_name, method_name)
                    loaded_modules[module_key] = module
                config_overrides: dict[str, Any] = {
                    "dataset_id": dataset_id,
                    "k_values": k_values,
                    **overrides,
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
                eval_dir = _eval_dir_for_config(module, method_name, config)
                is_complete, completion_reason = _completion_status(eval_dir, k_values)
                resume_enabled = not args.no_resume and not args.force_rerun
                missing_k_values = _missing_k_values(eval_dir, k_values)
                can_merge_partial = (
                    resume_enabled
                    and not is_complete
                    and bool(missing_k_values)
                    and _has_mergeable_final_outputs(eval_dir)
                )
                run_k_values = missing_k_values if can_merge_partial else k_values
                snapshot: dict[str, Any] | None = None
                print(f"  output: {eval_dir.relative_to(ROOT)}")
                print(f"  status: {completion_reason}")
                if is_complete and resume_enabled:
                    print("  action: skip completed output")
                elif can_merge_partial:
                    print(f"  action: run missing k_values only: {run_k_values}")
                else:
                    print(f"  action: run k_values: {run_k_values}")
                if args.dry_run:
                    continue
                if is_complete and resume_enabled:
                    print("  skipped: completed output already exists")
                    continue
                if can_merge_partial:
                    snapshot = _snapshot_mergeable_outputs(eval_dir)
                    config = replace(config, k_values=run_k_values)
                module.run(config)
                if snapshot is not None:
                    _merge_partial_run_outputs(
                        eval_dir,
                        snapshot,
                        run_k_values,
                        k_values,
                    )
                elapsed = time.perf_counter() - started_at
                print(f"  completed in {elapsed / 60.0:.1f} min")
            except BaseException as exc:
                failures.append((dataset_label, method_name, exc))
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                if not args.continue_on_error:
                    raise
            finally:
                _cleanup_accelerator_cache()

    if failures:
        print("\nFailures:")
        for dataset_label, method_name, exc in failures:
            print(f"  - {dataset_label} :: {method_name}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    if args.dry_run:
        print("\nDry run completed; no jobs were executed.")
    else:
        print("\nAll requested support-evaluation jobs completed.")


if __name__ == "__main__":
    main()

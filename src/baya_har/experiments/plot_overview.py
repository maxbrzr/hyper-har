import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

from .common import repo_relative_path, resolve_output_root

ROOT = Path(__file__).resolve().parents[3]
FONT_CACHE_DIR = ROOT / "artifacts" / ".cache"
FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(ROOT / "artifacts" / ".matplotlib"),
)
os.environ.setdefault("XDG_CACHE_HOME", str(FONT_CACHE_DIR))


@dataclass(frozen=True)
class Config:
    output_root: str | None = None
    comparison_stage_name: str = "overview"
    results_source: str = "paper"  # "paper" or "live"
    source_records_csv: str | None = str(
        ROOT / "artifacts" / "results" / "paper_overview_results.csv"
    )
    dataset_ids: tuple[str, ...] = ("hhar", "wear", "harth", "hapt")

    # Explicit order requested by user.
    method_order: tuple[str, ...] = (
        "Original Classifier",
        "Prior Proto",
        "MAP-EM Proto (16-Shot)",
        "MAP Proto (16-Shot)",
    )
    require_all_methods_per_dataset: bool = False

    # Stage selection by prefix.
    baseline_prefix: str = "original"
    fixed_prefix: str = "prior"
    map_em_prefix: str = "map_em"
    bayesian_prefix: str = "map"

    # Prefer centered EM runs when both exist.
    map_em_prefer_contains: tuple[str, ...] = ("_centered",)
    k_value_for_proto_methods: int = 16

    # Visual settings.
    title: str = "LOSO Macro-F1 by Dataset"
    y_label: str = "Mean LOSO Macro-F1"
    x_label: str = ""
    figure_width: float = 12
    figure_height: float = 2.7
    dpi: int = 220
    ylim_low: float = 0.45
    ylim_high: float = 1.0
    bar_label_fmt: str = "{:.3f}"
    bar_label_color: str = "white"
    bar_label_fontsize: int = 9
    bar_label_inside_offset: float = 0.02
    color_baseline_ce: str = "#9467bd"
    color_fixed_proto: str = "#d62728"
    color_map_em: str = "#1f77b4"
    color_bayesian: str = "#2ca02c"
    axis_label_fontsize: int = 12


RUN_CONFIG = Config()


def _list_dataset_dirs(output_root: Path) -> list[Path]:
    return sorted(
        [p for p in output_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    )


def _choose_stage_dir(
    dataset_dir: Path,
    prefix: str,
    prefer_contains: tuple[str, ...] = (),
    forbidden_contains: tuple[str, ...] = (),
) -> Path | None:
    candidates = sorted(
        [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    )
    if forbidden_contains:
        candidates = [
            path
            for path in candidates
            if not any(token in path.name for token in forbidden_contains)
        ]
    if not candidates:
        return None

    if prefer_contains:
        preferred = [
            p for p in candidates if any(token in p.name for token in prefer_contains)
        ]
        if preferred:
            return preferred[0]
    return candidates[0]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_result(
    dataset_dir: Path, config: Config
) -> tuple[float, float, str] | None:
    stage_dir = _choose_stage_dir(dataset_dir, config.baseline_prefix)
    if stage_dir is None:
        return None
    summary_path = stage_dir / "summary.json"
    if not summary_path.exists():
        return None

    summary = _load_json(summary_path)
    folds = summary.get("folds", [])
    if not folds:
        return None
    values = pd.Series([float(row["test_macro_f1"]) for row in folds], dtype=float)
    return (
        float(values.mean()),
        float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "baseline",
    )


def _fixed_result(dataset_dir: Path, config: Config) -> tuple[float, float, str] | None:
    stage_dir = _choose_stage_dir(dataset_dir, config.fixed_prefix)
    if stage_dir is None:
        return None
    summary_path = stage_dir / "summary.json"
    if not summary_path.exists():
        return None

    summary = _load_json(summary_path)
    folds = summary.get("folds", [])
    if not folds:
        return None
    values = pd.Series([float(row["test_macro_f1"]) for row in folds], dtype=float)
    return (
        float(values.mean()),
        float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "fixed",
    )


def _k_result(
    dataset_dir: Path,
    prefix: str,
    method_key: str,
    k_value: int,
    prefer_contains: tuple[str, ...] = (),
    forbidden_contains: tuple[str, ...] = (),
) -> tuple[float, float, str] | None:
    stage_dir = _choose_stage_dir(
        dataset_dir,
        prefix,
        prefer_contains=prefer_contains,
        forbidden_contains=forbidden_contains,
    )
    if stage_dir is None:
        return None
    csv_path = stage_dir / "overall_by_k_results.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    required = {"k", "macro_f1_mean", "macro_f1_subject_std"}
    if not required.issubset(df.columns):
        return None

    row_df = df[df["k"].astype(int) == int(k_value)]
    if row_df.empty:
        return None
    row = row_df.iloc[0]
    return (
        float(row["macro_f1_mean"]),
        float(row["macro_f1_subject_std"]),
        f"{method_key}:k={int(row['k'])}",
    )


def _load_source_records(config: Config) -> pd.DataFrame | None:
    if config.results_source not in {"paper", "live"}:
        raise ValueError("results_source must be 'paper' or 'live'.")
    if config.results_source == "live":
        return None
    if not config.source_records_csv:
        raise ValueError("Paper results require source_records_csv.")
    source_path = Path(config.source_records_csv)
    if not source_path.exists():
        raise FileNotFoundError(f"Paper results CSV not found: {source_path}")

    records = pd.read_csv(source_path)
    required = {"dataset_id", "method_key", "n", "value", "fold_std"}
    if not required.issubset(records.columns):
        raise ValueError(f"Source records CSV is missing columns: {source_path}")
    return records


def _source_record_result(
    records: pd.DataFrame | None,
    dataset_id: str,
    method_key: str,
    n_value: int,
) -> tuple[float, float, str] | None:
    if records is None:
        return None
    rows = records[
        (records["dataset_id"].astype(str).str.lower() == dataset_id.lower())
        & (records["method_key"].astype(str) == method_key)
        & (records["n"].astype(int) == int(n_value))
    ]
    if rows.empty:
        return None
    row = rows.iloc[0]
    meta = (
        "baseline"
        if method_key == "original"
        else f"{method_key}:n={int(n_value)}:source22"
    )
    return (
        float(row["value"]),
        float(row["fold_std"]),
        meta,
    )


def _collect(config: Config) -> tuple[pd.DataFrame, list[str]]:
    output_root = resolve_output_root(config.output_root, dataset_id="")
    source_records = _load_source_records(config)

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    source_specs = {
        "Original Classifier": ("original", 0),
        "Prior Proto": ("fixed", config.k_value_for_proto_methods),
        "MAP-EM Proto (16-Shot)": (
            "map_em_centered",
            config.k_value_for_proto_methods,
        ),
        "MAP Proto (16-Shot)": ("bayesian", config.k_value_for_proto_methods),
    }
    for dataset_id in sorted(config.dataset_ids):
        dataset_dir = output_root / dataset_id
        if config.results_source == "paper":
            method_results = {
                method_name: _source_record_result(
                    source_records,
                    dataset_id,
                    source_key,
                    source_k,
                )
                for method_name, (source_key, source_k) in source_specs.items()
            }
        elif not dataset_dir.exists():
            warnings.append(f"[{dataset_id}] dataset dir missing")
            continue
        else:
            method_results = {
                "Original Classifier": _baseline_result(dataset_dir, config),
                "Prior Proto": _fixed_result(dataset_dir, config),
                "MAP-EM Proto (16-Shot)": _k_result(
                    dataset_dir,
                    config.map_em_prefix,
                    method_key="map_em",
                    k_value=config.k_value_for_proto_methods,
                    prefer_contains=config.map_em_prefer_contains,
                ),
                "MAP Proto (16-Shot)": _k_result(
                    dataset_dir,
                    config.bayesian_prefix,
                    method_key="bayesian",
                    k_value=config.k_value_for_proto_methods,
                    forbidden_contains=("map_em",),
                ),
            }

        present_count = sum(result is not None for result in method_results.values())
        if present_count == 0:
            continue

        for method_name in config.method_order:
            result = method_results[method_name]
            if result is None:
                warnings.append(f"[{dataset_id}] missing method: {method_name}")
                continue
            macro_f1, fold_std, meta = result
            rows.append(
                {
                    "dataset": dataset_id,
                    "method": method_name,
                    "macro_f1": macro_f1,
                    "fold_std": fold_std,
                    "meta": meta,
                }
            )

    if not rows:
        raise RuntimeError(f"No method results found for source '{config.results_source}'.")

    df = pd.DataFrame(rows)
    df["macro_f1"] = df["macro_f1"].astype(float).round(3)
    df["fold_std"] = df["fold_std"].astype(float).round(3)
    if config.require_all_methods_per_dataset:
        counts = df.groupby("dataset")["method"].nunique()
        keep = set(
            counts[counts == len(config.method_order)].index.astype(str).tolist()
        )
        dropped = sorted(set(df["dataset"].astype(str).tolist()) - keep)
        for dataset_id in dropped:
            warnings.append(
                f"[{dataset_id}] dropped from plot (missing one or more methods)"
            )
        df = df[df["dataset"].astype(str).isin(keep)].copy()
        if df.empty:
            raise RuntimeError("No datasets with complete method results were found.")
    return df, warnings


def _add_bar_label(ax: plt.Axes, x: float, height: float, config: Config) -> None:
    y = max(config.ylim_low + 0.01, height - config.bar_label_inside_offset)
    ax.text(
        x,
        y,
        config.bar_label_fmt.format(height),
        ha="center",
        va="top",
        color=config.bar_label_color,
        fontsize=config.bar_label_fontsize,
        fontweight="bold",
        zorder=30,
        clip_on=False,
    )


def _add_delta_label(ax: plt.Axes, x: float, bar_bottom: float, delta: float) -> None:
    ax.text(
        x,
        bar_bottom + 0.006,
        f"{delta:+.3f}",
        ha="center",
        va="bottom",
        color="white",
        fontsize=8,
        fontweight="bold",
        zorder=35,
        clip_on=False,
    )


def _plot(df: pd.DataFrame, config: Config, output_root: Path) -> dict[str, str]:
    out_dir = output_root.parent / "figures" / config.comparison_stage_name
    out_dir.mkdir(parents=True, exist_ok=True)

    preferred_order = [x.lower() for x in config.dataset_ids]
    present = {x.lower(): x for x in df["dataset"].unique().tolist()}
    dataset_order = [present[k] for k in preferred_order if k in present]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(
        figsize=(config.figure_width, config.figure_height), dpi=config.dpi
    )

    pivot_mean = df.pivot(index="dataset", columns="method", values="macro_f1").reindex(
        index=dataset_order, columns=list(config.method_order)
    )
    pivot_std = df.pivot(index="dataset", columns="method", values="fold_std").reindex(
        index=dataset_order, columns=list(config.method_order)
    )

    n_datasets = len(dataset_order)
    n_methods = len(config.method_order)
    x = np.arange(n_datasets, dtype=float)
    total_width = 0.86
    bar_width = total_width / n_methods
    offsets = (np.arange(n_methods, dtype=float) - (n_methods - 1) / 2.0) * bar_width
    method_colors = {
        "Original Classifier": config.color_baseline_ce,
        "Prior Proto": config.color_fixed_proto,
        "MAP-EM Proto (16-Shot)": config.color_map_em,
        "MAP Proto (16-Shot)": config.color_bayesian,
    }

    for j, method in enumerate(config.method_order):
        means = pivot_mean[method].to_numpy(dtype=float)
        stds = pivot_std[method].to_numpy(dtype=float)
        centers = x + offsets[j]
        color = method_colors[method]
        legend_shown = False

        for i in range(n_datasets):
            mean_val = means[i]
            std_val = stds[i]
            if not np.isfinite(mean_val):
                continue
            label = method if not legend_shown else None
            ax.bar(
                [centers[i]],
                [mean_val],
                width=bar_width * 0.96,
                color=color,
                label=label,
                zorder=3,
            )
            legend_shown = True
            if np.isfinite(std_val):
                ax.errorbar(
                    x=[centers[i]],
                    y=[mean_val],
                    yerr=[[std_val], [std_val]],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.3,
                    capsize=3,
                    capthick=1.3,
                    zorder=10,
                    clip_on=False,
                )
            _add_bar_label(ax, centers[i], float(mean_val), config)
            if method in {
                "MAP-EM Proto (16-Shot)",
                "MAP Proto (16-Shot)",
            }:
                baseline_fixed = pivot_mean.loc[dataset_order[i], "Prior Proto"]
                if np.isfinite(baseline_fixed):
                    delta = float(mean_val - float(baseline_fixed))
                    _add_delta_label(ax, centers[i], float(config.ylim_low), delta)

    ax.set_xticks(x)
    ax.set_xticklabels([dataset.upper() for dataset in dataset_order])
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontweight("bold")
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.set_xlabel(config.x_label, fontsize=config.axis_label_fontsize)
    ax.set_ylabel(
        config.y_label,
        fontsize=config.axis_label_fontsize,
        fontweight="bold",
    )
    finite_error_tops = (pivot_mean + pivot_std).to_numpy(dtype=float)
    finite_error_tops = finite_error_tops[np.isfinite(finite_error_tops)]
    ylim_high = float(config.ylim_high)
    if finite_error_tops.size:
        ylim_high = max(ylim_high, float(np.max(finite_error_tops)) + 0.02)
    ax.set_ylim(config.ylim_low, ylim_high)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=len(labels),
            frameon=False,
            columnspacing=1.6,
            handletextpad=0.6,
            prop={"weight": "bold"},
        )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

    png_path = out_dir / "grouped_macro_f1_loso.png"
    pdf_path = out_dir / "grouped_macro_f1_loso.pdf"
    fig.savefig(png_path)
    pdf_metadata = (
        {"CreationDate": datetime(2026, 7, 22, 9, 48, 42)}
        if config.results_source == "paper"
        else None
    )
    fig.savefig(pdf_path, metadata=pdf_metadata)
    plt.close(fig)

    return {
        "png": repo_relative_path(png_path),
        "pdf": repo_relative_path(pdf_path),
    }


def run(config: Config) -> dict[str, Any]:
    output_root = resolve_output_root(config.output_root, dataset_id="")
    df, warnings = _collect(config)
    paths = _plot(df, config, output_root)

    out_dir = output_root.parent / "figures" / config.comparison_stage_name
    table_path = out_dir / "grouped_macro_f1_loso_values.csv"
    df.to_csv(table_path, index=False)

    config_payload = asdict(config)
    if config_payload.get("source_records_csv"):
        config_payload["source_records_csv"] = repo_relative_path(
            config_payload["source_records_csv"]
        )
    summary = {
        "config": config_payload,
        "output_root": str(output_root),
        "num_rows": int(df.shape[0]),
        "datasets": sorted(df["dataset"].unique().tolist()),
        "methods": list(config.method_order),
        "plot_png": paths["png"],
        "plot_pdf": paths["pdf"],
        "values_csv": repo_relative_path(table_path),
        "warnings": warnings,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for message in warnings:
        print(f"[warn] {message}")
    print(f"Saved plot: {paths['png']}")
    print(f"Saved values: {table_path}")
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

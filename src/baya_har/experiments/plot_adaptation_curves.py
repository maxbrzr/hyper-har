import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

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
class MethodSpec:
    key: str
    label: str
    stage_prefix: str
    color: str
    marker: str = "o"
    linestyle: str = "-"
    required_contains: tuple[str, ...] = ()
    forbidden_contains: tuple[str, ...] = ()
    summary_equals: tuple[tuple[str, str], ...] = ()
    is_fixed_baseline: bool = False


@dataclass(frozen=True)
class Config:
    output_root: str | None = None
    comparison_stage_name: str = "adaptation_curves"
    results_source: str = "paper"  # "paper" or "live"
    source_records_csv: str | None = str(
        ROOT / "artifacts" / "results" / "paper_results.csv"
    )
    mode: str = "all"  # "supervised", "unsupervised", "combined", or "all"
    metric: str = "macro_f1"

    dataset_ids: tuple[str, ...] = ("hhar", "wear", "harth", "hapt")
    dataset_titles: tuple[str, ...] = ("HHAR", "WEAR", "HARTH", "HAPT")
    max_k: int = 16
    k_values: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16)
    x_label_k_values: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 16)
    k0_plot_value: float = 0.5

    fixed_prefix: str = "prior"
    support_prefix: str = "protonet"
    bayesian_prefix: str = "map"
    map_em_prefix: str = "map_em"
    pda_prefix: str = "pda"
    oftta_prefix: str = "oftta"
    logistic_prefix: str = "logistic"

    figure_width: float = 12.0
    figure_height: float = 4
    dpi: int = 240
    y_axis_mode: str = "auto"  # "auto" or "zero"
    y_axis_padding: float = 0.025
    y_axis_min_floor: float = 0.0
    y_axis_max_ceiling: float = 1.02
    legend_y: float = 1.12
    line_alpha: float = 0.78
    fixed_line_alpha: float = 0.65
    show_markers: bool = False
    force_all_methods_start_at_n0: bool = True


RUN_CONFIG = Config()


def _method_specs(config: Config) -> tuple[MethodSpec, ...]:
    return _method_specs_for_mode(config, config.mode)


def _method_specs_for_mode(config: Config, mode: str) -> tuple[MethodSpec, ...]:
    if mode == "supervised":
        return (
            MethodSpec(
                key="fixed",
                label="Prior Proto\n(ours)",
                stage_prefix=config.fixed_prefix,
                color="#d62728",
                marker="",
                linestyle="--",
                is_fixed_baseline=True,
            ),
            MethodSpec(
                key="bayesian",
                label="MAP Proto\n(ours)",
                stage_prefix=config.bayesian_prefix,
                color="#2ca02c",
                marker="s",
                forbidden_contains=("map_em",),
            ),
            MethodSpec(
                key="support",
                label="ProtoNet\n(baseline)",
                stage_prefix=config.support_prefix,
                color="#ff7f0e",
                marker="o",
            ),
            MethodSpec(
                key="logistic",
                label="Logistic Probe\n(baseline)",
                stage_prefix=config.logistic_prefix,
                color="#9467bd",
                marker="D",
            ),
        )
    if mode == "unsupervised":
        return (
            MethodSpec(
                key="fixed",
                label="Prior Proto\n(ours)",
                stage_prefix=config.fixed_prefix,
                color="#d62728",
                marker="",
                linestyle="--",
                is_fixed_baseline=True,
            ),
            MethodSpec(
                key="map_em_centered",
                label="MAP-EM Proto\n(ours)",
                stage_prefix=config.map_em_prefix,
                color="#1f77b4",
                marker="s",
                required_contains=("_centered",),
                summary_equals=(
                    ("em_likelihood_variance_source", "fixed"),
                    ("em_responsibility_variance_source", "fixed"),
                ),
            ),
            MethodSpec(
                key="oftta",
                label="OFTTA\n(baseline)",
                stage_prefix=config.oftta_prefix,
                color="#7f7f7f",
                marker="o",
            ),
            MethodSpec(
                key="pda",
                label="PDA\n(baseline)",
                stage_prefix=config.pda_prefix,
                color="#17becf",
                marker="D",
                required_contains=("classifier_confidence",),
                forbidden_contains=("mcd",),
            ),
        )
    raise ValueError("mode must be 'supervised', 'unsupervised', 'combined', or 'all'.")


def _metric_mean_column(metric: str) -> str:
    return f"{metric}_mean"


def _fixed_metric_key(metric: str) -> str:
    return f"test_{metric}"


def _metric_label(metric: str) -> str:
    if metric == "macro_f1":
        return "Mean LOSO Macro-F1"
    return metric.replace("_", " ").title()


def _choose_stage_dir(
    dataset_dir: Path,
    spec: MethodSpec,
) -> Path | None:
    candidates = sorted(
        [
            p
            for p in dataset_dir.iterdir()
            if p.is_dir() and p.name.startswith(spec.stage_prefix)
        ]
    )
    if spec.required_contains:
        candidates = [
            p
            for p in candidates
            if all(token in p.name for token in spec.required_contains)
        ]
    if spec.forbidden_contains:
        candidates = [
            p
            for p in candidates
            if not any(token in p.name for token in spec.forbidden_contains)
        ]
    if spec.summary_equals:
        filtered = []
        for candidate in candidates:
            summary_path = candidate / "summary.json"
            if not summary_path.exists():
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            summary_config = dict(summary.get("config", {}))
            matches = True
            for key, expected in spec.summary_equals:
                value = summary.get(key, summary_config.get(key))
                if str(value) != str(expected):
                    matches = False
                    break
            if matches:
                filtered.append(candidate)
        candidates = filtered
    return candidates[0] if candidates else None


def _load_curve(
    dataset_dir: Path,
    spec: MethodSpec,
    metric: str,
    max_k: int,
) -> tuple[pd.DataFrame | None, Path | None]:
    stage_dir = _choose_stage_dir(dataset_dir, spec)
    if stage_dir is None:
        return None, None
    csv_path = stage_dir / "overall_by_k_results.csv"
    if not csv_path.exists():
        return None, stage_dir
    df = pd.read_csv(csv_path)
    mean_col = _metric_mean_column(metric)
    if "k" not in df.columns or mean_col not in df.columns:
        return None, stage_dir
    out = df[["k", mean_col]].copy()
    out["k"] = out["k"].astype(int)
    out["value"] = out[mean_col].astype(float)
    out = out[out["k"] <= int(max_k)]
    return out[["k", "value"]].sort_values("k"), stage_dir


def _load_source_records(config: Config) -> pd.DataFrame | None:
    if config.results_source not in {"paper", "live"}:
        raise ValueError("results_source must be 'paper' or 'live'.")
    if config.results_source == "live":
        return None
    if not config.source_records_csv:
        raise ValueError("Paper results require source_records_csv.")
    csv_path = Path(config.source_records_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Paper results CSV not found: {csv_path}")

    records = pd.read_csv(csv_path)
    required = {"dataset_id", "method_key", "stage_dir", "n", "value"}
    if not required.issubset(records.columns):
        raise ValueError(f"Source records CSV is missing columns: {csv_path}")
    return records


def _load_curve_from_source_records(
    records: pd.DataFrame | None,
    dataset_id: str,
    spec: MethodSpec,
    max_k: int,
) -> tuple[pd.DataFrame, Path | None] | None:
    if records is None:
        return None
    rows = records[
        (records["dataset_id"].astype(str).str.lower() == dataset_id.lower())
        & (records["method_key"].astype(str) == spec.key)
        & (records["n"].astype(int) <= int(max_k))
    ].copy()
    if rows.empty:
        return None
    rows = rows.drop_duplicates(subset=["n"], keep="first").copy()
    rows["k"] = rows["n"].astype(int)
    rows["value"] = rows["value"].astype(float)
    stage_values = [str(x) for x in rows["stage_dir"].dropna().unique().tolist()]
    stage_dir = Path(stage_values[0]) if stage_values else None
    return rows[["k", "value"]].sort_values("k"), stage_dir


def _load_fixed_value(
    dataset_dir: Path,
    spec: MethodSpec,
    metric: str,
) -> tuple[float | None, Path | None]:
    stage_dir = _choose_stage_dir(dataset_dir, spec)
    if stage_dir is None:
        return None, None
    summary_path = stage_dir / "summary.json"
    if not summary_path.exists():
        return None, stage_dir
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_key = f"mean_test_{metric}"
    if summary_key in summary:
        return float(summary[summary_key]), stage_dir
    fold_key = _fixed_metric_key(metric)
    folds = summary.get("folds", [])
    values = [float(row[fold_key]) for row in folds if fold_key in row]
    if not values:
        return None, stage_dir
    return float(np.mean(values)), stage_dir


def _fixed_curve_rows(
    value: float, k_values: tuple[int, ...], max_k: int
) -> pd.DataFrame:
    ks = [int(k) for k in k_values if int(k) <= int(max_k)]
    return pd.DataFrame({"k": ks, "value": [float(value)] * len(ks)})


def _force_k0_to_fixed(
    curve_df: pd.DataFrame | None,
    fixed_value: float | None,
) -> pd.DataFrame | None:
    if curve_df is None or fixed_value is None:
        return curve_df
    df = curve_df.copy()
    df = df[df["k"].astype(int) != 0]
    k0 = pd.DataFrame([{"k": 0, "value": float(fixed_value)}])
    return pd.concat([k0, df], ignore_index=True).sort_values("k")


def _with_plot_k(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    out = df.copy()
    out["plot_k"] = out["k"].astype(float)
    out.loc[out["plot_k"] <= 0, "plot_k"] = float(config.k0_plot_value)
    return out


def _append_records(
    records: list[dict[str, Any]],
    dataset_id: str,
    dataset_title: str,
    spec: MethodSpec,
    stage_dir: Path | None,
    df: pd.DataFrame,
) -> None:
    for row in df.to_dict(orient="records"):
        records.append(
            {
                "dataset_id": dataset_id,
                "dataset_title": dataset_title,
                "method_key": spec.key,
                "method": spec.label,
                "stage_dir": "" if stage_dir is None else repo_relative_path(stage_dir),
                "n": int(row["k"]),
                "value": float(row["value"]),
            }
        )


def _axis_ylim(values: list[float], config: Config) -> tuple[float, float]:
    if config.y_axis_mode == "zero":
        return 0.0, float(config.y_axis_max_ceiling)
    if config.y_axis_mode != "auto":
        raise ValueError("y_axis_mode must be 'auto' or 'zero'.")
    if not values:
        return 0.0, 1.0
    low = max(
        float(config.y_axis_min_floor),
        min(values) - float(config.y_axis_padding),
    )
    high = min(
        float(config.y_axis_max_ceiling),
        max(values) + float(config.y_axis_padding),
    )
    if high - low < 0.08:
        mid = (low + high) / 2.0
        low = max(float(config.y_axis_min_floor), mid - 0.04)
        high = min(float(config.y_axis_max_ceiling), mid + 0.04)
    return low, high


def _plot_mode_on_axes(
    axes_flat: Any,
    mode: str,
    config: Config,
    output_root: Path,
    show_dataset_titles: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    specs = _method_specs_for_mode(config, mode)
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    source_records = _load_source_records(config)

    for ax, dataset_id, dataset_title in zip(
        axes_flat,
        config.dataset_ids,
        config.dataset_titles,
    ):
        dataset_dir = output_root / dataset_id
        y_values: list[float] = []
        if config.results_source == "live" and not dataset_dir.exists():
            warnings.append(f"[{dataset_id}] dataset dir missing")
            ax.set_title(dataset_title, fontweight="bold")
            ax.text(
                0.5,
                0.5,
                "dataset missing",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            continue

        fixed_spec = next((spec for spec in specs if spec.is_fixed_baseline), None)
        fixed_value: float | None = None
        if source_records is not None and fixed_spec is not None:
            source_fixed = _load_curve_from_source_records(
                source_records, dataset_id, fixed_spec, config.max_k
            )
            if source_fixed is not None:
                fixed_df, _fixed_stage_dir = source_fixed
                fixed_rows = fixed_df[fixed_df["k"].astype(int) == 0]
                if not fixed_rows.empty:
                    fixed_value = float(fixed_rows.iloc[0]["value"])
        elif fixed_spec is not None:
            fixed_value, _fixed_stage_dir = _load_fixed_value(
                dataset_dir,
                fixed_spec,
                config.metric,
            )

        for spec in specs:
            source_curve = _load_curve_from_source_records(
                source_records, dataset_id, spec, config.max_k
            )
            if source_curve is not None:
                df, stage_dir = source_curve
            elif config.results_source == "paper":
                warnings.append(f"[{dataset_id}] missing paper result: {spec.label}")
                continue
            elif spec.is_fixed_baseline:
                value, stage_dir = _load_fixed_value(dataset_dir, spec, config.metric)
                if value is None:
                    warnings.append(f"[{dataset_id}] missing {spec.label}")
                    continue
                df = _fixed_curve_rows(value, config.k_values, config.max_k)
            else:
                df, stage_dir = _load_curve(
                    dataset_dir,
                    spec,
                    config.metric,
                    config.max_k,
                )
                if df is None or df.empty:
                    warnings.append(f"[{dataset_id}] missing {spec.label}")
                    continue
                if config.force_all_methods_start_at_n0:
                    df = _force_k0_to_fixed(df, fixed_value)
                else:
                    df = df[df["k"].astype(int) != 0]
            df = _with_plot_k(df, config)

            y_values.extend(float(v) for v in df["value"].tolist())
            sns.lineplot(
                data=df,
                x="plot_k",
                y="value",
                ax=ax,
                color=spec.color,
                marker=spec.marker if config.show_markers and spec.marker else None,
                linestyle=spec.linestyle,
                linewidth=2.0 if not spec.is_fixed_baseline else 1.7,
                markersize=(
                    5.0 if config.show_markers and not spec.is_fixed_baseline else 0.0
                ),
                estimator=None,
                errorbar=None,
                legend=False,
                alpha=(
                    float(config.fixed_line_alpha)
                    if spec.is_fixed_baseline
                    else float(config.line_alpha)
                ),
            )
            _append_records(records, dataset_id, dataset_title, spec, stage_dir, df)

        ax.set_title(
            dataset_title if show_dataset_titles else "",
            fontweight="bold",
        )
        ax.set_xscale("log")
        ax.set_xlim(float(config.k0_plot_value) * 0.9, int(config.max_k) * 1.08)
        tick_k_values = [int(k) for k in config.k_values if int(k) <= int(config.max_k)]
        labeled_k_values = {
            int(k) for k in config.x_label_k_values if int(k) <= int(config.max_k)
        }
        tick_values = [
            float(config.k0_plot_value) if int(k) == 0 else int(k)
            for k in tick_k_values
        ]
        tick_labels = [
            str(int(k)) if int(k) in labeled_k_values else "" for k in tick_k_values
        ]
        ax.set_xticks(tick_values)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(*_axis_ylim(y_values, config))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xlabel("")
        if ax is axes_flat[0]:
            ax.set_ylabel(_metric_label(config.metric), fontweight="bold")
        else:
            ax.set_ylabel("")

    return records, warnings


def _legend_handles(
    specs: tuple[MethodSpec, ...],
    config: Config,
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=spec.color,
            marker=spec.marker if config.show_markers and spec.marker else None,
            linestyle=spec.linestyle,
            linewidth=2.0 if not spec.is_fixed_baseline else 1.7,
            markersize=(
                5.0 if config.show_markers and not spec.is_fixed_baseline else 0.0
            ),
            alpha=(
                float(config.fixed_line_alpha)
                if spec.is_fixed_baseline
                else float(config.line_alpha)
            ),
            label=spec.label,
        )
        for spec in specs
    ]


def _add_horizontal_bracket(
    fig: plt.Figure,
    x0: float,
    x1: float,
    y: float,
    tick: float = 0.018,
) -> None:
    bracket_kwargs = {
        "transform": fig.transFigure,
        "color": "#2f2f2f",
        "linewidth": 1.4,
        "alpha": 0.75,
    }
    fig.add_artist(Line2D([x0, x1], [y, y], **bracket_kwargs))
    fig.add_artist(Line2D([x0, x0], [y, y - tick], **bracket_kwargs))
    fig.add_artist(Line2D([x1, x1], [y, y - tick], **bracket_kwargs))


def _axes_span(axes_group: Any, pad: float = 0.01) -> tuple[float, float]:
    positions = [ax.get_position() for ax in axes_group]
    x0 = max(0.0, min(pos.x0 for pos in positions) - pad)
    x1 = min(1.0, max(pos.x1 for pos in positions) + pad)
    return x0, x1


def _run_single_mode(config: Config) -> dict[str, Any]:
    output_root = resolve_output_root(config.output_root, dataset_id="")
    out_dir = output_root.parent / "figures" / config.comparison_stage_name
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _method_specs(config)

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(
        1,
        len(config.dataset_ids),
        figsize=(config.figure_width, config.figure_height),
        dpi=int(config.dpi),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    axes_flat = axes[0]
    records, warnings = _plot_mode_on_axes(axes_flat, config.mode, config, output_root)

    handles = _legend_handles(specs, config)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, float(config.legend_y)),
        ncol=len(handles),
        frameon=False,
        prop={"weight": "bold"},
    )
    fig.supxlabel("N Shots Per Activity Class (log scale)", y=0.02, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.98))

    records_df = pd.DataFrame(records)
    records_csv = out_dir / f"{config.mode}_{config.metric}_grid_values.csv"
    records_df.to_csv(records_csv, index=False)
    out_path = out_dir / f"{config.mode}_{config.metric}_n0_to_{config.max_k}_grid.png"
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(out_path, dpi=int(config.dpi))
    fig.savefig(pdf_path)
    plt.close(fig)

    config_payload = asdict(config)
    if config_payload.get("source_records_csv"):
        config_payload["source_records_csv"] = repo_relative_path(
            config_payload["source_records_csv"]
        )
    summary = {
        "config": config_payload,
        "mode": config.mode,
        "metric": config.metric,
        "output_path": repo_relative_path(out_path),
        "pdf_path": repo_relative_path(pdf_path),
        "records_csv": repo_relative_path(records_csv),
        "warnings": warnings,
        "methods": [asdict(spec) for spec in specs],
        "num_records": int(records_df.shape[0]),
    }
    (out_dir / f"{config.mode}_{config.metric}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    for warning in warnings:
        print(f"[warn] {warning}")
    print(f"[plot] {config.mode} {config.metric}: {out_path}")
    return summary


def _run_combined(config: Config) -> dict[str, Any]:
    output_root = resolve_output_root(config.output_root, dataset_id="")
    out_dir = output_root.parent / "figures" / config.comparison_stage_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(float(config.figure_width), float(config.figure_height)),
        dpi=int(config.dpi),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    left_axes = np.array([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]])
    right_axes = np.array([axes[0, 2], axes[0, 3], axes[1, 2], axes[1, 3]])
    mode_axes = {
        "supervised": left_axes,
        "unsupervised": right_axes,
    }

    all_records: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    for mode, axes_flat in mode_axes.items():
        records, warnings = _plot_mode_on_axes(
            axes_flat,
            mode,
            config,
            output_root,
            show_dataset_titles=True,
        )
        for ax in axes_flat:
            ax.set_ylabel("")
        all_records.extend(records)
        all_warnings.extend(warnings)

    supervised_handles = _legend_handles(
        _method_specs_for_mode(config, "supervised"),
        config,
    )
    unsupervised_handles = _legend_handles(
        _method_specs_for_mode(config, "unsupervised"),
        config,
    )

    fig.tight_layout(rect=(0.018, 0.065, 1.0, 0.79), w_pad=1.35, h_pad=1.1)
    left_x0, left_x1 = _axes_span(left_axes)
    right_x0, right_x1 = _axes_span(right_axes)
    left_center = (left_x0 + left_x1) / 2.0
    right_center = (right_x0 + right_x1) / 2.0

    fig.text(
        left_center,
        0.985,
        "Supervised Adaptation",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        right_center,
        0.985,
        "Weakly Supervised Adaptation",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    fig.legend(
        handles=supervised_handles,
        loc="upper center",
        bbox_to_anchor=(left_center, 0.89),
        ncol=len(supervised_handles),
        frameon=False,
        columnspacing=1.1,
        handlelength=1.9,
        prop={"weight": "bold"},
    )
    fig.legend(
        handles=unsupervised_handles,
        loc="upper center",
        bbox_to_anchor=(right_center, 0.89),
        ncol=len(unsupervised_handles),
        frameon=False,
        columnspacing=1.1,
        handlelength=1.9,
        prop={"weight": "bold"},
    )
    _add_horizontal_bracket(fig, left_x0, left_x1, 0.925, tick=0.01)
    _add_horizontal_bracket(fig, right_x0, right_x1, 0.925, tick=0.01)
    axis_label_size = plt.rcParams["figure.labelsize"]
    fig.supylabel(_metric_label(config.metric), x=0.006, fontweight="bold")
    fig.text(
        left_center,
        0.025,
        "N Shots Per Activity Class (log scale)",
        ha="center",
        va="center",
        fontsize=axis_label_size,
        fontweight="bold",
    )
    fig.text(
        right_center,
        0.025,
        "N Shots Per Activity Class (log scale)",
        ha="center",
        va="center",
        fontsize=axis_label_size,
        fontweight="bold",
    )

    records_df = pd.DataFrame(all_records)
    records_csv = out_dir / f"combined_{config.metric}_grid_values.csv"
    records_df.to_csv(records_csv, index=False)
    out_path = out_dir / f"combined_{config.metric}_n0_to_{config.max_k}_grid.png"
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(out_path, dpi=int(config.dpi))
    fig.savefig(pdf_path)
    plt.close(fig)

    config_payload = asdict(config)
    if config_payload.get("source_records_csv"):
        config_payload["source_records_csv"] = repo_relative_path(
            config_payload["source_records_csv"]
        )
    summary = {
        "config": config_payload,
        "mode": "combined",
        "metric": config.metric,
        "output_path": repo_relative_path(out_path),
        "pdf_path": repo_relative_path(pdf_path),
        "records_csv": repo_relative_path(records_csv),
        "warnings": all_warnings,
        "methods": {
            mode: [asdict(spec) for spec in _method_specs_for_mode(config, mode)]
            for mode in mode_axes
        },
        "num_records": int(records_df.shape[0]),
    }
    (out_dir / f"combined_{config.metric}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    for warning in all_warnings:
        print(f"[warn] {warning}")
    print(f"[plot] combined {config.metric}: {out_path}")
    return summary


def run(config: Config) -> dict[str, Any]:
    if len(config.dataset_titles) != len(config.dataset_ids):
        raise ValueError("dataset_titles must match dataset_ids length.")
    if config.mode == "all":
        summaries = {}
        for mode in ("supervised", "unsupervised", "combined"):
            summaries[mode] = run(replace(config, mode=mode))
        return {
            "config": asdict(config),
            "mode": "all",
            "summaries": summaries,
        }
    if config.mode == "combined":
        return _run_combined(config)
    return _run_single_mode(config)


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

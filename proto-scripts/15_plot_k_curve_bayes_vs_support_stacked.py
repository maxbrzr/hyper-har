import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FONT_CACHE_DIR = ROOT / "artifacts" / "proto_pipeline" / ".cache"
FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(ROOT / "artifacts" / "proto_pipeline" / ".matplotlib"),
)
os.environ.setdefault("XDG_CACHE_HOME", str(FONT_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from common import resolve_output_root
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


@dataclass(frozen=True)
class Config:
    output_root: str | None = None
    comparison_stage_name: str = "15_k_curve_bayes_vs_support_stacked"

    dataset_ids: tuple[str, ...] = ("hhar", "wear", "harth", "hapt")
    max_k: int = 16

    support_prefix: str = "05_supcon_support_prototypes_loso"
    bayesian_prefix: str = "07_bayesian_support_prototypes_loso"
    fixed_prefix: str = "04_supcon_fixed_prototypes_loso"
    map_em_prefix: str = "09_unlabeled_map_em_support_prototypes_loso"
    map_em_centered_contains: tuple[str, ...] = ("_centered",)
    include_map_em: bool = True

    figure_width: float = 4.0
    figure_height: float = 8.0
    dpi: int = 240
    ylim_high: float = 1.0
    y_padding_low: float = 0.015
    y_padding_high: float = 0.02
    min_ylim_low: float = 0.0

    color_fixed_proto: str = "#d62728"
    color_bayesian: str = "#2ca02c"
    color_support: str = "#ff7f0e"
    color_map_em: str = "#7f7f7f"
    color_map_em_centered: str = "#1f77b4"
    axis_label_fontsize: int = 12
    y_tick_step_default: float = 0.025
    y_tick_step_harth_hapt: float = 0.05
    show_std_bands: bool = False  # True


RUN_CONFIG = Config()


def _choose_stage_dir(dataset_dir: Path, prefix: str) -> Path | None:
    candidates = sorted(
        [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    )
    return candidates[0] if candidates else None


def _choose_stage_dirs_by_prefix(dataset_dir: Path, prefix: str) -> list[Path]:
    return sorted(
        [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    )


def _load_curve(dataset_dir: Path, prefix: str, max_k: int) -> pd.DataFrame | None:
    stage_dir = _choose_stage_dir(dataset_dir, prefix)
    if stage_dir is None:
        return None
    csv_path = stage_dir / "overall_by_k_results.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    required = {"k", "macro_f1_mean"}
    if not required.issubset(df.columns):
        return None
    df = df.copy()
    df["k"] = df["k"].astype(int)
    df["macro_f1_mean"] = df["macro_f1_mean"].astype(float)
    if "macro_f1_subject_std" in df.columns:
        df["macro_f1_subject_std"] = df["macro_f1_subject_std"].astype(float)
    else:
        df["macro_f1_subject_std"] = 0.0
    return df[df["k"] <= int(max_k)].sort_values("k")


def _load_fixed_baseline(dataset_dir: Path, prefix: str) -> tuple[float, float] | None:
    stage_dir = _choose_stage_dir(dataset_dir, prefix)
    if stage_dir is None:
        return None
    summary_path = stage_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    folds = summary.get("folds", [])
    values = [float(row["test_macro_f1"]) for row in folds if "test_macro_f1" in row]
    if values:
        vals = np.asarray(values, dtype=float)
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        return mean_val, std_val

    value = summary.get("mean_test_macro_f1")
    if value is None:
        return None
    return float(value), 0.0


def _prepend_k0_from_fixed(
    curve_df: pd.DataFrame | None, fixed_value: float | None
) -> pd.DataFrame | None:
    if curve_df is None or fixed_value is None:
        return curve_df
    df = curve_df.copy()
    df = df[df["k"].astype(int) != 0]
    k0 = pd.DataFrame(
        [{"k": 0, "macro_f1_mean": float(fixed_value), "macro_f1_subject_std": 0.0}]
    )
    return pd.concat([k0, df], ignore_index=True).sort_values("k")


def _plot_line_with_band(
    ax: Any,
    df: pd.DataFrame | None,
    label: str,
    color: str,
    marker: str,
    show_std_bands: bool,
) -> None:
    if df is None or df.empty:
        return
    plot_df = df.sort_values("k").copy()
    sns.lineplot(
        data=plot_df,
        x="k",
        y="macro_f1_mean",
        ax=ax,
        label=label,
        color=color,
        marker=marker,
        linewidth=1.9,
        markersize=4.0,
        errorbar=None,
    )
    if show_std_bands:
        y = plot_df["macro_f1_mean"].astype(float).to_numpy()
        s = plot_df["macro_f1_subject_std"].astype(float).fillna(0.0).to_numpy()
        x = plot_df["k"].astype(float).to_numpy()
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.15, linewidth=0)


def run(config: Config) -> dict[str, Any]:
    output_root = resolve_output_root(config.output_root, dataset_id="")
    out_dir = output_root / config.comparison_stage_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        len(config.dataset_ids),
        1,
        figsize=(config.figure_width, config.figure_height),
        dpi=config.dpi,
        sharex=True,
        sharey=False,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    warnings: list[str] = []
    records: list[dict[str, Any]] = []

    for idx, dataset_id in enumerate(config.dataset_ids):
        ax = axes[idx]
        dataset_dir = output_root / dataset_id
        if not dataset_dir.exists():
            warnings.append(f"[{dataset_id}] dataset dir missing")
            ax.set_title(dataset_id.upper(), loc="left")
            ax.text(
                0.5,
                0.5,
                "dataset missing",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            continue

        fixed_stats = _load_fixed_baseline(dataset_dir, config.fixed_prefix)
        fixed_baseline = fixed_stats[0] if fixed_stats is not None else None
        fixed_std = fixed_stats[1] if fixed_stats is not None else 0.0

        support_df = _prepend_k0_from_fixed(
            _load_curve(dataset_dir, config.support_prefix, config.max_k),
            fixed_baseline,
        )
        bayes_df = _prepend_k0_from_fixed(
            _load_curve(dataset_dir, config.bayesian_prefix, config.max_k),
            fixed_baseline,
        )

        map_em_uncentered_df = None
        map_em_centered_df = None
        if config.include_map_em:
            candidates = _choose_stage_dirs_by_prefix(dataset_dir, config.map_em_prefix)
            centered_stage = next(
                (
                    p
                    for p in candidates
                    if any(t in p.name for t in config.map_em_centered_contains)
                ),
                None,
            )
            uncentered_stage = next(
                (p for p in candidates if p != centered_stage), None
            )
            if centered_stage is not None:
                map_em_centered_df = _prepend_k0_from_fixed(
                    _load_curve(dataset_dir, centered_stage.name, config.max_k),
                    fixed_baseline,
                )
            if uncentered_stage is not None:
                map_em_uncentered_df = _prepend_k0_from_fixed(
                    _load_curve(dataset_dir, uncentered_stage.name, config.max_k),
                    fixed_baseline,
                )
            if map_em_centered_df is None and map_em_uncentered_df is None:
                map_em_uncentered_df = _prepend_k0_from_fixed(
                    _load_curve(dataset_dir, config.map_em_prefix, config.max_k),
                    fixed_baseline,
                )
        if support_df is not None and not support_df.empty:
            _plot_line_with_band(
                ax,
                support_df,
                "Support Proto",
                config.color_support,
                "o",
                config.show_std_bands,
            )
            for _, r in support_df.iterrows():
                records.append(
                    {
                        "dataset": dataset_id,
                        "method": "Support Proto",
                        "k": int(r["k"]),
                        "macro_f1": float(r["macro_f1_mean"]),
                    }
                )
        if bayes_df is not None and not bayes_df.empty:
            _plot_line_with_band(
                ax,
                bayes_df,
                "Bayesian",
                config.color_bayesian,
                "s",
                config.show_std_bands,
            )
            for _, r in bayes_df.iterrows():
                records.append(
                    {
                        "dataset": dataset_id,
                        "method": "Bayesian",
                        "k": int(r["k"]),
                        "macro_f1": float(r["macro_f1_mean"]),
                    }
                )
        if map_em_uncentered_df is not None and not map_em_uncentered_df.empty:
            _plot_line_with_band(
                ax,
                map_em_uncentered_df,
                "MAP-EM",
                config.color_map_em,
                "^",
                config.show_std_bands,
            )
            for _, r in map_em_uncentered_df.iterrows():
                records.append(
                    {
                        "dataset": dataset_id,
                        "method": "MAP-EM",
                        "k": int(r["k"]),
                        "macro_f1": float(r["macro_f1_mean"]),
                    }
                )
        if map_em_centered_df is not None and not map_em_centered_df.empty:
            _plot_line_with_band(
                ax,
                map_em_centered_df,
                "MAP-EM centered",
                config.color_map_em_centered,
                "D",
                config.show_std_bands,
            )
            for _, r in map_em_centered_df.iterrows():
                records.append(
                    {
                        "dataset": dataset_id,
                        "method": "MAP-EM centered",
                        "k": int(r["k"]),
                        "macro_f1": float(r["macro_f1_mean"]),
                    }
                )

        if fixed_baseline is not None:
            x_band = np.asarray([0, int(config.max_k)], dtype=float)
            y_low = np.asarray(
                [fixed_baseline - fixed_std, fixed_baseline - fixed_std], dtype=float
            )
            y_high = np.asarray(
                [fixed_baseline + fixed_std, fixed_baseline + fixed_std], dtype=float
            )
            if config.show_std_bands:
                ax.fill_between(
                    x_band,
                    y_low,
                    y_high,
                    color=config.color_fixed_proto,
                    alpha=0.20,
                    linewidth=0,
                )
            ax.axhline(
                fixed_baseline,
                color=config.color_fixed_proto,
                linewidth=1.1,
                linestyle="--",
                label="Fixed Proto",
            )
            records.append(
                {
                    "dataset": dataset_id,
                    "method": "Fixed Proto",
                    "k": -1,
                    "macro_f1": float(fixed_baseline),
                }
            )

        yvals = []
        for d in (support_df, bayes_df, map_em_uncentered_df, map_em_centered_df):
            if d is not None and not d.empty:
                yvals.extend(d["macro_f1_mean"].astype(float).tolist())
        if fixed_baseline is not None:
            yvals.append(float(fixed_baseline))
        if yvals:
            ymin = max(config.min_ylim_low, min(yvals) - config.y_padding_low)
            ymax = min(config.ylim_high, max(yvals) + config.y_padding_high)
            if ymax <= ymin:
                ymax = min(config.ylim_high, ymin + 0.05)
            ax.set_ylim(ymin, ymax)

        ax.set_title(dataset_id.upper(), loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        if dataset_id.lower() in {"harth", "hapt"}:
            ax.yaxis.set_major_locator(MultipleLocator(config.y_tick_step_harth_hapt))
        else:
            ax.yaxis.set_major_locator(MultipleLocator(config.y_tick_step_default))
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    xticks = [k for k in [0, 1, 2, 4, 8, 16, 32] if k <= int(config.max_k)]
    axes[-1].set_xticks(xticks)

    all_handles, all_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)
    unique: dict[str, Any] = {}
    for h, l in zip(all_handles, all_labels):
        if l and l != "_nolegend_" and l not in unique:
            unique[l] = h

    desired = ["Support Proto", "Bayesian", "MAP-EM", "MAP-EM centered", "Fixed Proto"]
    ordered_labels = [l for l in desired if l in unique]
    ordered_handles = [unique[l] for l in ordered_labels]

    # For ncol=2, matplotlib fills columns top->bottom.
    # Re-index so the rendered reading order becomes row-wise:
    # Row1: Support Proto | Bayesian
    # Row2: MAP-EM | MAP-EM centered
    # Row3: Fixed Proto | (empty)
    if len(ordered_handles) == 5:
        ordered_handles = [
            ordered_handles[0],
            ordered_handles[2],
            ordered_handles[4],
            ordered_handles[1],
            ordered_handles[3],
        ]
        ordered_labels = [
            ordered_labels[0],
            ordered_labels[2],
            ordered_labels[4],
            ordered_labels[1],
            ordered_labels[3],
        ]

    if ordered_handles:
        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.99),
            columnspacing=1.4,
            handletextpad=0.5,
        )

    fig.supxlabel(
        "Support Examples per Activity Class",
        x=0.5,
        y=0.03,
        fontsize=config.axis_label_fontsize,
    )
    fig.supylabel(
        "Mean Macro-F1 over Subjects", x=0.02, fontsize=config.axis_label_fontsize
    )
    plt.tight_layout()
    fig.subplots_adjust(top=0.84)

    png_path = out_dir / "k_curve_bayesian_vs_support_stacked.png"
    pdf_path = out_dir / "k_curve_bayesian_vs_support_stacked.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    values_csv = out_dir / "k_curve_bayesian_vs_support_stacked_values.csv"
    pd.DataFrame(records).to_csv(values_csv, index=False)

    summary = {
        "config": asdict(config),
        "plot_png": str(png_path),
        "plot_pdf": str(pdf_path),
        "values_csv": str(values_csv),
        "warnings": warnings,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved plot: {png_path}")
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

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
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter


@dataclass(frozen=True)
class Config:
    output_root: str | None = None
    comparison_stage_name: str = "12_k_curve_bayes_vs_support_grid"

    dataset_ids: tuple[str, ...] = ("hhar", "wear", "harth", "hapt")
    max_k: int = 16

    support_prefix: str = "05_supcon_support_prototypes_loso"
    bayesian_prefix: str = "07_bayesian_support_prototypes_loso"
    fixed_prefix: str = "04_supcon_fixed_prototypes_loso"
    map_em_prefix: str = "09_unlabeled_map_em_support_prototypes_loso"
    map_em_centered_contains: tuple[str, ...] = ("_centered",)
    include_map_em: bool = True

    figure_width: float = 12
    figure_height: float = 4
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


RUN_CONFIG = Config()


def _choose_stage_dir(dataset_dir: Path, prefix: str) -> Path | None:
    candidates = sorted(
        [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    )
    if not candidates:
        return None
    return candidates[0]


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
    return df[df["k"] <= int(max_k)].sort_values("k")


def _choose_stage_dirs_by_prefix(dataset_dir: Path, prefix: str) -> list[Path]:
    return sorted(
        [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    )


def _load_fixed_baseline(dataset_dir: Path, prefix: str) -> float | None:
    stage_dir = _choose_stage_dir(dataset_dir, prefix)
    if stage_dir is None:
        return None
    summary_path = stage_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    value = summary.get("mean_test_macro_f1")
    if value is None:
        folds = summary.get("folds", [])
        if not folds:
            return None
        values = [
            float(row["test_macro_f1"]) for row in folds if "test_macro_f1" in row
        ]
        if not values:
            return None
        return float(np.mean(values))
    return float(value)


def _prepend_k0_from_fixed(
    curve_df: pd.DataFrame | None, fixed_value: float | None
) -> pd.DataFrame | None:
    if curve_df is None or fixed_value is None:
        return curve_df
    df = curve_df.copy()
    df = df[df["k"].astype(int) != 0]
    k0 = pd.DataFrame([{"k": 0, "macro_f1_mean": float(fixed_value)}])
    return pd.concat([k0, df], ignore_index=True).sort_values("k")


def run(config: Config) -> dict[str, Any]:
    output_root = resolve_output_root(config.output_root, dataset_id="")
    out_dir = output_root / config.comparison_stage_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(config.figure_width, config.figure_height),
        dpi=config.dpi,
        sharex=True,
        sharey=False,
    )
    axes_flat = axes.flatten()

    warnings: list[str] = []
    records: list[dict[str, Any]] = []

    for idx, dataset_id in enumerate(config.dataset_ids):
        ax = axes_flat[idx]
        dataset_dir = output_root / dataset_id
        if not dataset_dir.exists():
            warnings.append(f"[{dataset_id}] dataset dir missing")
            ax.set_title(dataset_id)
            ax.text(
                0.5,
                0.5,
                "dataset missing",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            continue

        support_df = _load_curve(dataset_dir, config.support_prefix, config.max_k)
        bayes_df = _load_curve(dataset_dir, config.bayesian_prefix, config.max_k)
        fixed_baseline = _load_fixed_baseline(dataset_dir, config.fixed_prefix)
        support_df = _prepend_k0_from_fixed(support_df, fixed_baseline)
        bayes_df = _prepend_k0_from_fixed(bayes_df, fixed_baseline)
        map_em_uncentered_df = None
        map_em_centered_df = None
        if config.include_map_em:
            map_em_candidates = _choose_stage_dirs_by_prefix(
                dataset_dir, config.map_em_prefix
            )
            centered_stage = next(
                (
                    p
                    for p in map_em_candidates
                    if any(tok in p.name for tok in config.map_em_centered_contains)
                ),
                None,
            )
            uncentered_stage = next(
                (p for p in map_em_candidates if p != centered_stage), None
            )
            if centered_stage is not None:
                map_em_centered_df = _load_curve(
                    dataset_dir, centered_stage.name, config.max_k
                )
                map_em_centered_df = _prepend_k0_from_fixed(
                    map_em_centered_df, fixed_baseline
                )
            if uncentered_stage is not None:
                map_em_uncentered_df = _load_curve(
                    dataset_dir, uncentered_stage.name, config.max_k
                )
                map_em_uncentered_df = _prepend_k0_from_fixed(
                    map_em_uncentered_df, fixed_baseline
                )
            if map_em_centered_df is None and map_em_uncentered_df is None:
                fallback_df = _load_curve(
                    dataset_dir, config.map_em_prefix, config.max_k
                )
                map_em_uncentered_df = _prepend_k0_from_fixed(
                    fallback_df, fixed_baseline
                )

        if support_df is None:
            warnings.append(f"[{dataset_id}] support curve missing")
        if bayes_df is None:
            warnings.append(f"[{dataset_id}] bayesian curve missing")
        if fixed_baseline is None:
            warnings.append(f"[{dataset_id}] fixed-prototype baseline missing")
        if (
            config.include_map_em
            and map_em_centered_df is None
            and map_em_uncentered_df is None
        ):
            warnings.append(f"[{dataset_id}] map-em curves missing")

        if support_df is not None and not support_df.empty:
            ax.plot(
                support_df["k"],
                support_df["macro_f1_mean"],
                marker="o",
                linewidth=1.9,
                markersize=3.8,
                color=config.color_support,
                label="Support Proto",
            )
            for _, row in support_df.iterrows():
                records.append(
                    {
                        "dataset": dataset_id,
                        "method": "Support Proto",
                        "k": int(row["k"]),
                        "macro_f1": float(row["macro_f1_mean"]),
                    }
                )

        if bayes_df is not None and not bayes_df.empty:
            ax.plot(
                bayes_df["k"],
                bayes_df["macro_f1_mean"],
                marker="s",
                linewidth=1.9,
                markersize=3.8,
                color=config.color_bayesian,
                label="Bayesian",
            )
            for _, row in bayes_df.iterrows():
                records.append(
                    {
                        "dataset": dataset_id,
                        "method": "Bayesian",
                        "k": int(row["k"]),
                        "macro_f1": float(row["macro_f1_mean"]),
                    }
                )

        if map_em_uncentered_df is not None and not map_em_uncentered_df.empty:
            ax.plot(
                map_em_uncentered_df["k"],
                map_em_uncentered_df["macro_f1_mean"],
                marker="^",
                linewidth=1.8,
                markersize=3.6,
                color=config.color_map_em,
                label="MAP-EM",
            )
            for _, row in map_em_uncentered_df.iterrows():
                records.append(
                    {
                        "dataset": dataset_id,
                        "method": "MAP-EM",
                        "k": int(row["k"]),
                        "macro_f1": float(row["macro_f1_mean"]),
                    }
                )

        if map_em_centered_df is not None and not map_em_centered_df.empty:
            ax.plot(
                map_em_centered_df["k"],
                map_em_centered_df["macro_f1_mean"],
                marker="D",
                linewidth=1.8,
                markersize=3.6,
                color=config.color_map_em_centered,
                label="MAP-EM centered",
            )
            for _, row in map_em_centered_df.iterrows():
                records.append(
                    {
                        "dataset": dataset_id,
                        "method": "MAP-EM centered",
                        "k": int(row["k"]),
                        "macro_f1": float(row["macro_f1_mean"]),
                    }
                )

        if fixed_baseline is not None:
            ax.axhline(
                fixed_baseline,
                color=config.color_fixed_proto,
                linewidth=1.1,
                linestyle="--",
                label="Fixed Proto" if idx == 0 else None,
            )
            records.append(
                {
                    "dataset": dataset_id,
                    "method": "Fixed Proto",
                    "k": -1,
                    "macro_f1": float(fixed_baseline),
                }
            )
        else:
            ax.text(
                0.03,
                0.06,
                "fixed baseline missing",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7,
                color="dimgray",
            )

        y_candidates: list[float] = []
        if support_df is not None and not support_df.empty:
            y_candidates.extend(support_df["macro_f1_mean"].astype(float).tolist())
        if bayes_df is not None and not bayes_df.empty:
            y_candidates.extend(bayes_df["macro_f1_mean"].astype(float).tolist())
        if map_em_uncentered_df is not None and not map_em_uncentered_df.empty:
            y_candidates.extend(
                map_em_uncentered_df["macro_f1_mean"].astype(float).tolist()
            )
        if map_em_centered_df is not None and not map_em_centered_df.empty:
            y_candidates.extend(
                map_em_centered_df["macro_f1_mean"].astype(float).tolist()
            )
        if fixed_baseline is not None:
            y_candidates.append(float(fixed_baseline))

        if y_candidates:
            y_min = min(y_candidates)
            y_max = max(y_candidates)
            ylim_low = max(config.min_ylim_low, y_min - config.y_padding_low)
            ylim_high = min(config.ylim_high, y_max + config.y_padding_high)
            if ylim_high <= ylim_low:
                ylim_high = min(config.ylim_high, ylim_low + 0.05)
            ax.set_ylim(ylim_low, ylim_high)

        ax.set_title(dataset_id)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        if dataset_id.lower() in {"harth", "hapt"}:
            ax.yaxis.set_major_locator(MultipleLocator(config.y_tick_step_harth_hapt))
        else:
            ax.yaxis.set_major_locator(MultipleLocator(config.y_tick_step_default))
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

    for idx in range(len(config.dataset_ids), 4):
        axes_flat[idx].axis("off")

    xticks = [k for k in [0, 1, 2, 4, 8, 16, 32] if k <= int(config.max_k)]
    for ax in axes_flat[: min(4, len(config.dataset_ids))]:
        ax.set_xticks(xticks)

    all_handles = []
    all_labels = []
    for ax in axes_flat[: min(4, len(config.dataset_ids))]:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)
    unique: dict[str, Any] = {}
    for handle, label in zip(all_handles, all_labels):
        if not label or label == "_nolegend_":
            continue
        if label not in unique:
            unique[label] = handle

    if unique:
        legend_order = [
            "Support Proto",
            "Bayesian",
            "MAP-EM",
            "MAP-EM centered",
            "Fixed Proto",
        ]
        ordered_labels = [label for label in legend_order if label in unique]
        for label in unique.keys():
            if label not in ordered_labels:
                ordered_labels.append(label)
        fig.legend(
            [unique[label] for label in ordered_labels],
            ordered_labels,
            loc="upper center",
            ncol=5 if config.include_map_em else 3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
        )

    fig.supxlabel(
        "Support Examples per Activity Class",
        x=0.5,
        y=0.045,
        fontsize=config.axis_label_fontsize,
    )
    fig.text(
        0.015,
        0.5,
        "Mean Macro-F1 over Subjects",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=config.axis_label_fontsize,
    )
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 0.95])
    fig.subplots_adjust(bottom=0.16)

    png_path = out_dir / "k_curve_bayesian_vs_support_2x2.png"
    pdf_path = out_dir / "k_curve_bayesian_vs_support_2x2.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    values_csv = out_dir / "k_curve_bayesian_vs_support_values.csv"
    pd.DataFrame(records).to_csv(values_csv, index=False)

    summary = {
        "config": asdict(config),
        "plot_png": str(png_path),
        "plot_pdf": str(pdf_path),
        "values_csv": str(values_csv),
        "warnings": warnings,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for line in warnings:
        print(f"[warn] {line}")
    print(f"Saved plot: {png_path}")
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

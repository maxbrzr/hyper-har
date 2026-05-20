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
from matplotlib.ticker import FormatStrFormatter

from common import resolve_output_root


@dataclass(frozen=True)
class Config:
    output_root: str | None = None
    comparison_stage_name: str = "14_loso_grouped_macro_f1_stacked"
    dataset_ids: tuple[str, ...] = ("hhar", "wear", "harth", "hapt")

    method_order: tuple[str, ...] = (
        "Baseline CE",
        "ProtoNet Fixed",
        "MAP-EM centered (k=16)",
        "Bayesian (k=16)",
        "Support Proto (k=16)",
    )

    baseline_prefix: str = "06_ce_baseline_eval_loso"
    fixed_prefix: str = "04_supcon_fixed_prototypes_loso"
    map_em_prefix: str = "09_unlabeled_map_em_support_prototypes_loso"
    bayesian_prefix: str = "07_bayesian_support_prototypes_loso"
    support_prefix: str = "05_supcon_support_prototypes_loso"
    map_em_prefer_contains: tuple[str, ...] = ("_centered",)
    k_value_for_proto_methods: int = 16

    title: str = "LOSO Mean Macro-F1 by Method (Stacked by Dataset)"
    y_label: str = "Mean Macro-F1 over Subjects"
    x_label: str = "Method"
    figure_width: float = 8.0
    figure_height: float = 10.0
    dpi: int = 220
    ylim_low: float = 0.5
    ylim_high: float = 1.0
    axis_label_fontsize: int = 12

    color_baseline_ce: str = "#9467bd"
    color_fixed_proto: str = "#d62728"
    color_map_em: str = "#1f77b4"
    color_bayesian: str = "#2ca02c"
    color_support: str = "#ff7f0e"


RUN_CONFIG = Config()


def _choose_stage_dir(dataset_dir: Path, prefix: str, prefer_contains: tuple[str, ...] = ()) -> Path | None:
    candidates = sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if not candidates:
        return None
    if prefer_contains:
        preferred = [p for p in candidates if any(token in p.name for token in prefer_contains)]
        if preferred:
            return preferred[0]
    return candidates[0]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_result(dataset_dir: Path, config: Config) -> tuple[float, float] | None:
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
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _fixed_result(dataset_dir: Path, config: Config) -> tuple[float, float] | None:
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
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _k_result(dataset_dir: Path, prefix: str, k_value: int, prefer_contains: tuple[str, ...] = ()) -> tuple[float, float] | None:
    stage_dir = _choose_stage_dir(dataset_dir, prefix, prefer_contains=prefer_contains)
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
    return float(row["macro_f1_mean"]), float(row["macro_f1_subject_std"])


def _collect(config: Config) -> tuple[pd.DataFrame, list[str]]:
    output_root = resolve_output_root(config.output_root, dataset_id="")
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    for dataset_id in config.dataset_ids:
        dataset_dir = output_root / dataset_id
        if not dataset_dir.exists():
            warnings.append(f"[{dataset_id}] dataset dir missing")
            continue

        method_results = {
            "Baseline CE": _baseline_result(dataset_dir, config),
            "ProtoNet Fixed": _fixed_result(dataset_dir, config),
            "MAP-EM centered (k=16)": _k_result(
                dataset_dir,
                config.map_em_prefix,
                config.k_value_for_proto_methods,
                prefer_contains=config.map_em_prefer_contains,
            ),
            "Bayesian (k=16)": _k_result(
                dataset_dir,
                config.bayesian_prefix,
                config.k_value_for_proto_methods,
            ),
            "Support Proto (k=16)": _k_result(
                dataset_dir,
                config.support_prefix,
                config.k_value_for_proto_methods,
            ),
        }

        for method in config.method_order:
            result = method_results[method]
            if result is None:
                warnings.append(f"[{dataset_id}] missing method: {method}")
                continue
            mean_val, std_val = result
            rows.append(
                {
                    "dataset": dataset_id,
                    "method": method,
                    "macro_f1": round(float(mean_val), 3),
                    "fold_std": round(float(std_val), 3),
                }
            )

    if not rows:
        raise RuntimeError("No data rows found for stacked plot.")

    return pd.DataFrame(rows), warnings


def _plot(df: pd.DataFrame, config: Config, output_root: Path) -> dict[str, str]:
    out_dir = output_root / config.comparison_stage_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(config.figure_width, config.figure_height), dpi=config.dpi, sharex=True)

    method_colors = {
        "Baseline CE": config.color_baseline_ce,
        "ProtoNet Fixed": config.color_fixed_proto,
        "MAP-EM centered (k=16)": config.color_map_em,
        "Bayesian (k=16)": config.color_bayesian,
        "Support Proto (k=16)": config.color_support,
    }

    x = np.arange(len(config.method_order), dtype=float)

    for i, dataset_id in enumerate(config.dataset_ids):
        ax = axes[i]
        sub = df[df["dataset"].str.lower() == dataset_id.lower()].copy()
        values = {row["method"]: float(row["macro_f1"]) for _, row in sub.iterrows()}
        errs = {row["method"]: float(row["fold_std"]) for _, row in sub.iterrows()}

        for j, method in enumerate(config.method_order):
            if method not in values:
                continue
            y = values[method]
            e = errs.get(method, 0.0)
            ax.bar([x[j]], [y], width=0.72, color=method_colors[method], zorder=3, label=method if i == 0 else None)
            ax.errorbar([x[j]], [y], yerr=[[e], [e]], fmt="none", ecolor="black", elinewidth=1.2, capsize=3, capthick=1.2, zorder=10)
            ax.text(x[j], max(config.ylim_low + 0.01, y - 0.02), f"{y:.3f}", ha="center", va="top", color="white", fontsize=8, fontweight="bold", zorder=20)

        ax.set_title(dataset_id.upper(), loc="left", fontsize=10, fontweight="bold")
        ax.set_ylim(config.ylim_low, config.ylim_high)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(list(config.method_order), rotation=0)

    fig.supylabel(config.y_label, fontsize=config.axis_label_fontsize)
    fig.supxlabel(config.x_label, fontsize=config.axis_label_fontsize)
    fig.suptitle(config.title, y=0.995)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.975))

    plt.tight_layout(rect=[0.04, 0.04, 1.0, 0.95])

    png_path = out_dir / "grouped_macro_f1_loso_stacked.png"
    pdf_path = out_dir / "grouped_macro_f1_loso_stacked.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    return {"png": str(png_path), "pdf": str(pdf_path)}


def run(config: Config) -> dict[str, Any]:
    output_root = resolve_output_root(config.output_root, dataset_id="")
    df, warnings = _collect(config)
    paths = _plot(df, config, output_root)

    out_dir = output_root / config.comparison_stage_name
    values_csv = out_dir / "grouped_macro_f1_loso_stacked_values.csv"
    df.to_csv(values_csv, index=False)

    summary = {
        "config": asdict(config),
        "plot_png": paths["png"],
        "plot_pdf": paths["pdf"],
        "values_csv": str(values_csv),
        "warnings": warnings,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for w in warnings:
        print(f"[warn] {w}")
    print(f"Saved plot: {paths['png']}")
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

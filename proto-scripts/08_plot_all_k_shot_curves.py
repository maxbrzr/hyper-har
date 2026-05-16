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
import pandas as pd


@dataclass(frozen=True)
class CurveSpec:
    stage_dir_name: str
    label: str | None = None


@dataclass(frozen=True)
class Config:
    output_root: str = str(ROOT / "artifacts" / "proto_pipeline")
    comparison_stage_name: str = "08_k_shot_curve_comparison"

    # Leave empty to auto-discover every overall_by_k_results.csv under output_root.
    curve_specs: tuple[CurveSpec, ...] = ()
    include_stage_prefixes: tuple[str, ...] = ("05_", "07_")
    exclude_stage_names: tuple[str, ...] = ("08_k_shot_curve_comparison",)

    metrics: tuple[str, ...] = ("macro_f1", "accuracy")
    aggregation: str = "subject"  # "subject" or "trial"
    error_band: str = "ci95"  # "ci95", "std", or "none"
    min_subjects: int = 1
    title_prefix: str = "K-Shot Prototype Evaluation"
    figure_width: float = 9.5
    figure_height: float = 5.8
    dpi: int = 220
    y_axis_mode: str = "auto"  # "auto" or "zero"
    y_axis_padding: float = 0.03
    force_rerun: bool = True


RUN_CONFIG = Config()


def _humanize_stage_name(stage_name: str, summary: dict[str, Any] | None) -> str:
    if summary is not None:
        config = dict(summary.get("config", {}))
        backbone_source = summary.get("backbone_source", config.get("backbone_source"))
        embedding_space = config.get("embedding_space")
    else:
        backbone_source = None
        embedding_space = None

    name = stage_name
    method = "Support"
    if "bayesian" in name:
        method = "Bayesian"
    elif "support" in name:
        method = "Support"

    if backbone_source is None:
        if "_ce_backbone" in name:
            backbone_source = "ce"
        elif "_supcon_backbone" in name or "supcon" in name:
            backbone_source = "supcon"

    label_parts = [method]
    if backbone_source:
        label_parts.append(f"{str(backbone_source).upper()} backbone")
    if embedding_space and str(backbone_source) == "supcon":
        label_parts.append(str(embedding_space))
    return " - ".join(label_parts)


def _load_summary(stage_dir: Path) -> dict[str, Any] | None:
    summary_path = stage_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _discover_curve_specs(config: Config) -> list[CurveSpec]:
    output_root = Path(config.output_root)
    specs: list[CurveSpec] = []
    for csv_path in sorted(output_root.glob("*/overall_by_k_results.csv")):
        stage_name = csv_path.parent.name
        if stage_name in set(config.exclude_stage_names):
            continue
        if config.include_stage_prefixes and not any(
            stage_name.startswith(prefix) for prefix in config.include_stage_prefixes
        ):
            continue
        specs.append(CurveSpec(stage_dir_name=stage_name, label=None))
    return specs


def _metric_columns(metric: str, aggregation: str, error_band: str) -> tuple[str, str | None]:
    if aggregation == "subject":
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_subject_{error_band}"
    elif aggregation == "trial":
        mean_col = f"{metric}_trial_mean"
        std_col = f"{metric}_trial_{error_band}"
    else:
        raise ValueError("aggregation must be 'subject' or 'trial'.")

    if error_band == "none":
        return mean_col, None
    if error_band not in {"ci95", "std"}:
        raise ValueError("error_band must be 'ci95', 'std', or 'none'.")
    return mean_col, std_col


def _plot_metric(
    curves: list[dict[str, Any]],
    metric: str,
    config: Config,
    out_path: Path,
) -> None:
    mean_col, err_col = _metric_columns(metric, config.aggregation, config.error_band)
    fig, ax = plt.subplots(figsize=(config.figure_width, config.figure_height))

    plotted = 0
    y_values: list[float] = []
    y_err_values: list[float] = []
    for curve in curves:
        df = curve["df"]
        missing = [col for col in ("k", mean_col) if col not in df.columns]
        if missing:
            print(
                f"[skip] {curve['stage_name']} missing columns for {metric}: {missing}"
            )
            continue
        if "num_subjects" in df.columns:
            df = df[df["num_subjects"].astype(int) >= int(config.min_subjects)]
        if df.empty:
            print(f"[skip] {curve['stage_name']} has no rows after filtering")
            continue

        df = df.sort_values("k")
        yerr = None
        if err_col is not None and err_col in df.columns:
            yerr = df[err_col].astype(float).to_numpy()
            y_err_values.extend(float(x) for x in yerr.tolist())
        y = df[mean_col].astype(float).to_numpy()
        y_values.extend(float(x) for x in y.tolist())
        ax.errorbar(
            df["k"].astype(int).to_numpy(),
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=3 if yerr is not None else 0,
            label=curve["label"],
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise RuntimeError(f"No curves could be plotted for metric {metric}.")

    metric_title = metric.replace("_", " ").title()
    aggregation_title = "Subject mean" if config.aggregation == "subject" else "Trial mean"
    ax.set_title(f"{config.title_prefix}: {metric_title}")
    ax.set_xlabel("K shots per activity")
    ax.set_ylabel(metric_title)
    ax.grid(True, alpha=0.25)
    if config.y_axis_mode == "zero":
        ax.set_ylim(bottom=0.0, top=1.02)
    elif config.y_axis_mode == "auto":
        err_max = max(y_err_values) if y_err_values else 0.0
        lower = max(0.0, min(y_values) - err_max - float(config.y_axis_padding))
        upper = min(1.02, max(y_values) + err_max + float(config.y_axis_padding))
        if upper - lower < 0.08:
            mid = (upper + lower) / 2.0
            lower = max(0.0, mid - 0.04)
            upper = min(1.02, mid + 0.04)
        ax.set_ylim(bottom=lower, top=upper)
    else:
        raise ValueError("y_axis_mode must be 'auto' or 'zero'.")
    ax.legend(loc="best", fontsize=9)
    ax.text(
        0.01,
        0.01,
        f"{aggregation_title}, error={config.error_band}",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.65,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(config.dpi))
    plt.close(fig)


def run(config: Config) -> dict[str, Any]:
    output_root = Path(config.output_root)
    comparison_dir = output_root / config.comparison_stage_name
    comparison_dir.mkdir(parents=True, exist_ok=True)

    specs = list(config.curve_specs) if config.curve_specs else _discover_curve_specs(config)
    if not specs:
        raise RuntimeError(
            f"No overall_by_k_results.csv files found under {output_root}."
        )

    curves: list[dict[str, Any]] = []
    combined_rows: list[pd.DataFrame] = []
    for spec in specs:
        stage_dir = output_root / spec.stage_dir_name
        csv_path = stage_dir / "overall_by_k_results.csv"
        if not csv_path.exists():
            print(f"[skip] missing {csv_path}")
            continue
        summary = _load_summary(stage_dir)
        label = spec.label or _humanize_stage_name(spec.stage_dir_name, summary)
        df = pd.read_csv(csv_path)
        df.insert(0, "curve_label", label)
        df.insert(1, "stage_name", spec.stage_dir_name)
        curves.append(
            {
                "stage_name": spec.stage_dir_name,
                "label": label,
                "csv_path": str(csv_path),
                "summary_path": str(stage_dir / "summary.json"),
                "df": df,
            }
        )
        combined_rows.append(df)
        print(f"[load] {label}: {csv_path}")

    if not curves:
        raise RuntimeError("No available curves were loaded.")

    combined_df = pd.concat(combined_rows, ignore_index=True)
    combined_csv = comparison_dir / "combined_k_shot_curves.csv"
    combined_df.to_csv(combined_csv, index=False)

    plot_paths: dict[str, str] = {}
    for metric in config.metrics:
        out_path = comparison_dir / f"all_k_shot_curves_{metric}.png"
        _plot_metric(curves, metric, config, out_path)
        plot_paths[metric] = str(out_path)
        print(f"[plot] {metric}: {out_path}")

    summary = {
        "config": asdict(config),
        "comparison_dir": str(comparison_dir),
        "combined_csv": str(combined_csv),
        "plot_paths": plot_paths,
        "curves": [
            {
                "stage_name": curve["stage_name"],
                "label": curve["label"],
                "csv_path": curve["csv_path"],
                "summary_path": curve["summary_path"],
            }
            for curve in curves
        ],
    }
    (comparison_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

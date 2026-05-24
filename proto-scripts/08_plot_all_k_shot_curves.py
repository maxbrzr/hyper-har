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
from common import DEFAULT_DATASET_ID, resolve_output_root


@dataclass(frozen=True)
class CurveSpec:
    stage_dir_name: str
    label: str | None = None


@dataclass(frozen=True)
class FixedBaselineSpec:
    stage_dir_name: str
    label: str | None = None


@dataclass(frozen=True)
class Config:
    dataset_id: str = DEFAULT_DATASET_ID
    output_root: str | None = None
    comparison_stage_name: str = "08_k_shot_curve_comparison"

    # Leave empty to auto-discover every overall_by_k_results.csv under output_root.
    curve_specs: tuple[CurveSpec, ...] = ()
    fixed_baseline_specs: tuple[FixedBaselineSpec, ...] = ()
    include_stage_prefixes: tuple[str, ...] = (
        "05_",
        "07_",
        "09_",
        "16_",
        "17_",
        "18_",
        "19_",
        "20_",
    )
    include_fixed_baseline_prefixes: tuple[str, ...] = ("04_", "06_", "21_")
    exclude_stage_names: tuple[str, ...] = ("08_k_shot_curve_comparison",)
    include_fixed_baselines: bool = True

    metrics: tuple[str, ...] = ("macro_f1", "accuracy")
    aggregation: str = "subject"  # "subject" or "trial"
    error_band: str = "ci95"  # "ci95", "std", or "none"
    min_subjects: int = 1
    title_prefix: str = "K-Shot Adaptation Evaluation"
    figure_width: float = 9.5
    figure_height: float = 5.8
    dpi: int = 220
    y_axis_mode: str = "auto"  # "auto" or "zero"
    y_axis_padding: float = 0.03
    color_palette: tuple[str, ...] = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#bcbd22",
        "#393b79",
        "#637939",
        "#8c6d31",
        "#843c39",
        "#7b4173",
        "#3182bd",
        "#31a354",
        "#756bb1",
        "#636363",
        "#e6550d",
        "#969696",
    )
    force_rerun: bool = True


RUN_CONFIG = Config()


def _humanize_stage_name(stage_name: str, summary: dict[str, Any] | None) -> str:
    if summary is not None:
        config = dict(summary.get("config", {}))
        backbone_source = summary.get("backbone_source", config.get("backbone_source"))
        embedding_space = config.get("embedding_space")
        active_transform = summary.get("active_embedding_transform")
        normalize_embeddings = summary.get(
            "effective_normalize_embeddings", summary.get("normalize_embeddings")
        )
        effective_distance = summary.get(
            "effective_distance_metric", summary.get("distance_metric")
        )
        folds = summary.get("folds", [])
        first_fold = folds[0] if folds else {}
    else:
        backbone_source = None
        embedding_space = None
        active_transform = None
        normalize_embeddings = None
        effective_distance = None
        first_fold = {}

    name = stage_name
    method = "Support Proto"
    if name.startswith("06_") or "ce_baseline" in name:
        method = "CE classifier"
    elif name.startswith("21_") or "classifier_weight" in name:
        method = "Classifier-weight proto"
    elif name.startswith("20_") or "logistic_linear_head" in name:
        method = "Logistic head"
    elif name.startswith("19_") or "closed_form_linear_head" in name:
        method = "Closed-form head"
    elif name.startswith("18_") or "oftta" in name:
        method = "OFTTA"
    elif name.startswith("17_") or "neo_recenter" in name:
        method = "NEO"
    elif name.startswith("16_") or "pda" in name:
        method = "PDA"
    elif name.startswith("04_") or "fixed" in name:
        method = "Train fixed proto"
    elif "map_em" in name or "unlabeled" in name:
        method = "MAP-EM"
    elif "bayesian" in name:
        method = "Bayesian"
    elif "support" in name:
        method = "Support Proto"

    if backbone_source is None and first_fold:
        backbone_source = first_fold.get("backbone_source")
    if embedding_space is None and first_fold:
        embedding_space = first_fold.get("effective_embedding_space") or first_fold.get(
            "embedding_space"
        )
    if effective_distance is None and first_fold:
        effective_distance = first_fold.get("effective_distance_metric") or first_fold.get(
            "distance_metric"
        )
    if backbone_source is None:
        if "_ce_backbone" in name:
            backbone_source = "ce"
        elif "_supcon_backbone" in name or "supcon" in name:
            backbone_source = "supcon"
    if effective_distance is None:
        if "_cosine" in name:
            effective_distance = "cosine"
        elif "_euclidean" in name:
            effective_distance = "euclidean"
    if (
        effective_distance is None
        and backbone_source in {"supcon", "ce"}
        and method in {"Support Proto", "Train fixed proto", "Bayesian", "MAP-EM"}
    ):
        effective_distance = "cosine" if backbone_source == "supcon" else "euclidean"

    label_parts = [method]
    if backbone_source:
        label_parts.append(f"{str(backbone_source).upper()} backbone")
    if embedding_space and str(backbone_source) == "supcon":
        label_parts.append(str(embedding_space))
    if effective_distance:
        label_parts.append(str(effective_distance))
    if active_transform and active_transform != "none":
        label_parts.append(str(active_transform))
        label_parts.append("L2" if normalize_embeddings else "raw stats")
    if "centered" in name and method == "MAP-EM":
        label_parts.append("centered")
    if "respvar" in name and method == "MAP-EM":
        label_parts.append("resp-var")
    if "diaglike" in name:
        label_parts.append("diag-like")
    if "mcd" in name and method == "PDA":
        label_parts.append("MCD")
    elif "classifier_confidence" in name and method == "PDA":
        label_parts.append("classifier")
    if method == "OFTTA":
        edtn_decay = config.get("edtn_decay") if summary is not None else None
        if edtn_decay is not None:
            label_parts.append(f"EDTN d={float(edtn_decay):g}")
    if method == "Closed-form head":
        ridge = summary.get("ridge_lambda") if summary is not None else None
        if ridge is not None:
            label_parts.append(f"ridge={float(ridge):g}")
    if method == "Logistic head":
        inverse_regularization = (
            summary.get("inverse_regularization") if summary is not None else None
        )
        if inverse_regularization is not None:
            label_parts.append(f"C={float(inverse_regularization):g}")
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
    output_root = resolve_output_root(config.output_root, config.dataset_id)
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


def _discover_fixed_baseline_specs(config: Config) -> list[FixedBaselineSpec]:
    output_root = resolve_output_root(config.output_root, config.dataset_id)
    specs: list[FixedBaselineSpec] = []
    for summary_path in sorted(output_root.glob("*/summary.json")):
        stage_name = summary_path.parent.name
        if stage_name in set(config.exclude_stage_names):
            continue
        if config.include_fixed_baseline_prefixes and not any(
            stage_name.startswith(prefix)
            for prefix in config.include_fixed_baseline_prefixes
        ):
            continue
        specs.append(FixedBaselineSpec(stage_dir_name=stage_name, label=None))
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


def _fixed_baseline_metric_column(metric: str) -> str:
    if metric == "macro_f1":
        return "test_macro_f1"
    if metric == "accuracy":
        return "test_accuracy"
    if metric == "weighted_f1":
        return "test_weighted_f1"
    return f"test_{metric}"


def _fixed_baseline_metrics(summary: dict[str, Any]) -> dict[str, float]:
    folds = summary.get("folds", [])
    if not folds:
        return {}
    df = pd.DataFrame(folds)
    metrics: dict[str, float] = {}
    for column in ("test_macro_f1", "test_accuracy", "test_weighted_f1"):
        if column in df.columns:
            values = df[column].dropna().astype(float)
            if not values.empty:
                metrics[column] = float(values.mean())
                metrics[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                metrics[f"{column}_num_subjects"] = int(values.shape[0])
    return metrics


def _plot_metric(
    curves: list[dict[str, Any]],
    fixed_baselines: list[dict[str, Any]],
    metric: str,
    config: Config,
    out_path: Path,
) -> None:
    mean_col, err_col = _metric_columns(metric, config.aggregation, config.error_band)
    fig, ax = plt.subplots(figsize=(config.figure_width, config.figure_height))
    palette = tuple(config.color_palette)
    if not palette:
        raise ValueError("color_palette must contain at least one color.")

    plotted = 0
    y_values: list[float] = []
    y_err_values: list[float] = []
    for curve_idx, curve in enumerate(curves):
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
            color=palette[curve_idx % len(palette)],
            label=curve["label"],
        )
        plotted += 1

    baseline_col = _fixed_baseline_metric_column(metric)
    k_values: list[int] = []
    for curve in curves:
        df = curve["df"]
        if "k" in df.columns:
            k_values.extend(int(x) for x in df["k"].dropna().astype(int).tolist())
    x_min = min(k_values) if k_values else 0
    x_max = max(k_values) if k_values else 1
    for baseline_idx, baseline in enumerate(fixed_baselines):
        if baseline_col not in baseline["metrics"]:
            print(
                f"[skip] {baseline['stage_name']} missing fixed baseline metric "
                f"{baseline_col}"
            )
            continue
        value = float(baseline["metrics"][baseline_col])
        y_values.append(value)
        ax.hlines(
            value,
            xmin=x_min,
            xmax=x_max,
            linestyles="--",
            linewidth=2,
            alpha=0.85,
            color=palette[(len(curves) + baseline_idx) % len(palette)],
            label=baseline["label"],
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
    output_root = resolve_output_root(config.output_root, config.dataset_id)
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

    fixed_baselines: list[dict[str, Any]] = []
    fixed_baseline_rows: list[dict[str, Any]] = []
    if config.include_fixed_baselines:
        baseline_specs = (
            list(config.fixed_baseline_specs)
            if config.fixed_baseline_specs
            else _discover_fixed_baseline_specs(config)
        )
        for spec in baseline_specs:
            stage_dir = output_root / spec.stage_dir_name
            summary = _load_summary(stage_dir)
            if summary is None:
                print(f"[skip] missing fixed baseline summary: {stage_dir}")
                continue
            metrics = _fixed_baseline_metrics(summary)
            if not metrics:
                print(f"[skip] no fixed baseline fold metrics: {stage_dir}")
                continue
            label = spec.label or _humanize_stage_name(spec.stage_dir_name, summary)
            row = {
                "stage_name": spec.stage_dir_name,
                "label": label,
                "summary_path": str(stage_dir / "summary.json"),
                **metrics,
            }
            fixed_baselines.append(
                {
                    "stage_name": spec.stage_dir_name,
                    "label": label,
                    "summary_path": str(stage_dir / "summary.json"),
                    "metrics": metrics,
                }
            )
            fixed_baseline_rows.append(row)
            print(f"[load] {label} fixed baseline: {stage_dir / 'summary.json'}")

    combined_df = pd.concat(combined_rows, ignore_index=True)
    combined_csv = comparison_dir / "combined_k_shot_curves.csv"
    combined_df.to_csv(combined_csv, index=False)
    fixed_baseline_csv = comparison_dir / "fixed_baselines.csv"
    pd.DataFrame(fixed_baseline_rows).to_csv(fixed_baseline_csv, index=False)

    plot_paths: dict[str, str] = {}
    for metric in config.metrics:
        out_path = comparison_dir / f"all_k_shot_curves_{metric}.png"
        _plot_metric(curves, fixed_baselines, metric, config, out_path)
        plot_paths[metric] = str(out_path)
        print(f"[plot] {metric}: {out_path}")

    summary = {
        "config": asdict(config),
        "comparison_dir": str(comparison_dir),
        "combined_csv": str(combined_csv),
        "fixed_baseline_csv": str(fixed_baseline_csv),
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
        "fixed_baselines": fixed_baseline_rows,
    }
    (comparison_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

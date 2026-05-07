from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = ROOT / "artifacts"
DEFAULT_LOSO_DIR = DEFAULT_ARTIFACTS / "loso_cv"
DEFAULT_META_ROOT = DEFAULT_ARTIFACTS / "meta_loso_cv"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)}.")
    return payload


def _subject_id_from_dir_name(name: str) -> int | None:
    match = re.fullmatch(r"subject_(\d+)", name)
    if match is None:
        return None
    return int(match.group(1))


def _discover_latest_meta_variant(meta_root: Path) -> Path:
    candidates: list[tuple[float, Path]] = []
    for variant_dir in meta_root.iterdir():
        if not variant_dir.is_dir():
            continue
        summary_path = variant_dir / "summary.json"
        if summary_path.exists():
            candidates.append((summary_path.stat().st_mtime, variant_dir))
            continue

        # Fallback for in-progress runs without summary.json:
        # consider variant dirs that already contain at least one subject meta_metrics file.
        subject_metric_files = list(variant_dir.glob("subject_*/meta_metrics.json"))
        if subject_metric_files:
            latest_metric_mtime = max(p.stat().st_mtime for p in subject_metric_files)
            candidates.append((latest_metric_mtime, variant_dir))
    if not candidates:
        raise FileNotFoundError(
            f"No variant directory with summary.json found in: {meta_root}"
        )
    return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]


def _resolve_meta_variant(meta_root: Path, meta_variant: str | None) -> Path:
    if meta_variant is None:
        return _discover_latest_meta_variant(meta_root)

    as_path = Path(meta_variant)
    if as_path.exists() and as_path.is_dir():
        return as_path

    candidate = meta_root / meta_variant
    if candidate.exists() and candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        "Could not resolve --meta-variant. "
        f"Tried path '{as_path}' and '{candidate}'."
    )


def _resolve_meta_subfolder(
    artifacts_root: Path,
    meta_root: Path,
    meta_subfolder: str | None,
    meta_variant: str | None,
) -> Path:
    if meta_subfolder is not None:
        as_path = Path(meta_subfolder)
        if as_path.exists() and as_path.is_dir():
            return as_path

        artifacts_candidate = artifacts_root / meta_subfolder
        if artifacts_candidate.exists() and artifacts_candidate.is_dir():
            return artifacts_candidate

        meta_root_candidate = meta_root / meta_subfolder
        if meta_root_candidate.exists() and meta_root_candidate.is_dir():
            return meta_root_candidate

        raise FileNotFoundError(
            "Could not resolve --meta-subfolder. "
            f"Tried '{as_path}', '{artifacts_candidate}', and '{meta_root_candidate}'."
        )

    return _resolve_meta_variant(meta_root, meta_variant)


def _infer_steps_per_epoch_from_variant_name(variant_name: str) -> int | None:
    match = re.search(r"train-ep-(\d+)", variant_name)
    if match is None:
        return None
    return int(match.group(1))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _collect_subject_rows(loso_dir: Path, meta_variant_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for subject_dir in sorted(meta_variant_dir.iterdir(), key=lambda p: p.name):
        if not subject_dir.is_dir():
            continue
        subject_dir_id = _subject_id_from_dir_name(subject_dir.name)

        meta_metrics_path = subject_dir / "meta_metrics.json"
        if not meta_metrics_path.exists():
            continue

        meta_metrics = _load_json(meta_metrics_path)
        subject_id = _optional_int(
            meta_metrics.get("test_subject_id", meta_metrics.get("subject_id"))
        )
        if subject_id is None:
            subject_id = subject_dir_id
        if subject_id is None:
            continue

        loso_metrics_path = loso_dir / f"subject_{subject_id}" / "metrics.json"
        if not loso_metrics_path.exists():
            continue

        loso_metrics = _load_json(loso_metrics_path)

        loso_f1 = float(loso_metrics["test_macro_f1"])
        meta_f1 = float(meta_metrics["test_macro_f1"])
        best_epoch = int(meta_metrics["best_epoch"])
        improvement = meta_f1 - loso_f1

        rows.append(
            {
                "subject_id": subject_id,
                "meta_subject_dir": str(subject_dir),
                "loso_test_macro_f1": loso_f1,
                "meta_test_macro_f1": meta_f1,
                "improvement": improvement,
                "best_epoch": best_epoch,
                "best_val_loss": _optional_float(meta_metrics.get("best_val_loss")),
                "best_val_macro_f1": _optional_float(
                    meta_metrics.get(
                        "best_val_macro_f1", meta_metrics.get("val_macro_f1_at_best")
                    )
                ),
                "final_val_loss": _optional_float(meta_metrics.get("final_val_loss")),
                "final_val_macro_f1": _optional_float(
                    meta_metrics.get("final_val_macro_f1")
                ),
            }
        )

    rows.sort(key=lambda r: int(r["subject_id"]))
    return rows


def _aggregate_epoch_means(values: list[float], steps_per_epoch: int) -> list[float]:
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
    means: list[float] = []
    for start in range(0, len(values), steps_per_epoch):
        chunk = values[start : start + steps_per_epoch]
        if not chunk:
            continue
        means.append(float(mean(chunk)))
    return means


def _plot_meta_train_loss(
    rows: list[dict[str, Any]],
    output_path: Path,
    steps_per_epoch: int | None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    plotted = 0
    for row in rows:
        subject_id = int(row["subject_id"])
        subject_dir = Path(str(row["meta_subject_dir"]))
        history_path = subject_dir / "meta_history.json"
        if not history_path.exists():
            continue

        history = _load_json(history_path)
        raw_values = history.get("meta_train_loss", [])
        if not isinstance(raw_values, list) or len(raw_values) == 0:
            continue

        loss_values = [float(v) for v in raw_values]
        best_epoch = int(row["best_epoch"])

        if steps_per_epoch is not None:
            y = _aggregate_epoch_means(loss_values, steps_per_epoch)
            x = list(range(1, len(y) + 1))
            label = f"subject {subject_id}"
            ax.plot(x, y, linewidth=1.5, alpha=0.85, label=label)
            if 1 <= best_epoch <= len(y):
                ax.scatter(
                    [best_epoch],
                    [y[best_epoch - 1]],
                    s=25,
                    zorder=3,
                )
        else:
            x = list(range(1, len(loss_values) + 1))
            label = f"subject {subject_id}"
            ax.plot(x, loss_values, linewidth=1.0, alpha=0.75, label=label)

        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise RuntimeError("No meta history data found to plot.")

    if steps_per_epoch is not None:
        ax.set_xlabel("Epoch")
        ax.set_title("Meta Training Loss by Subject (Epoch Mean)")
    else:
        ax.set_xlabel("Meta Training Step")
        ax.set_title("Meta Training Loss by Subject (Raw Steps)")

    ax.set_ylabel("Meta Train Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_meta_val_metric(
    rows: list[dict[str, Any]],
    output_path: Path,
    metric_key: str,
    title: str,
    y_label: str,
) -> int:
    fig, ax = plt.subplots(figsize=(14, 7))
    plotted = 0

    for row in rows:
        subject_id = int(row["subject_id"])
        best_epoch = int(row["best_epoch"])
        subject_dir = Path(str(row["meta_subject_dir"]))
        history_path = subject_dir / "meta_history.json"
        if not history_path.exists():
            continue

        history = _load_json(history_path)
        raw_values = history.get(metric_key, [])
        if not isinstance(raw_values, list) or len(raw_values) == 0:
            continue

        y = [float(v) for v in raw_values]
        x = list(range(1, len(y) + 1))
        ax.plot(x, y, linewidth=1.5, alpha=0.85, label=f"subject {subject_id}")
        if 1 <= best_epoch <= len(y):
            ax.scatter([best_epoch], [y[best_epoch - 1]], s=25, zorder=3)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return 0

    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return plotted


def _plot_improvement_bars(rows: list[dict[str, Any]], output_path: Path) -> None:
    subject_ids = [int(r["subject_id"]) for r in rows]
    improvements = [float(r["improvement"]) for r in rows]
    mean_improvement = sum(improvements) / len(improvements)
    mean_x = (max(subject_ids) + 1) if subject_ids else 0

    all_x = subject_ids + [mean_x]
    all_y = improvements + [mean_improvement]
    colors = ["#2a9d8f" if v >= 0 else "#d62828" for v in improvements] + ["#264653"]

    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(all_x, all_y, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Improvement in Test Macro F1 (Meta LOSO - LOSO)")
    ax.set_title("Per-Subject Test Macro F1 Improvement After Meta Adaptation")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticks(all_x)
    ax.set_xticklabels([str(s) for s in subject_ids] + ["mean"])

    y_min = min(all_y) if all_y else 0.0
    y_max = max(all_y) if all_y else 0.0
    y_range = max(1e-9, y_max - y_min)
    pad = 0.03 * y_range

    for bar, row in zip(bars[:-1], rows):
        improvement = float(row["improvement"])
        loso_f1 = float(row["loso_test_macro_f1"])
        meta_f1 = float(row["meta_test_macro_f1"])
        best_epoch = int(row["best_epoch"])
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()

        annotation = (
            f"{improvement:+.4f}\n"
            f"{loso_f1:.4f} -> {meta_f1:.4f}\n"
            f"best ep {best_epoch}"
        )
        if improvement >= 0:
            ax.text(
                x,
                y + pad,
                annotation,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
        else:
            ax.text(
                x,
                y - pad,
                annotation,
                ha="center",
                va="top",
                fontsize=8,
                rotation=90,
            )

    mean_bar = bars[-1]
    mean_xpos = mean_bar.get_x() + mean_bar.get_width() / 2.0
    mean_y = mean_bar.get_height()
    mean_annotation = f"mean\n{mean_improvement:+.4f}"
    if mean_improvement >= 0:
        ax.text(
            mean_xpos,
            mean_y + pad,
            mean_annotation,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    else:
        ax.text(
            mean_xpos,
            mean_y - pad,
            mean_annotation,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "subject_id",
        "loso_test_macro_f1",
        "meta_test_macro_f1",
        "improvement",
        "best_epoch",
        "best_val_loss",
        "best_val_macro_f1",
        "final_val_loss",
        "final_val_macro_f1",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create plots for meta-training artifacts: "
            "lineplots of meta train/validation curves and barplot of LOSO-to-meta F1 improvement."
        )
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help=f"Artifacts root path (default: {DEFAULT_ARTIFACTS})",
    )
    parser.add_argument(
        "--loso-dir",
        type=Path,
        default=DEFAULT_LOSO_DIR,
        help=f"Path to LOSO artifacts directory (default: {DEFAULT_LOSO_DIR})",
    )
    parser.add_argument(
        "--meta-root",
        type=Path,
        default=DEFAULT_META_ROOT,
        help=f"Path to meta artifacts root directory (default: {DEFAULT_META_ROOT})",
    )
    parser.add_argument(
        "--meta-variant",
        type=str,
        default=None,
        help=(
            "Meta variant directory name under --meta-root, or full path. "
            "If omitted, the most recently modified variant is used."
        ),
    )
    parser.add_argument(
        "--meta-subfolder",
        type=str,
        default=None,
        help=(
            "Meta-training artifacts subfolder to plot. Can be an absolute path, "
            "a path under --artifacts-root, or a path under --meta-root. "
            "This takes priority over --meta-variant."
        ),
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help=(
            "Meta training steps per epoch for epoch-level averaging. "
            "If omitted, auto-inferred from variant name (pattern: train-ep-<N>)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for plots and CSV summary. "
            "Default: <resolved-meta-variant>/plots"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    loso_dir = args.loso_dir
    meta_root = args.meta_root
    if args.artifacts_root != DEFAULT_ARTIFACTS:
        if args.loso_dir == DEFAULT_LOSO_DIR:
            loso_dir = args.artifacts_root / "loso_cv"
        if args.meta_root == DEFAULT_META_ROOT:
            meta_root = args.artifacts_root / "meta_loso_cv"

    meta_variant_dir = _resolve_meta_subfolder(
        artifacts_root=args.artifacts_root,
        meta_root=meta_root,
        meta_subfolder=args.meta_subfolder,
        meta_variant=args.meta_variant,
    )
    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is None:
        steps_per_epoch = _infer_steps_per_epoch_from_variant_name(meta_variant_dir.name)

    output_dir = args.output_dir or (meta_variant_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_subject_rows(loso_dir, meta_variant_dir)
    if not rows:
        raise RuntimeError(
            "No overlapping LOSO and meta subject metrics found. "
            "Check artifact directories and variant selection."
        )

    lineplot_path = output_dir / "meta_train_loss_lineplot.png"
    val_loss_path = output_dir / "val_loss_lineplot.png"
    val_macro_f1_path = output_dir / "val_macro_f1_lineplot.png"
    barplot_path = output_dir / "macro_f1_improvement_barplot.png"
    summary_csv_path = output_dir / "subject_improvement_summary.csv"

    _plot_meta_train_loss(
        rows=rows,
        output_path=lineplot_path,
        steps_per_epoch=steps_per_epoch,
    )
    val_loss_plotted = _plot_meta_val_metric(
        rows=rows,
        output_path=val_loss_path,
        metric_key="val_loss",
        title="Validation Loss by Subject",
        y_label="Validation Loss",
    )
    val_macro_f1_plotted = _plot_meta_val_metric(
        rows=rows,
        output_path=val_macro_f1_path,
        metric_key="val_macro_f1",
        title="Validation Macro F1 by Subject",
        y_label="Validation Macro F1",
    )
    _plot_improvement_bars(rows=rows, output_path=barplot_path)
    _write_summary_csv(rows=rows, output_path=summary_csv_path)

    mean_improvement = sum(float(r["improvement"]) for r in rows) / len(rows)
    print(f"Meta variant: {meta_variant_dir}")
    print(f"Subjects plotted: {len(rows)}")
    print(f"Mean improvement (macro F1): {mean_improvement:+.4f}")
    if steps_per_epoch is not None:
        print(f"Using steps_per_epoch={steps_per_epoch} for loss aggregation.")
    else:
        print("Could not infer steps_per_epoch; plotted raw meta training steps.")
    print(f"Saved lineplot: {lineplot_path}")
    if val_loss_plotted > 0:
        print(f"Saved val loss plot: {val_loss_path}")
    else:
        print("Skipped val loss plot (no val_loss series found in selected subfolder).")
    if val_macro_f1_plotted > 0:
        print(f"Saved val macro F1 plot: {val_macro_f1_path}")
    else:
        print(
            "Skipped val macro F1 plot (no val_macro_f1 series found in selected subfolder)."
        )
    print(f"Saved barplot: {barplot_path}")
    print(f"Saved summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()

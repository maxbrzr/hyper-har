from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "artifacts" / "lora_finetune_loso_cv"


def _subject_id_from_dir_name(name: str) -> int | None:
    match = re.fullmatch(r"subject_(\d+)", name)
    if match is None:
        return None
    return int(match.group(1))


def _load_json_if_ready(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Allows plotting while another process is writing the JSON file.
        return None
    return payload if isinstance(payload, dict) else None


def _collect_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject_dir in sorted(input_dir.glob("subject_*"), key=lambda p: p.name):
        if not subject_dir.is_dir():
            continue
        metrics_path = subject_dir / "lora_finetune_metrics.json"
        if not metrics_path.exists():
            continue

        metrics = _load_json_if_ready(metrics_path)
        if metrics is None:
            continue

        subject_id = metrics.get("test_subject_id", metrics.get("subject_id"))
        if subject_id is None:
            subject_id = _subject_id_from_dir_name(subject_dir.name)
        if subject_id is None:
            continue

        required = [
            "base_macro_f1",
            "finetuned_macro_f1",
            "macro_f1_improvement",
            "base_loss",
            "finetuned_loss",
        ]
        if any(key not in metrics for key in required):
            continue

        rows.append(
            {
                "subject_id": int(subject_id),
                "subject_dir": str(subject_dir),
                "split_index": int(metrics.get("split_index", -1)),
                "episodes": int(metrics.get("episodes", 0)),
                "query_per_class": int(metrics.get("query_per_class", 0)),
                "finetune_steps": int(metrics.get("finetune_steps", 0)),
                "finetune_lr": float(metrics.get("finetune_lr", 0.0)),
                "base_loss": float(metrics["base_loss"]),
                "finetuned_loss": float(metrics["finetuned_loss"]),
                "loss_improvement": float(metrics["base_loss"])
                - float(metrics["finetuned_loss"]),
                "base_macro_f1": float(metrics["base_macro_f1"]),
                "finetuned_macro_f1": float(metrics["finetuned_macro_f1"]),
                "macro_f1_improvement": float(metrics["macro_f1_improvement"]),
                "support_per_class_choices": "-".join(
                    str(k) for k in metrics.get("support_per_class_choices", [])
                ),
                "lora_rank": int(metrics.get("lora_rank", 0)),
                "lora_alpha": float(metrics.get("lora_alpha", 0.0)),
                "adapter_modules": ",".join(metrics.get("adapter_modules", [])),
            }
        )

    rows.sort(key=lambda row: int(row["subject_id"]))
    return rows


def _mean_std_sem(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean_v = float(arr.mean())
    std_v = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sem_v = float(std_v / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean_v, std_v, sem_v, float(1.96 * sem_v)


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_improvement_bar(rows: list[dict[str, Any]], output_path: Path) -> None:
    subject_ids = [int(row["subject_id"]) for row in rows]
    improvements = [float(row["macro_f1_improvement"]) for row in rows]
    mean_improvement = float(np.mean(improvements))
    mean_x = max(subject_ids) + 1 if subject_ids else 0
    all_x = subject_ids + [mean_x]
    all_y = improvements + [mean_improvement]
    colors = ["#2a9d8f" if value >= 0 else "#d62828" for value in improvements]
    colors.append("#264653")

    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(all_x, all_y, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Macro F1 Improvement (Fine-tuned LoRA - Base)")
    ax.set_title("LoRA Fine-tuning Same-Query Improvement by Subject")
    ax.set_xticks(all_x)
    ax.set_xticklabels([str(sid) for sid in subject_ids] + ["mean"])
    ax.grid(axis="y", alpha=0.25)

    y_min = min(all_y) if all_y else 0.0
    y_max = max(all_y) if all_y else 0.0
    pad = 0.03 * max(1e-9, y_max - y_min)
    for bar, row in zip(bars[:-1], rows):
        value = float(row["macro_f1_improvement"])
        base_f1 = float(row["base_macro_f1"])
        tuned_f1 = float(row["finetuned_macro_f1"])
        label = f"{value:+.4f}\n{base_f1:.4f}->{tuned_f1:.4f}"
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()
        ax.text(
            x,
            y + pad if value >= 0 else y - pad,
            label,
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
            rotation=90,
        )

    mean_bar = bars[-1]
    mean_y = mean_bar.get_height()
    ax.text(
        mean_bar.get_x() + mean_bar.get_width() / 2.0,
        mean_y + pad if mean_improvement >= 0 else mean_y - pad,
        f"mean\n{mean_improvement:+.4f}",
        ha="center",
        va="bottom" if mean_improvement >= 0 else "top",
        fontsize=9,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_base_vs_tuned(rows: list[dict[str, Any]], output_path: Path) -> None:
    subject_ids = [int(row["subject_id"]) for row in rows]
    x = np.arange(len(subject_ids), dtype=np.float64)
    width = 0.38
    base_values = [float(row["base_macro_f1"]) for row in rows]
    tuned_values = [float(row["finetuned_macro_f1"]) for row in rows]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(x - width / 2.0, base_values, width=width, label="Base", color="#a8dadc")
    ax.bar(
        x + width / 2.0,
        tuned_values,
        width=width,
        label="Fine-tuned LoRA",
        color="#1d3557",
    )
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Macro F1")
    ax.set_title("Base vs Fine-tuned LoRA Macro F1 by Subject")
    ax.set_xticks(x)
    ax.set_xticklabels([str(sid) for sid in subject_ids])
    ax.set_ylim(0.0, min(1.0, max(tuned_values + base_values) + 0.08))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_loss_bar(rows: list[dict[str, Any]], output_path: Path) -> None:
    subject_ids = [int(row["subject_id"]) for row in rows]
    improvements = [float(row["loss_improvement"]) for row in rows]
    colors = ["#2a9d8f" if value >= 0 else "#d62828" for value in improvements]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(subject_ids, improvements, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Loss Improvement (Base - Fine-tuned LoRA)")
    ax.set_title("LoRA Fine-tuning Query Loss Improvement by Subject")
    ax.set_xticks(subject_ids)
    ax.set_xticklabels([str(sid) for sid in subject_ids])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_summary(rows: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    improvements = [float(row["macro_f1_improvement"]) for row in rows]
    base_f1s = [float(row["base_macro_f1"]) for row in rows]
    tuned_f1s = [float(row["finetuned_macro_f1"]) for row in rows]
    loss_improvements = [float(row["loss_improvement"]) for row in rows]
    mean_imp, std_imp, sem_imp, ci95_imp = _mean_std_sem(improvements)

    summary = {
        "num_completed_subjects": len(rows),
        "completed_subject_ids": [int(row["subject_id"]) for row in rows],
        "mean_base_macro_f1": float(np.mean(base_f1s)) if base_f1s else 0.0,
        "mean_finetuned_macro_f1": float(np.mean(tuned_f1s)) if tuned_f1s else 0.0,
        "mean_macro_f1_improvement": mean_imp,
        "std_macro_f1_improvement": std_imp,
        "sem_macro_f1_improvement": sem_imp,
        "ci95_macro_f1_improvement": ci95_imp,
        "mean_loss_improvement": float(np.mean(loss_improvements))
        if loss_improvements
        else 0.0,
        "rows": rows,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _render_once(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _collect_rows(input_dir)
    if not rows:
        raise RuntimeError(
            f"No completed LoRA fine-tuning metrics found under {input_dir}."
        )

    _write_csv(rows, output_dir / "lora_finetune_subject_summary.csv")
    _plot_improvement_bar(rows, output_dir / "lora_finetune_improvement_barplot.png")
    _plot_base_vs_tuned(rows, output_dir / "lora_finetune_base_vs_tuned_barplot.png")
    _plot_loss_bar(rows, output_dir / "lora_finetune_loss_improvement_barplot.png")
    return _write_summary(rows, output_dir / "lora_finetune_partial_summary.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot partial or completed LoRA fine-tuning LOSO-CV artifacts. "
            "Safe to run while lora_finetune_baseline.py is still running."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"LoRA fine-tuning artifact directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output plot directory (default: <input-dir>/plots)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep refreshing plots until interrupted.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Refresh interval in seconds when --watch is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or (args.input_dir / "plots")

    while True:
        summary = _render_once(args.input_dir, output_dir)
        print(
            "Plotted "
            f"{summary['num_completed_subjects']} completed subjects "
            f"| mean improvement={summary['mean_macro_f1_improvement']:+.4f} "
            f"| output={output_dir}"
        )
        if not args.watch:
            break
        time.sleep(max(1.0, float(args.interval)))


if __name__ == "__main__":
    main()

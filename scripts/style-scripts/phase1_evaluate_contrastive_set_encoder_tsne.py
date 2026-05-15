from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from whar_datasets import Loader, PostProcessingPipeline, PreProcessingPipeline, WHARDatasetID

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT_PATH = ROOT / "style-scripts" / "contrastive_set_encoder.py"


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTRASTIVE_MODULE = _load_module_from_path(
    "contrastive_set_encoder_train_module",
    TRAIN_SCRIPT_PATH,
)

ContrastiveRunConfig = CONTRASTIVE_MODULE.ContrastiveRunConfig
ContrastiveSetEncoder = CONTRASTIVE_MODULE.ContrastiveSetEncoder
WindowSplits = CONTRASTIVE_MODULE.WindowSplits
build_model = CONTRASTIVE_MODULE.build_model
extract_subject_embeddings = CONTRASTIVE_MODULE.extract_subject_embeddings
prepare_one_activity_cfg = CONTRASTIVE_MODULE.prepare_one_activity_cfg
split_windows_by_subject = CONTRASTIVE_MODULE.split_windows_by_subject


@dataclass(frozen=True)
class TSNEEvalConfig:
    checkpoint_path: str = str(
        ROOT / "artifacts" / "contrastive_set_encoder" / "best_contrastive_set_encoder.pt"
    )
    output_dir: str = str(ROOT / "artifacts" / "contrastive_set_encoder_tsne_eval")
    k_values: tuple[int, ...] = (2, 4, 8, 16, 32)
    sets_per_subject: int = 80
    embedding_batch_size: int = 64
    knn_neighbors: int = 5
    tsne_perplexity: float = 40.0
    tsne_learning_rate: str | float = "auto"
    seed: int = 0
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


EVAL_CONFIG = TSNEEvalConfig(
    checkpoint_path=str(
        ROOT / "artifacts" / "contrastive_set_encoder" / "best_contrastive_set_encoder.pt"
    ),
    output_dir=str(ROOT / "artifacts" / "contrastive_set_encoder_tsne_eval"),
    k_values=(2, 4, 8, 16, 32),
    sets_per_subject=80,
    embedding_batch_size=64,
    knn_neighbors=5,
    tsne_perplexity=40.0,
    seed=0,
)


def load_training_config(checkpoint: dict[str, Any]) -> ContrastiveRunConfig:
    raw_config = dict(checkpoint["config"])
    if "train_set_size_choices" in raw_config:
        choices = tuple(int(k) for k in raw_config.pop("train_set_size_choices"))
        raw_config.setdefault("train_min_set_size", min(choices))
        raw_config.setdefault("train_max_set_size", max(choices))
    return ContrastiveRunConfig(**raw_config)


def prepare_loader_and_splits(
    train_config: ContrastiveRunConfig,
) -> tuple[Any, Loader, WindowSplits]:
    dataset_id = WHARDatasetID(train_config.dataset_id)
    cfg, selected_activity = prepare_one_activity_cfg(
        dataset_id=dataset_id,
        datasets_dir=ROOT / "datasets",
        activity=train_config.activity,
    )
    print(
        f"Using dataset={dataset_id.value}, activity='{selected_activity}', "
        f"selected_activities={cfg.selected_activities}, window_overlap={cfg.window_overlap}"
    )

    pre_pipeline = PreProcessingPipeline(cfg)
    _raw_df, session_df, window_df = pre_pipeline.run()
    splits = split_windows_by_subject(
        session_df=session_df,
        window_df=window_df,
        val_size=train_config.val_size,
        test_size=train_config.test_size,
        seed=train_config.seed,
    )
    post_pipeline = PostProcessingPipeline(
        cfg, pre_pipeline, window_df, splits.train_indices
    )
    samples = post_pipeline.run()
    loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)
    return cfg, loader, splits


def load_model(
    checkpoint: dict[str, Any],
    train_config: ContrastiveRunConfig,
    num_channels: int,
    device: torch.device,
) -> ContrastiveSetEncoder:
    model = build_model(
        encoder_name=train_config.encoder,
        num_channels=num_channels,
        window_size=int(checkpoint["window_size"]),
        projection_hidden_dim=train_config.projection_hidden_dim,
        projection_dim=train_config.projection_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def run_knn_for_k(
    model: ContrastiveSetEncoder,
    loader: Loader,
    splits: WindowSplits,
    train_config: ContrastiveRunConfig,
    eval_config: TSNEEvalConfig,
    k: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    train_embeddings, train_labels = extract_subject_embeddings(
        model=model,
        loader=loader,
        indices=splits.train_indices,
        set_size=k,
        sets_per_subject=eval_config.sets_per_subject,
        batch_size=eval_config.embedding_batch_size,
        seed=train_config.seed + 1000 + k,
        device=device,
    )
    test_embeddings, test_labels = extract_subject_embeddings(
        model=model,
        loader=loader,
        indices=splits.test_indices,
        set_size=k,
        sets_per_subject=eval_config.sets_per_subject,
        batch_size=eval_config.embedding_batch_size,
        seed=train_config.seed + 2000 + k,
        device=device,
    )
    n_neighbors = max(1, min(eval_config.knn_neighbors, len(train_embeddings)))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_embeddings, train_labels)
    predictions = knn.predict(test_embeddings)
    accuracy = float(accuracy_score(test_labels, predictions))
    return train_embeddings, train_labels, test_embeddings, test_labels, accuracy


def save_tsne_for_k(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    k: int,
    accuracy: float,
    output_path: Path,
    eval_config: TSNEEvalConfig,
) -> str:
    embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)
    split_markers = np.asarray(
        ["train"] * len(train_embeddings) + ["test"] * len(test_embeddings)
    )
    max_perplexity = max(2.0, float(len(embeddings) - 1) / 3.0)
    perplexity = min(float(eval_config.tsne_perplexity), max_perplexity)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=eval_config.tsne_learning_rate,
        init="pca",
        metric="cosine",
        random_state=eval_config.seed + k,
    )
    coords = tsne.fit_transform(embeddings)

    unique_labels = sorted(int(x) for x in np.unique(labels).tolist())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = np.asarray([label_to_idx[int(x)] for x in labels], dtype=np.int64)
    n_classes = max(1, len(unique_labels))
    cmap = plt.cm.get_cmap("tab20", n_classes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=170)
    train_mask = split_markers == "train"
    test_mask = ~train_mask

    ax.scatter(
        coords[train_mask, 0],
        coords[train_mask, 1],
        c=encoded[train_mask],
        cmap=cmap,
        vmin=0,
        vmax=max(0, n_classes - 1),
        s=16,
        alpha=0.28,
        marker="o",
        linewidths=0.0,
    )
    ax.scatter(
        coords[test_mask, 0],
        coords[test_mask, 1],
        c=encoded[test_mask],
        cmap=cmap,
        vmin=0,
        vmax=max(0, n_classes - 1),
        s=36,
        alpha=0.88,
        marker="x",
        linewidths=0.9,
    )
    ax.set_title(f"Raw c_subject t-SNE | K={k} | test KNN acc={accuracy:.4f}")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    subject_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=cmap(idx),
            markeredgecolor="none",
            markersize=6,
            label=str(label),
        )
        for idx, label in enumerate(unique_labels)
    ]
    subject_legend = ax.legend(
        handles=subject_handles,
        title="Subject",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
    )
    ax.add_artist(subject_legend)
    split_handles = [
        plt.Line2D([0], [0], marker="o", color="black", linestyle="", alpha=0.35, label="train"),
        plt.Line2D([0], [0], marker="x", color="black", linestyle="", alpha=0.9, label="test"),
    ]
    ax.legend(handles=split_handles, title="Split", loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


def run_tsne_evaluation(eval_config: TSNEEvalConfig) -> dict[str, Any]:
    output_dir = Path(eval_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(eval_config.device)

    checkpoint_path = Path(eval_config.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_config = load_training_config(checkpoint)
    train_config = replace(train_config, device=eval_config.device)

    cfg, loader, splits = prepare_loader_and_splits(train_config)
    model = load_model(
        checkpoint=checkpoint,
        train_config=train_config,
        num_channels=cfg.num_of_channels,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    for k in eval_config.k_values:
        print(
            f"Evaluating K={k} with {eval_config.sets_per_subject} sampled sets per subject..."
        )
        train_embeddings, train_labels, test_embeddings, test_labels, accuracy = run_knn_for_k(
            model=model,
            loader=loader,
            splits=splits,
            train_config=train_config,
            eval_config=eval_config,
            k=int(k),
            device=device,
        )
        tsne_path = save_tsne_for_k(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            k=int(k),
            accuracy=accuracy,
            output_path=output_dir / f"tsne_k_{int(k)}.png",
            eval_config=eval_config,
        )
        row = {
            "k": int(k),
            "test_subject_knn_accuracy": accuracy,
            "num_train_embeddings": int(len(train_embeddings)),
            "num_test_embeddings": int(len(test_embeddings)),
            "num_subjects": int(len(set(test_labels.tolist()))),
            "tsne_path": tsne_path,
        }
        rows.append(row)
        print(
            f"K={k}: test KNN accuracy={accuracy:.4f}, "
            f"t-SNE saved to {tsne_path}"
        )

    result = {
        "eval_config": asdict(eval_config),
        "checkpoint_path": str(checkpoint_path),
        "training_config": asdict(train_config),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "best_val_subject_knn_accuracy": float(
            checkpoint.get("best_val_subject_knn_accuracy", float("nan"))
        ),
        "results_by_k": rows,
    }
    metrics_path = output_dir / "tsne_eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    return result


def main() -> None:
    run_tsne_evaluation(EVAL_CONFIG)


if __name__ == "__main__":
    main()

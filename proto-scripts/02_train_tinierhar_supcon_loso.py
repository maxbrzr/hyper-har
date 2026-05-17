import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import (
    ROOT,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASETS_DIR,
    DEFAULT_SELECTED_ACTIVITIES,
    DEFAULT_SEED,
    DEFAULT_TEST_SUBJECTS,
    DEFAULT_VAL_SUBJECTS,
    DEFAULT_WINDOW_OVERLAP,
    DEFAULT_SPLIT_STRATEGY,
    DEFAULT_VAL_PERCENTAGE,
    SharedConfig,
    WindowDataset,
    build_loader,
    build_or_load_loso_folds,
    build_tinierhar,
    class_names,
    config_fingerprint,
    cosine_logits,
    infer_window_size,
    labels_subjects_for_indices,
    make_class_prototypes,
    prepare_cfg,
    prepare_inputs,
    resolve_output_root,
    set_seed,
    split_indices_for_fold,
)
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm
from whar_datasets import PreProcessingPipeline, WHARDatasetID

from hyper_har.backbone.tinierhar import TinierHAR


@dataclass(frozen=True)
class Config:
    dataset_id: str = DEFAULT_DATASET_ID
    datasets_dir: str = DEFAULT_DATASETS_DIR
    selected_activities: list[str] | None = DEFAULT_SELECTED_ACTIVITIES
    window_overlap: float = DEFAULT_WINDOW_OVERLAP
    val_subjects: int = DEFAULT_VAL_SUBJECTS
    test_subjects: int = DEFAULT_TEST_SUBJECTS
    seed: int = DEFAULT_SEED
    split_strategy: str = DEFAULT_SPLIT_STRATEGY
    val_percentage: float = DEFAULT_VAL_PERCENTAGE

    projection_hidden_dim: int = 128
    projection_dim: int | None = None
    supcon_temperature: float = 0.1
    classes_per_batch: int | None = None
    samples_per_class: int = 32
    batches_per_epoch: int = 128
    val_batches_per_epoch: int = 32
    sample_with_replacement: bool = False
    subject_balanced_within_class: bool = True

    learning_rate: float = 1e-4
    weight_decay: float = 0.0  # 1e-4
    epochs: int = 80
    patience: int = 10
    min_delta: float = 0.0
    grad_clip_norm: float | None = 5.0
    num_workers: int = 0
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    early_stopping_metric: str = "val_proto_macro_f1"  # or "val_supcon_loss"
    proto_eval_batch_size: int = 256
    proto_eval_temperature: float = 0.1
    tsne_every_n_epochs: int = 5
    tsne_splits: tuple[str, ...] = ("train", "val")
    tsne_batch_size: int = 256
    tsne_max_samples_per_subject_activity: int = 20
    tsne_max_samples_per_split: int | None = 1200
    tsne_perplexity: float = 30.0
    tsne_max_iter: int = 1000
    tsne_metric: str = "cosine"
    tsne_init: str = "random"
    tsne_point_size: float = 24.0
    tsne_point_alpha: float = 0.78

    output_root: str | None = None
    stage_name: str = "02_tinierhar_supcon_loso"
    max_folds: int | None = None
    force_rerun: bool = False


RUN_CONFIG = Config()

MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8")


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.shape[0], device=features.device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)
        mask_sum = mask.sum(1).clamp(min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        return (-mean_log_prob_pos).mean()


class TinierHARSupCon(nn.Module):
    def __init__(
        self,
        backbone: TinierHAR,
        projection_hidden_dim: int,
        projection_dim: int | None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        feature_dim = int(2 * backbone.nb_units_gru)
        out_dim = feature_dim if projection_dim is None else int(projection_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, int(projection_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(projection_hidden_dim), out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.encode(x)
        return self.projection_head(features)


class ActivityBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: np.ndarray,
        subjects: np.ndarray,
        classes_per_batch: int | None,
        samples_per_class: int,
        batches_per_epoch: int,
        seed: int,
        sample_with_replacement: bool,
        subject_balanced_within_class: bool,
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.subjects = np.asarray(subjects, dtype=np.int64)
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = int(samples_per_class)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.sample_with_replacement = bool(sample_with_replacement)
        self.subject_balanced_within_class = bool(subject_balanced_within_class)
        self.epoch = 0
        self.class_to_positions = {
            int(cls): np.flatnonzero(self.labels == int(cls)).astype(np.int64)
            for cls in sorted(np.unique(self.labels).tolist())
        }
        self.class_subject_positions: dict[int, dict[int, np.ndarray]] = {}
        for cls, positions in self.class_to_positions.items():
            nested: dict[int, np.ndarray] = {}
            for sid in sorted(np.unique(self.subjects[positions]).tolist()):
                nested[int(sid)] = positions[self.subjects[positions] == int(sid)]
            self.class_subject_positions[int(cls)] = nested
        eligible = []
        for cls, positions in self.class_to_positions.items():
            if self.sample_with_replacement or len(positions) >= self.samples_per_class:
                eligible.append(int(cls))
        if len(eligible) < 2:
            raise ValueError("SupCon needs at least two eligible activity classes.")
        if self.samples_per_class < 2:
            raise ValueError("samples_per_class must be >= 2 for positive pairs.")
        self.eligible_classes = eligible

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _sample_positions_for_class(
        self,
        cls: int,
        rng: np.random.Generator,
    ) -> list[int]:
        if not self.subject_balanced_within_class:
            candidates = self.class_to_positions[int(cls)]
            replace = (
                self.sample_with_replacement or len(candidates) < self.samples_per_class
            )
            return [
                int(x)
                for x in rng.choice(
                    candidates, size=self.samples_per_class, replace=replace
                )
            ]

        by_subject = self.class_subject_positions[int(cls)]
        subject_ids = np.asarray(sorted(by_subject.keys()), dtype=np.int64)
        out: list[int] = []
        used: set[int] = set()
        while len(out) < self.samples_per_class:
            sid = int(rng.choice(subject_ids))
            candidates = by_subject[sid]
            available = [int(x) for x in candidates.tolist() if int(x) not in used]
            if not available:
                if self.sample_with_replacement:
                    out.append(int(rng.choice(candidates)))
                else:
                    remaining = [
                        int(x)
                        for x in self.class_to_positions[int(cls)].tolist()
                        if int(x) not in used
                    ]
                    if not remaining:
                        break
                    choice = int(rng.choice(np.asarray(remaining, dtype=np.int64)))
                    used.add(choice)
                    out.append(choice)
                continue
            choice = int(rng.choice(np.asarray(available, dtype=np.int64)))
            used.add(choice)
            out.append(choice)
        if len(out) < self.samples_per_class:
            candidates = self.class_to_positions[int(cls)]
            extra = rng.choice(
                candidates,
                size=self.samples_per_class - len(out),
                replace=True,
            )
            out.extend(int(x) for x in extra.tolist())
        return out

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        n_classes = (
            len(self.eligible_classes)
            if self.classes_per_batch is None
            else min(int(self.classes_per_batch), len(self.eligible_classes))
        )
        classes_np = np.asarray(self.eligible_classes, dtype=np.int64)
        for _ in range(self.batches_per_epoch):
            sampled_classes = rng.choice(classes_np, size=n_classes, replace=False)
            batch: list[int] = []
            for cls in sampled_classes.tolist():
                batch.extend(self._sample_positions_for_class(int(cls), rng))
            rng.shuffle(batch)
            yield batch


def _build_sampler(
    dataset: WindowDataset,
    config: Config,
    batches_per_epoch: int,
    seed: int,
) -> ActivityBalancedBatchSampler:
    return ActivityBalancedBatchSampler(
        labels=dataset.labels,
        subjects=dataset.subjects,
        classes_per_batch=config.classes_per_batch,
        samples_per_class=config.samples_per_class,
        batches_per_epoch=batches_per_epoch,
        seed=seed,
        sample_with_replacement=config.sample_with_replacement,
        subject_balanced_within_class=config.subject_balanced_within_class,
    )


def _run_epoch(
    model: TinierHARSupCon,
    dataloader: DataLoader,
    criterion: SupConLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float | None,
    desc: str,
) -> float:
    train = optimizer is not None
    model.train(train)
    running_loss = 0.0
    running_total = 0
    iterator = tqdm(dataloader, desc=desc, leave=False)
    for batch in iterator:
        x = prepare_inputs(batch["x"]).to(device).float()
        y = batch["y"].to(device).long().view(-1)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            projected = model(x)
            loss = criterion(projected, y)
            if train:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(grad_clip_norm)
                    )
                optimizer.step()
        running_loss += float(loss.item()) * int(y.size(0))
        running_total += int(y.size(0))
        iterator.set_postfix(loss=f"{running_loss / max(1, running_total):.4f}")
    return running_loss / max(1, running_total)


def _sample_tsne_indices(
    loader: Any,
    indices: list[int],
    max_samples_per_subject_activity: int,
    max_samples_per_split: int | None,
    seed: int,
) -> list[int]:
    sorted_indices = sorted(int(idx) for idx in indices)
    labels, subjects = labels_subjects_for_indices(loader, sorted_indices)
    rng = np.random.default_rng(int(seed))
    groups: dict[tuple[int, int], list[int]] = {}
    for window_idx, label, subject in zip(
        sorted_indices, labels.tolist(), subjects.tolist()
    ):
        groups.setdefault((int(subject), int(label)), []).append(int(window_idx))

    selected: list[int] = []
    for (_subject, _activity), group_indices in sorted(groups.items()):
        candidates = np.asarray(group_indices, dtype=np.int64)
        take = min(int(max_samples_per_subject_activity), len(candidates))
        if take > 0:
            selected.extend(
                int(x)
                for x in rng.choice(candidates, size=take, replace=False).tolist()
            )

    if max_samples_per_split is not None and len(selected) > int(max_samples_per_split):
        selected_np = np.asarray(selected, dtype=np.int64)
        selected = [
            int(x)
            for x in rng.choice(
                selected_np,
                size=int(max_samples_per_split),
                replace=False,
            ).tolist()
        ]
    return sorted(selected)


@torch.no_grad()
def _extract_tsne_payload(
    model: TinierHARSupCon,
    loader: Any,
    indices: list[int],
    split_name: str,
    config: Config,
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray]:
    sampled_indices = _sample_tsne_indices(
        loader=loader,
        indices=indices,
        max_samples_per_subject_activity=config.tsne_max_samples_per_subject_activity,
        max_samples_per_split=config.tsne_max_samples_per_split,
        seed=seed,
    )
    dataset = WindowDataset(loader, sampled_indices)
    dataloader = DataLoader(
        dataset,
        batch_size=config.tsne_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    model.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    subjects: list[torch.Tensor] = []
    for batch in dataloader:
        x = prepare_inputs(batch["x"]).to(device).float()
        emb = model(x)
        emb = F.normalize(emb, p=2, dim=1)
        embeddings.append(emb.cpu())
        labels.append(batch["y"].long().view(-1).cpu())
        subjects.append(batch["subject_id"].long().view(-1).cpu())
    return {
        "embeddings": torch.cat(embeddings, dim=0).numpy(),
        "activity_ids": torch.cat(labels, dim=0).numpy(),
        "subject_ids": torch.cat(subjects, dim=0).numpy(),
        "window_indices": np.asarray(sampled_indices, dtype=np.int64),
        "split_names": np.full(len(sampled_indices), split_name, dtype=object),
    }


def _effective_tsne_perplexity(n_samples: int, requested: float) -> float:
    if n_samples < 4:
        raise ValueError(f"Need at least 4 samples for t-SNE, got {n_samples}.")
    return float(min(float(requested), max(2.0, (n_samples - 1) / 3.0)))


def _plot_tsne_subject_color_activity_shape(
    payload: dict[str, np.ndarray],
    activity_names: list[str],
    out_path: Path,
    title: str,
    config: Config,
) -> None:
    embeddings = payload["embeddings"]
    activity_ids = payload["activity_ids"]
    subject_ids = payload["subject_ids"]
    points = TSNE(
        n_components=2,
        perplexity=_effective_tsne_perplexity(
            embeddings.shape[0], config.tsne_perplexity
        ),
        max_iter=int(config.tsne_max_iter),
        learning_rate="auto",
        metric=config.tsne_metric,
        init=config.tsne_init,
        random_state=int(config.seed),
    ).fit_transform(embeddings)

    unique_subjects = sorted(int(x) for x in np.unique(subject_ids).tolist())
    unique_activities = sorted(int(x) for x in np.unique(activity_ids).tolist())
    subject_to_color_idx = {subject: idx for idx, subject in enumerate(unique_subjects)}
    subject_cmap = plt.get_cmap("turbo", max(1, len(unique_subjects)))

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for activity_id in unique_activities:
        mask = activity_ids == int(activity_id)
        marker = MARKERS[int(activity_id) % len(MARKERS)]
        colors = [
            subject_cmap(subject_to_color_idx[int(subject)])
            for subject in subject_ids[mask].tolist()
        ]
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            c=colors,
            marker=marker,
            s=float(config.tsne_point_size),
            alpha=float(config.tsne_point_alpha),
            linewidths=0.35,
            edgecolors="black",
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.grid(True, alpha=0.18)
    subject_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=subject_cmap(subject_to_color_idx[int(subject)]),
            markeredgecolor="black",
            markersize=7,
            label=f"subject {subject}",
        )
        for subject in unique_subjects
    ]
    activity_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[int(activity_id) % len(MARKERS)],
            color="black",
            linestyle="",
            markersize=7,
            label=(
                activity_names[int(activity_id)]
                if int(activity_id) < len(activity_names)
                else f"activity {activity_id}"
            ),
        )
        for activity_id in unique_activities
    ]
    subject_legend = ax.legend(
        handles=subject_handles,
        title="Subject color",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
        ncol=1 if len(subject_handles) <= 14 else 2,
    )
    ax.add_artist(subject_legend)
    ax.legend(
        handles=activity_handles,
        title="Activity shape",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_training_tsne(
    model: TinierHARSupCon,
    loader: Any,
    split_indices: dict[str, list[int]],
    fold_id: str,
    epoch: int,
    split_dir: Path,
    activity_names: list[str],
    config: Config,
    device: torch.device,
) -> dict[str, str]:
    out_paths: dict[str, str] = {}
    tsne_dir = split_dir / "tsne"
    for split_offset, split_name in enumerate(config.tsne_splits):
        if split_name not in split_indices:
            raise ValueError(f"Unknown t-SNE split: {split_name}.")
        payload = _extract_tsne_payload(
            model=model,
            loader=loader,
            indices=split_indices[split_name],
            split_name=split_name,
            config=config,
            device=device,
            seed=int(config.seed) + 10_000 * int(epoch) + split_offset,
        )
        out_path = tsne_dir / f"epoch_{int(epoch):03d}_{split_name}.png"
        _plot_tsne_subject_color_activity_shape(
            payload=payload,
            activity_names=activity_names,
            out_path=out_path,
            title=f"{fold_id} SupCon embeddings epoch {epoch} ({split_name})",
            config=config,
        )
        out_paths[f"{split_name}_tsne_path"] = str(out_path)
    return out_paths


@torch.no_grad()
def _extract_projected_embeddings(
    model: TinierHARSupCon,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in dataloader:
        x = prepare_inputs(batch["x"]).to(device).float()
        y = batch["y"].long().view(-1)
        z = model(x)
        z = F.normalize(z, p=2, dim=1)
        embeddings.append(z.cpu())
        labels.append(y.cpu())
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def _evaluate_projected_proto(
    model: TinierHARSupCon,
    train_dataset: WindowDataset,
    val_dataset: WindowDataset,
    config: Config,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.proto_eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.proto_eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    train_emb, train_y = _extract_projected_embeddings(model, train_loader, device)
    val_emb, val_y = _extract_projected_embeddings(model, val_loader, device)
    prototypes = make_class_prototypes(
        train_emb,
        train_y,
        num_classes=int(num_classes),
        normalize=True,
    )
    logits = cosine_logits(val_emb, prototypes, config.proto_eval_temperature)
    pred = logits.argmax(dim=1).numpy()
    true = val_y.numpy()
    labels = list(range(int(num_classes)))
    return {
        "val_proto_accuracy": float(accuracy_score(true, pred)),
        "val_proto_macro_f1": float(
            f1_score(true, pred, labels=labels, average="macro", zero_division=0)
        ),
    }


def run(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)
    output_root = resolve_output_root(config.output_root, config.dataset_id)
    stage_dir = output_root / config.stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = WHARDatasetID(config.dataset_id)
    cfg = prepare_cfg(
        dataset_id=dataset_id,
        datasets_dir=Path(config.datasets_dir),
        selected_activities=config.selected_activities,
        window_overlap=config.window_overlap,
    )
    pre = PreProcessingPipeline(cfg)
    _raw_df, session_df, window_df = pre.run()
    shared_cfg = SharedConfig(
        dataset_id=config.dataset_id,
        datasets_dir=config.datasets_dir,
        selected_activities=config.selected_activities,
        window_overlap=config.window_overlap,
        val_subjects=config.val_subjects,
        test_subjects=config.test_subjects,
        seed=config.seed,
        split_strategy=config.split_strategy,
        val_percentage=config.val_percentage,
    )
    manifest_path = output_root / "shared_splits" / "loso_subject_folds.json"
    folds = build_or_load_loso_folds(session_df, window_df, shared_cfg, manifest_path)
    if config.max_folds is not None:
        folds = folds[: int(config.max_folds)]

    summary_rows: list[dict[str, Any]] = []
    skipped_folds: list[str] = []
    for fold in folds:
        split = split_indices_for_fold(session_df, window_df, fold)
        split_dir = stage_dir / fold.fold_id
        split_dir.mkdir(parents=True, exist_ok=True)
        fold_fp = config_fingerprint(
            {
                "stage": config.stage_name,
                "config": asdict(config),
                "shared_cfg": asdict(shared_cfg),
                "fold": asdict(fold),
            }
        )
        metrics_path = split_dir / "metrics.json"
        ckpt_path = split_dir / "best_supcon_backbone.pt"
        if not config.force_rerun and metrics_path.exists() and ckpt_path.exists():
            try:
                existing = json.loads(metrics_path.read_text(encoding="utf-8"))
                if existing.get("config_fingerprint") == fold_fp:
                    print(f"[{fold.fold_id}] skipping (already complete)")
                    summary_rows.append(existing)
                    skipped_folds.append(fold.fold_id)
                    continue
            except Exception:
                pass

        loader = build_loader(cfg, session_df, pre, window_df, split.train_indices)
        train_ds = WindowDataset(loader, split.train_indices)
        val_ds = WindowDataset(loader, split.val_indices)
        train_sampler = _build_sampler(
            train_ds,
            config,
            batches_per_epoch=config.batches_per_epoch,
            seed=config.seed + 101,
        )
        val_sampler = _build_sampler(
            val_ds,
            config,
            batches_per_epoch=config.val_batches_per_epoch,
            seed=config.seed + 202,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=config.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_sampler,
            num_workers=config.num_workers,
        )

        window_size = infer_window_size(loader, split.train_indices)
        backbone = build_tinierhar(
            num_channels=int(cfg.num_of_channels),
            num_classes=int(cfg.num_of_activities),
            window_size=window_size,
        )
        model = TinierHARSupCon(
            backbone=backbone,
            projection_hidden_dim=config.projection_hidden_dim,
            projection_dim=config.projection_dim,
        ).to(device)
        criterion = SupConLoss(config.supcon_temperature)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        history: dict[str, Any] = {
            "train_supcon_loss": [],
            "val_supcon_loss": [],
            "val_proto_macro_f1": [],
            "val_proto_accuracy": [],
            "tsne_paths": [],
        }
        best_val_loss = float("inf")
        best_val_proto_macro_f1 = float("-inf")
        best_epoch = -1
        patience_counter = 0
        for epoch in range(1, int(config.epochs) + 1):
            train_loss = _run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                config.grad_clip_norm,
                desc=f"{fold.fold_id} SupCon train {epoch}/{config.epochs}",
            )
            val_loss = _run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                grad_clip_norm=None,
                desc=f"{fold.fold_id} SupCon val {epoch}/{config.epochs}",
            )
            proto_metrics = _evaluate_projected_proto(
                model=model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                config=config,
                device=device,
                num_classes=int(cfg.num_of_activities),
            )
            history["train_supcon_loss"].append(float(train_loss))
            history["val_supcon_loss"].append(float(val_loss))
            history["val_proto_macro_f1"].append(
                float(proto_metrics["val_proto_macro_f1"])
            )
            history["val_proto_accuracy"].append(
                float(proto_metrics["val_proto_accuracy"])
            )
            if config.early_stopping_metric == "val_proto_macro_f1":
                improved = proto_metrics["val_proto_macro_f1"] > (
                    best_val_proto_macro_f1 + float(config.min_delta)
                )
            elif config.early_stopping_metric == "val_supcon_loss":
                improved = val_loss < best_val_loss - float(config.min_delta)
            else:
                raise ValueError(
                    "early_stopping_metric must be 'val_proto_macro_f1' "
                    "or 'val_supcon_loss'."
                )
            if improved:
                best_val_loss = float(val_loss)
                best_val_proto_macro_f1 = float(proto_metrics["val_proto_macro_f1"])
                best_epoch = int(epoch)
                patience_counter = 0
                projection_dim = int(model.projection_head[-1].out_features)
                torch.save(
                    {
                        "backbone_state_dict": model.backbone.state_dict(),
                        "projection_head_state_dict": model.projection_head.state_dict(),
                        "config": asdict(config),
                        "model_meta": {
                            "window_size": int(window_size),
                            "num_channels": int(cfg.num_of_channels),
                            "num_classes": int(cfg.num_of_activities),
                            "class_names": class_names(cfg),
                            "feature_dim": int(2 * model.backbone.nb_units_gru),
                            "projection_hidden_dim": int(config.projection_hidden_dim),
                            "projection_dim": projection_dim,
                        },
                        "fold": asdict(fold),
                        "best_epoch": int(best_epoch),
                        "best_val_supcon_loss": float(best_val_loss),
                        "best_val_proto_macro_f1": float(best_val_proto_macro_f1),
                    },
                    ckpt_path,
                )
            else:
                patience_counter += 1
            print(
                f"[{fold.fold_id} epoch {epoch:03d}] "
                f"train_supcon_loss={train_loss:.4f} val_supcon_loss={val_loss:.4f} "
                f"val_proto_f1={proto_metrics['val_proto_macro_f1']:.4f} "
                f"best_val_supcon_loss={best_val_loss:.4f} "
                f"best_val_proto_f1={best_val_proto_macro_f1:.4f} "
                f"patience={patience_counter}/{config.patience}"
            )
            if (
                int(config.tsne_every_n_epochs) > 0
                and epoch % int(config.tsne_every_n_epochs) == 0
            ):
                tsne_paths = _plot_training_tsne(
                    model=model,
                    loader=loader,
                    split_indices={
                        "train": split.train_indices,
                        "val": split.val_indices,
                        "test": split.test_indices,
                    },
                    fold_id=fold.fold_id,
                    epoch=epoch,
                    split_dir=split_dir,
                    activity_names=class_names(cfg),
                    config=config,
                    device=device,
                )
                history["tsne_paths"].append({"epoch": int(epoch), **tsne_paths})
                print(
                    f"[{fold.fold_id} epoch {epoch:03d}] wrote t-SNE plots: {tsne_paths}"
                )
            if patience_counter >= int(config.patience):
                print(f"[{fold.fold_id}] early stopping at epoch {epoch}.")
                break

        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
            model.projection_head.load_state_dict(
                checkpoint["projection_head_state_dict"]
            )

        result = {
            "config_fingerprint": fold_fp,
            "fold_id": fold.fold_id,
            "train_subject_ids": fold.train_subject_ids,
            "val_subject_ids": fold.val_subject_ids,
            "test_subject_ids": fold.test_subject_ids,
            "best_epoch": int(best_epoch),
            "best_val_supcon_loss": float(best_val_loss),
            "best_val_proto_macro_f1": float(best_val_proto_macro_f1),
            "window_size": int(window_size),
            "num_channels": int(cfg.num_of_channels),
            "num_classes": int(cfg.num_of_activities),
            "projection_dim": int(model.projection_head[-1].out_features),
            "checkpoint_path": str(ckpt_path),
        }
        metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        (split_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        summary_rows.append(result)
        print(
            f"[{fold.fold_id}] SupCon best_val_loss={best_val_loss:.4f} "
            f"best_val_proto_f1={best_val_proto_macro_f1:.4f} "
            f"best_epoch={best_epoch}"
        )

    mean_val = sum(r["best_val_supcon_loss"] for r in summary_rows) / max(
        1, len(summary_rows)
    )
    summary = {
        "config": asdict(config),
        "splits_manifest_path": str(manifest_path),
        "num_folds": len(summary_rows),
        "skipped_folds": skipped_folds,
        "mean_best_val_supcon_loss": float(mean_val),
        "mean_best_val_proto_macro_f1": float(
            sum(r["best_val_proto_macro_f1"] for r in summary_rows)
            / max(1, len(summary_rows))
        ),
        "folds": summary_rows,
    }
    (stage_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from whar_datasets import Loader, PostProcessingPipeline, PreProcessingPipeline, WHARDatasetID

ROOT = Path(__file__).resolve().parents[1]
PHASE1_SCRIPT_PATH = ROOT / "style-scripts" / "phase1_contrastive_set_encoder.py"
PHASE2_SCRIPT_PATH = ROOT / "style-scripts" / "phase2_contrastive_set_encoder_subject_disjoint.py"


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PHASE1 = _load_module_from_path("phase1_for_phase3_supcon", PHASE1_SCRIPT_PATH)
PHASE2 = _load_module_from_path("phase2_for_phase3_supcon", PHASE2_SCRIPT_PATH)

ContrastiveSetEncoder = PHASE1.ContrastiveSetEncoder
build_model = PHASE1.build_model
evaluate_subject_knn = PHASE1.evaluate_subject_knn
indices_by_subject = PHASE1.indices_by_subject
infer_window_size = PHASE1.infer_window_size
plot_split_tsne = PHASE1.plot_split_tsne
prepare_one_activity_cfg = PHASE1.prepare_one_activity_cfg
sample_window_array = PHASE1.sample_window_array
split_subject_disjoint = PHASE2.split_subject_disjoint


@dataclass(frozen=True)
class Phase3Config:
    dataset_id: str = WHARDatasetID.WEAR.value
    activity: str = "Walking"
    encoder: str = "attention"
    set_size: int = 8  # fixed K used for plotting/evaluation embeddings
    train_min_set_size: int = 1
    train_max_set_size: int = 32
    n_subjects_per_batch: int = 10
    m_sets_per_subject: int = 2
    train_batches_per_epoch: int = 128
    val_batches_per_epoch: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    supcon_temperature: float = 0.1
    projection_hidden_dim: int = 128
    projection_dim: int = 64
    eval_sets_per_subject: int = 24
    tsne_every_n_epochs: int = 1
    enforce_strict_nm_batching: bool = True
    subject_train_fraction: float = 0.7
    subject_val_fraction: float = 0.15
    seed: int = 0
    output_dir: str = str(ROOT / "artifacts" / "phase3_supcon_subject_disjoint")
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


RUN_CONFIG = Phase3Config()


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        sim_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - sim_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)

        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        return (-mean_log_prob_pos).mean()


@dataclass(frozen=True)
class SubjectSetIndices:
    subject_id: int
    set_indices: np.ndarray


class NMSetsBatchSampler(Sampler[list[SubjectSetIndices]]):
    """Samples N subjects and M disjoint K-window sets per subject."""

    def __init__(
        self,
        indices_by_subject: Mapping[int, np.ndarray],
        min_set_size: int,
        max_set_size: int,
        n_subjects: int,
        m_sets_per_subject: int,
        batches_per_epoch: int,
        seed: int,
    ) -> None:
        self.indices_by_subject = {
            int(subject_id): np.asarray(indices, dtype=np.int64)
            for subject_id, indices in indices_by_subject.items()
        }
        self.min_set_size = int(min_set_size)
        self.max_set_size = int(max_set_size)
        if self.min_set_size < 1:
            raise ValueError("min_set_size must be >= 1.")
        if self.max_set_size < self.min_set_size:
            raise ValueError("max_set_size must be >= min_set_size.")
        self.n_subjects = int(n_subjects)
        self.m_sets_per_subject = int(m_sets_per_subject)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

        self.eligible_subject_ids_by_k: dict[int, list[int]] = {}
        for k in range(self.min_set_size, self.max_set_size + 1):
            min_windows = self.m_sets_per_subject * k
            eligible = sorted(
                subject_id
                for subject_id, indices in self.indices_by_subject.items()
                if len(indices) >= min_windows
            )
            if len(eligible) >= self.n_subjects:
                self.eligible_subject_ids_by_k[k] = eligible
        self.valid_set_sizes = tuple(sorted(self.eligible_subject_ids_by_k.keys()))
        if not self.valid_set_sizes:
            raise ValueError(
                f"Need at least {self.n_subjects} eligible subjects for N x M sampling, "
                f"but found no feasible K in range [{self.min_set_size}, {self.max_set_size}]."
            )

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[SubjectSetIndices]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        valid_k = np.asarray(self.valid_set_sizes, dtype=np.int64)
        for _ in range(self.batches_per_epoch):
            set_size = int(rng.choice(valid_k))
            eligible = np.asarray(self.eligible_subject_ids_by_k[set_size], dtype=np.int64)
            sampled_subjects = rng.choice(eligible, size=self.n_subjects, replace=False)
            batch: list[SubjectSetIndices] = []
            for subject_id in sampled_subjects.tolist():
                candidate = self.indices_by_subject[int(subject_id)]
                picked = rng.choice(
                    candidate, size=self.m_sets_per_subject * set_size, replace=False
                )
                for set_idx in range(self.m_sets_per_subject):
                    start = set_idx * set_size
                    end = (set_idx + 1) * set_size
                    batch.append(
                        SubjectSetIndices(
                            subject_id=int(subject_id),
                            set_indices=picked[start:end],
                        )
                    )
            rng.shuffle(batch)
            yield batch


class NMSetsDataset(Dataset[dict[str, torch.Tensor | int]]):
    def __init__(self, loader: Loader) -> None:
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader.window_df)

    def __getitem__(self, item: SubjectSetIndices) -> dict[str, torch.Tensor | int]:
        if not isinstance(item, SubjectSetIndices):
            raise TypeError("NMSetsDataset expects SubjectSetIndices from NMSetsBatchSampler.")
        windows = [
            torch.from_numpy(sample_window_array(self.loader, int(idx)).copy())
            .float()
            .unsqueeze(0)
            for idx in item.set_indices.tolist()
        ]
        x_set = torch.stack(windows, dim=0)  # (K, 1, T, S)
        return {"x_set": x_set, "subject_id": int(item.subject_id)}


def nm_collate(samples: Sequence[dict[str, torch.Tensor | int]]) -> dict[str, torch.Tensor]:
    sets = [sample["x_set"] for sample in samples]
    if not all(isinstance(x, torch.Tensor) for x in sets):
        raise TypeError("All x_set entries must be tensors.")
    x_batch = torch.stack([x for x in sets if isinstance(x, torch.Tensor)], dim=0)
    y_subject = torch.tensor([int(sample["subject_id"]) for sample in samples], dtype=torch.long)
    return {"x": x_batch, "subject_id": y_subject}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_nm_batch_labels(
    labels: torch.Tensor,
    expected_n_subjects: int,
    expected_m_sets: int,
) -> None:
    labels_np = labels.detach().cpu().numpy()
    unique_labels, counts = np.unique(labels_np, return_counts=True)
    if int(len(unique_labels)) != int(expected_n_subjects):
        raise ValueError(
            "N x M batch violation: unexpected number of subjects. "
            f"Expected N={expected_n_subjects}, got {len(unique_labels)}."
        )
    bad_counts = [int(c) for c in counts.tolist() if int(c) != int(expected_m_sets)]
    if bad_counts:
        raise ValueError(
            "N x M batch violation: at least one subject does not have exactly M sets. "
            f"Expected M={expected_m_sets}, got counts={counts.tolist()}."
        )


def _phase1_eval_cfg(cfg: Phase3Config) -> Any:
    return PHASE1.ContrastiveRunConfig(
        dataset_id=cfg.dataset_id,
        activity=cfg.activity,
        encoder=cfg.encoder,
        set_size=cfg.set_size,
        train_min_set_size=1,
        train_max_set_size=1,
        triplets_per_batch=cfg.n_subjects_per_batch * cfg.m_sets_per_subject,
        batches_per_epoch=1,
        epochs=1,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        margin=1.0,
        projection_hidden_dim=cfg.projection_hidden_dim,
        projection_dim=cfg.projection_dim,
        eval_sets_per_subject=cfg.eval_sets_per_subject,
        tsne_every_n_epochs=1,
        knn_neighbors=1,
        seed=cfg.seed,
        val_size=0.15,
        test_size=0.15,
        output_dir=cfg.output_dir,
        device=cfg.device,
    )


def evaluate_nm_losses(
    model: ContrastiveSetEncoder,
    loader: Loader,
    indices: Sequence[int],
    config: Phase3Config,
    supcon_loss_fn: SupConLoss,
    device: torch.device,
    seed_offset: int,
) -> float:
    by_subject = indices_by_subject(
        loader, indices, min_windows=config.m_sets_per_subject * config.train_min_set_size
    )
    max_eligible_subject_count = 0
    for k in range(config.train_min_set_size, config.train_max_set_size + 1):
        min_windows = config.m_sets_per_subject * k
        eligible_count = sum(1 for _sid, arr in by_subject.items() if len(arr) >= min_windows)
        max_eligible_subject_count = max(max_eligible_subject_count, eligible_count)
    eval_n_subjects = min(config.n_subjects_per_batch, max_eligible_subject_count)
    if eval_n_subjects < 2:
        raise ValueError(
            "Need at least 2 eligible subjects for SupCon evaluation batching, "
            f"but found {max_eligible_subject_count}."
        )
    sampler = NMSetsBatchSampler(
        indices_by_subject=by_subject,
        min_set_size=config.train_min_set_size,
        max_set_size=config.train_max_set_size,
        n_subjects=eval_n_subjects,
        m_sets_per_subject=config.m_sets_per_subject,
        batches_per_epoch=config.val_batches_per_epoch,
        seed=config.seed + seed_offset,
    )
    dataset = NMSetsDataset(loader)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=nm_collate, num_workers=0)
    model.eval()
    supcon_losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            subject_ids = batch["subject_id"].to(device)
            if config.enforce_strict_nm_batching:
                validate_nm_batch_labels(
                    subject_ids,
                    expected_n_subjects=eval_n_subjects,
                    expected_m_sets=config.m_sets_per_subject,
                )
            y_support = torch.zeros((x.size(0), x.size(1)), dtype=torch.long, device=device)
            projected = model(x, y_support)
            loss_supcon = supcon_loss_fn(projected, subject_ids)
            supcon_losses.append(float(loss_supcon.item()))
    if not supcon_losses:
        return float("inf")
    return float(np.mean(supcon_losses))


def run_phase3_training(config: Phase3Config) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    dataset_id = WHARDatasetID(config.dataset_id)
    cfg, selected_activity = prepare_one_activity_cfg(
        dataset_id=dataset_id,
        datasets_dir=ROOT / "datasets",
        activity=config.activity,
    )
    pre_pipeline = PreProcessingPipeline(cfg)
    _raw_df, session_df, window_df = pre_pipeline.run()

    splits = split_subject_disjoint(
        session_df=session_df,
        window_df=window_df,
        train_fraction=config.subject_train_fraction,
        val_fraction=config.subject_val_fraction,
        seed=config.seed,
    )
    print(
        "Phase 3 subject-disjoint split: "
        f"train_subjects={splits.train_subject_ids} "
        f"val_subjects={splits.val_subject_ids} "
        f"test_subjects={splits.test_subject_ids}"
    )

    post_pipeline = PostProcessingPipeline(cfg, pre_pipeline, window_df, splits.train_indices)
    samples = post_pipeline.run()
    loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)

    window_size = infer_window_size(loader, splits.train_indices)
    model = build_model(
        encoder_name=config.encoder,
        num_channels=cfg.num_of_channels,
        window_size=window_size,
        projection_hidden_dim=config.projection_hidden_dim,
        projection_dim=config.projection_dim,
    )
    model.to(device)

    train_by_subject = indices_by_subject(
        loader,
        splits.train_indices,
        min_windows=config.m_sets_per_subject * config.train_min_set_size,
    )
    train_sampler = NMSetsBatchSampler(
        indices_by_subject=train_by_subject,
        min_set_size=config.train_min_set_size,
        max_set_size=config.train_max_set_size,
        n_subjects=config.n_subjects_per_batch,
        m_sets_per_subject=config.m_sets_per_subject,
        batches_per_epoch=config.train_batches_per_epoch,
        seed=config.seed,
    )
    train_dataset = NMSetsDataset(loader)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=nm_collate,
        num_workers=0,
    )
    print(
        f"Training with N x M = {config.n_subjects_per_batch} x {config.m_sets_per_subject}, "
        f"random K in [{config.train_min_set_size}, {config.train_max_set_size}], "
        f"feasible K values={train_sampler.valid_set_sizes}."
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    supcon_loss_fn = SupConLoss(temperature=config.supcon_temperature)
    phase1_eval_cfg = _phase1_eval_cfg(config)

    history: list[dict[str, float | int | str]] = []
    best_val_supcon = float("inf")
    checkpoint_path = output_dir / "best_phase3_supcon_subject_disjoint.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_supcon_losses: list[float] = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for batch in progress:
            x = batch["x"].to(device)
            subject_ids = batch["subject_id"].to(device)
            if config.enforce_strict_nm_batching:
                validate_nm_batch_labels(
                    subject_ids,
                    expected_n_subjects=config.n_subjects_per_batch,
                    expected_m_sets=config.m_sets_per_subject,
                )
            y_support = torch.zeros((x.size(0), x.size(1)), dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            projected = model(x, y_support)
            loss_supcon = supcon_loss_fn(projected, subject_ids)
            loss_supcon.backward()
            optimizer.step()

            train_supcon_losses.append(float(loss_supcon.item()))
            progress.set_postfix(
                supcon=f"{np.mean(train_supcon_losses):.4f}",
            )

        val_supcon = evaluate_nm_losses(
            model=model,
            loader=loader,
            indices=splits.val_indices,
            config=config,
            supcon_loss_fn=supcon_loss_fn,
            device=device,
            seed_offset=5000 + epoch,
        )
        epoch_row: dict[str, float | int | str] = {
            "epoch": epoch,
            "train_supcon_loss": float(np.mean(train_supcon_losses)),
            "val_supcon_loss": val_supcon,
        }

        if config.tsne_every_n_epochs > 0 and epoch % config.tsne_every_n_epochs == 0:
            epoch_row["train_tsne_path"] = plot_split_tsne(
                model=model,
                loader=loader,
                indices=splits.train_indices,
                split_name=f"phase3_train_epoch_{epoch:03d}",
                output_dir=output_dir,
                config=phase1_eval_cfg,
                device=device,
                seed_offset=7000 + epoch,
            ) or ""
            epoch_row["val_tsne_path"] = plot_split_tsne(
                model=model,
                loader=loader,
                indices=splits.val_indices,
                split_name=f"phase3_val_epoch_{epoch:03d}",
                output_dir=output_dir,
                config=phase1_eval_cfg,
                device=device,
                seed_offset=8000 + epoch,
            ) or ""

        history.append(epoch_row)
        print(
            f"train_supcon={epoch_row['train_supcon_loss']:.4f}, "
            f"val_supcon={epoch_row['val_supcon_loss']:.4f}"
        )

        if val_supcon < best_val_supcon:
            best_val_supcon = val_supcon
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "selected_activity": selected_activity,
                    "window_size": window_size,
                    "best_epoch": epoch,
                    "best_val_supcon_loss": best_val_supcon,
                    "subject_split": asdict(splits),
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_tsne = plot_split_tsne(
        model=model,
        loader=loader,
        indices=splits.train_indices,
        split_name="phase3_train",
        output_dir=output_dir,
        config=phase1_eval_cfg,
        device=device,
        seed_offset=101,
    )
    val_tsne = plot_split_tsne(
        model=model,
        loader=loader,
        indices=splits.val_indices,
        split_name="phase3_val",
        output_dir=output_dir,
        config=phase1_eval_cfg,
        device=device,
        seed_offset=102,
    )
    test_tsne = plot_split_tsne(
        model=model,
        loader=loader,
        indices=splits.test_indices,
        split_name="phase3_test",
        output_dir=output_dir,
        config=phase1_eval_cfg,
        device=device,
        seed_offset=103,
    )

    # Diagnostic only: subject-disjoint KNN accuracy should not be interpreted as a main objective.
    test_knn_diag = evaluate_subject_knn(
        model=model,
        loader=loader,
        train_indices=splits.train_indices,
        eval_indices=splits.test_indices,
        split_name="phase3_test_subject_disjoint",
        output_dir=output_dir,
        config=phase1_eval_cfg,
        device=device,
    )

    result = {
        "config": asdict(config),
        "selected_activity": selected_activity,
        "subject_split": asdict(splits),
        "history": history,
        "best_checkpoint_path": str(checkpoint_path),
        "best_val_supcon_loss": float(best_val_supcon),
        "final_tsne": {"train": train_tsne, "val": val_tsne, "test": test_tsne},
        "diagnostic_test_knn_subject_disjoint": asdict(test_knn_diag),
    }
    with (output_dir / "phase3_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with (output_dir / "phase3_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Finished Phase 3. Best val SupCon loss: {best_val_supcon:.4f}")
    print(f"Saved checkpoint: {checkpoint_path}")
    return result


def main() -> None:
    run_phase3_training(RUN_CONFIG)


if __name__ == "__main__":
    main()

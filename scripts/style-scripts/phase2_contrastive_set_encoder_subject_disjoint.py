from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from whar_datasets import Loader, PostProcessingPipeline, PreProcessingPipeline, WHARDatasetID

ROOT = Path(__file__).resolve().parents[1]
PHASE1_SCRIPT_PATH = ROOT / "style-scripts" / "phase1_contrastive_set_encoder.py"


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PHASE1 = _load_module_from_path("phase1_contrastive_module_for_phase2", PHASE1_SCRIPT_PATH)

ContrastiveRunConfig = PHASE1.ContrastiveRunConfig
ContrastiveSetEncoder = PHASE1.ContrastiveSetEncoder
ContrastiveTripletDataset = PHASE1.ContrastiveTripletDataset
SubjectTripletBatchSampler = PHASE1.SubjectTripletBatchSampler
build_model = PHASE1.build_model
evaluate_subject_knn = PHASE1.evaluate_subject_knn
extract_subject_embeddings = PHASE1.extract_subject_embeddings
indices_by_subject = PHASE1.indices_by_subject
plot_split_tsne = PHASE1.plot_split_tsne
prepare_one_activity_cfg = PHASE1.prepare_one_activity_cfg
set_seed = PHASE1.set_seed
triplet_collate = PHASE1.triplet_collate


@dataclass(frozen=True)
class SubjectDisjointSplits:
    train_subject_ids: list[int]
    val_subject_ids: list[int]
    test_subject_ids: list[int]
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]


@dataclass(frozen=True)
class Phase2Config:
    dataset_id: str = WHARDatasetID.WEAR.value
    activity: str = "Walking"
    encoder: str = "attention"
    set_size: int = 8
    train_min_set_size: int = 1
    train_max_set_size: int = 32
    triplets_per_batch: int = 16
    train_batches_per_epoch: int = 128
    val_batches_per_epoch: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    margin: float = 1.0
    projection_hidden_dim: int = 128
    projection_dim: int = 64
    eval_sets_per_subject: int = 24
    tsne_every_n_epochs: int = 1
    subject_train_fraction: float = 0.7
    subject_val_fraction: float = 0.15
    seed: int = 0
    output_dir: str = str(ROOT / "artifacts" / "phase2_subject_disjoint")
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


RUN_CONFIG = Phase2Config()


def split_subject_disjoint(
    session_df: pd.DataFrame,
    window_df: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> SubjectDisjointSplits:
    subjects = sorted(int(x) for x in session_df["subject_id"].unique().tolist())
    if len(subjects) < 3:
        raise ValueError("Need at least 3 subjects for train/val/test disjoint split.")
    rng = np.random.default_rng(seed)
    shuffled = [int(x) for x in rng.permutation(np.asarray(subjects, dtype=np.int64))]

    n_total = len(shuffled)
    n_train = max(1, int(round(n_total * train_fraction)))
    n_val = max(1, int(round(n_total * val_fraction)))
    n_train = min(n_train, n_total - 2)
    n_val = min(n_val, n_total - n_train - 1)
    if n_total - n_train - n_val < 1:
        n_val = max(1, n_total - n_train - 1)
    n_test = n_total - n_train - n_val
    if n_test < 1:
        raise ValueError("Split produced no test subjects.")

    train_subject_ids = sorted(shuffled[:n_train])
    val_subject_ids = sorted(shuffled[n_train : n_train + n_val])
    test_subject_ids = sorted(shuffled[n_train + n_val :])

    window_meta = window_df[["session_id"]].copy()
    window_meta["window_index"] = window_meta.index.astype(int)
    session_meta = (
        session_df[["session_id", "subject_id"]].drop_duplicates("session_id").copy()
    )
    merged = window_meta.merge(session_meta, on="session_id", how="left")

    def _indices_for(subject_ids: Sequence[int]) -> list[int]:
        mask = merged["subject_id"].isin(list(subject_ids))
        return sorted(int(x) for x in merged.loc[mask, "window_index"].tolist())

    return SubjectDisjointSplits(
        train_subject_ids=train_subject_ids,
        val_subject_ids=val_subject_ids,
        test_subject_ids=test_subject_ids,
        train_indices=_indices_for(train_subject_ids),
        val_indices=_indices_for(val_subject_ids),
        test_indices=_indices_for(test_subject_ids),
    )


def _to_phase1_eval_config(cfg: Phase2Config) -> ContrastiveRunConfig:
    return ContrastiveRunConfig(
        dataset_id=cfg.dataset_id,
        activity=cfg.activity,
        encoder=cfg.encoder,
        set_size=cfg.set_size,
        train_min_set_size=cfg.train_min_set_size,
        train_max_set_size=cfg.train_max_set_size,
        triplets_per_batch=cfg.triplets_per_batch,
        batches_per_epoch=cfg.train_batches_per_epoch,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        margin=cfg.margin,
        projection_hidden_dim=cfg.projection_hidden_dim,
        projection_dim=cfg.projection_dim,
        eval_sets_per_subject=cfg.eval_sets_per_subject,
        tsne_every_n_epochs=cfg.tsne_every_n_epochs,
        knn_neighbors=1,
        seed=cfg.seed,
        val_size=0.15,
        test_size=0.15,
        output_dir=cfg.output_dir,
        device=cfg.device,
    )


def _evaluate_triplet_loss(
    model: ContrastiveSetEncoder,
    loader: Loader,
    indices: Sequence[int],
    config: Phase2Config,
    device: torch.device,
    seed_offset: int,
) -> float:
    by_subject = indices_by_subject(loader, indices, min_windows=2 * config.train_min_set_size)
    sampler = SubjectTripletBatchSampler(
        indices_by_subject=by_subject,
        min_set_size=config.train_min_set_size,
        max_set_size=config.train_max_set_size,
        triplets_per_batch=config.triplets_per_batch,
        batches_per_epoch=config.val_batches_per_epoch,
        seed=config.seed + seed_offset,
    )
    dataset = ContrastiveTripletDataset(loader)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=triplet_collate,
        num_workers=0,
    )
    criterion = nn.TripletMarginLoss(margin=config.margin)
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            anchor = batch["anchor"].to(device)
            positive = batch["positive"].to(device)
            negative = batch["negative"].to(device)
            y_support = torch.zeros(
                (anchor.size(0), anchor.size(1)), dtype=torch.long, device=device
            )
            embed_anchor = model(anchor, y_support)
            embed_positive = model(positive, y_support)
            embed_negative = model(negative, y_support)
            losses.append(float(criterion(embed_anchor, embed_positive, embed_negative).item()))
    return float(np.mean(losses)) if losses else float("inf")


def run_phase2_training(config: Phase2Config) -> dict[str, Any]:
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
        "Phase 2 subject-disjoint split: "
        f"train_subjects={splits.train_subject_ids} "
        f"val_subjects={splits.val_subject_ids} "
        f"test_subjects={splits.test_subject_ids}"
    )
    print(
        "Window counts: "
        f"train={len(splits.train_indices)} val={len(splits.val_indices)} test={len(splits.test_indices)}"
    )

    post_pipeline = PostProcessingPipeline(
        cfg, pre_pipeline, window_df, splits.train_indices
    )
    samples = post_pipeline.run()
    loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)

    phase1_eval_cfg = _to_phase1_eval_config(config)
    window_size = PHASE1.infer_window_size(loader, splits.train_indices)
    model = build_model(
        encoder_name=config.encoder,
        num_channels=cfg.num_of_channels,
        window_size=window_size,
        projection_hidden_dim=config.projection_hidden_dim,
        projection_dim=config.projection_dim,
    )
    model.to(device)

    train_by_subject = indices_by_subject(
        loader, splits.train_indices, min_windows=2 * config.train_min_set_size
    )
    train_sampler = SubjectTripletBatchSampler(
        indices_by_subject=train_by_subject,
        min_set_size=config.train_min_set_size,
        max_set_size=config.train_max_set_size,
        triplets_per_batch=config.triplets_per_batch,
        batches_per_epoch=config.train_batches_per_epoch,
        seed=config.seed,
    )
    dataset = ContrastiveTripletDataset(loader)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_sampler,
        collate_fn=triplet_collate,
        num_workers=0,
    )
    print(
        f"Training random K in [{config.train_min_set_size}, {config.train_max_set_size}] "
        f"with feasible K values={train_sampler.valid_set_sizes}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.TripletMarginLoss(margin=config.margin)
    history: list[dict[str, float | int | str]] = []
    best_val_loss = float("inf")
    checkpoint_path = output_dir / "best_phase2_subject_disjoint.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for batch in progress:
            anchor = batch["anchor"].to(device)
            positive = batch["positive"].to(device)
            negative = batch["negative"].to(device)
            y_support = torch.zeros(
                (anchor.size(0), anchor.size(1)), dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            embed_anchor = model(anchor, y_support)
            embed_positive = model(positive, y_support)
            embed_negative = model(negative, y_support)
            loss = criterion(embed_anchor, embed_positive, embed_negative)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{np.mean(train_losses):.4f}")

        val_triplet_loss = _evaluate_triplet_loss(
            model=model,
            loader=loader,
            indices=splits.val_indices,
            config=config,
            device=device,
            seed_offset=7000 + epoch,
        )
        epoch_row: dict[str, float | int | str] = {
            "epoch": epoch,
            "train_triplet_loss": float(np.mean(train_losses)),
            "val_triplet_loss": float(val_triplet_loss),
        }
        if config.tsne_every_n_epochs > 0 and epoch % config.tsne_every_n_epochs == 0:
            epoch_row["train_tsne_path"] = plot_split_tsne(
                model=model,
                loader=loader,
                indices=splits.train_indices,
                split_name=f"phase2_train_epoch_{epoch:03d}",
                output_dir=output_dir,
                config=phase1_eval_cfg,
                device=device,
                seed_offset=8000 + epoch,
            ) or ""
            epoch_row["val_tsne_path"] = plot_split_tsne(
                model=model,
                loader=loader,
                indices=splits.val_indices,
                split_name=f"phase2_val_epoch_{epoch:03d}",
                output_dir=output_dir,
                config=phase1_eval_cfg,
                device=device,
                seed_offset=9000 + epoch,
            ) or ""
        history.append(epoch_row)
        print(
            f"Epoch {epoch:03d}: train_triplet_loss={epoch_row['train_triplet_loss']:.4f}, "
            f"val_triplet_loss={epoch_row['val_triplet_loss']:.4f}"
        )

        if val_triplet_loss < best_val_loss:
            best_val_loss = val_triplet_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "selected_activity": selected_activity,
                    "window_size": window_size,
                    "best_epoch": epoch,
                    "best_val_triplet_loss": best_val_loss,
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
        split_name="phase2_train",
        output_dir=output_dir,
        config=phase1_eval_cfg,
        device=device,
        seed_offset=101,
    )
    val_tsne = plot_split_tsne(
        model=model,
        loader=loader,
        indices=splits.val_indices,
        split_name="phase2_val",
        output_dir=output_dir,
        config=phase1_eval_cfg,
        device=device,
        seed_offset=102,
    )
    test_tsne = plot_split_tsne(
        model=model,
        loader=loader,
        indices=splits.test_indices,
        split_name="phase2_test",
        output_dir=output_dir,
        config=phase1_eval_cfg,
        device=device,
        seed_offset=103,
    )

    # KNN here is only diagnostic and expected to fail with unseen subject labels.
    test_knn = evaluate_subject_knn(
        model=model,
        loader=loader,
        train_indices=splits.train_indices,
        eval_indices=splits.test_indices,
        split_name="phase2_test_subject_disjoint",
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
        "best_val_triplet_loss": float(best_val_loss),
        "final_tsne": {
            "train": train_tsne,
            "val": val_tsne,
            "test": test_tsne,
        },
        "diagnostic_test_knn_subject_disjoint": asdict(test_knn),
    }
    with (output_dir / "phase2_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with (output_dir / "phase2_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Finished Phase 2. Best val triplet loss: {best_val_loss:.4f}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved final t-SNE train/val/test under: {output_dir}")
    return result


def main() -> None:
    run_phase2_training(RUN_CONFIG)


if __name__ == "__main__":
    main()

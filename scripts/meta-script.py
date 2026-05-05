from __future__ import annotations

import importlib.util
import inspect
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from whar_datasets import (
    Loader,
    PostProcessingPipeline,
    PreProcessingPipeline,
    WHARDatasetID,
    get_dataset_cfg,
)

from hyper_har.backbone.tinierhar import TinierHAR
from hyper_har.config import DEFAULT_CONFIG
from hyper_har.hypernet.hypernet import HyperNet
from hyper_har.splitting import MetaLOSOSplitter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SET_ENCODER_SIMPLE_MODULE = _load_module_from_path(
    "hyper_har_set_encoder_simple", SRC / "hyper_har" / "set-encoder" / "simple.py"
)
SET_ENCODER_ATTENTION_MODULE = _load_module_from_path(
    "hyper_har_set_encoder_attention",
    SRC / "hyper_har" / "set-encoder" / "attention.py",
)
META_TRAINER_MODULE = _load_module_from_path(
    "hyper_har_meta_trainer", SRC / "hyper_har" / "training" / "meta-trainer.py"
)

PrototypicalSetEncoder = SET_ENCODER_SIMPLE_MODULE.PrototypicalSetEncoder
AttentionSetEncoder = SET_ENCODER_ATTENTION_MODULE.AttentionSetEncoder
MetaTrainerConfig = META_TRAINER_MODULE.MetaTrainerConfig
SetToLoRAMetaTrainer = META_TRAINER_MODULE.SetToLoRAMetaTrainer


# Script-level run settings
DATASET_ID = WHARDatasetID.WEAR
BACKBONE_DIR = ROOT / "artifacts" / "loso_cv"
OUTPUT_ROOT_DIR = ROOT / "artifacts" / "meta_loso_cv"
META_VARIANT_NAME = os.getenv("META_VARIANT_NAME")

# Meta episodic settings
TRAIN_SUBJECTS_PER_EPISODE = 4
SUPPORT_PER_CLASS = 20  # 4
QUERY_PER_CLASS = 8
TRAIN_EPISODES_PER_EPOCH = 64
EVAL_EPISODES = 32
USE_VMAP = True
SET_ENCODER_KIND = "prototypical"  # "prototypical" or "attention"


@dataclass
class SplitMetaResult:
    split_index: int
    subject_id: int
    best_epoch: int
    best_val_loss: float
    train_loss_at_best: float
    train_macro_f1_at_best: float
    val_macro_f1_at_best: float
    test_loss: float
    test_macro_f1: float
    checkpoint_path: str


def _slugify_path_component(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "run"


def _resolve_variant_name(train_cfg: Any) -> str:
    if META_VARIANT_NAME and META_VARIANT_NAME.strip():
        return _slugify_path_component(META_VARIANT_NAME)

    parts = [
        f"enc-{SET_ENCODER_KIND}",
        f"b-{TRAIN_SUBJECTS_PER_EPISODE}",
        f"k-{SUPPORT_PER_CLASS}",
        f"q-{QUERY_PER_CLASS}",
        f"train-ep-{TRAIN_EPISODES_PER_EPOCH}",
        f"eval-ep-{EVAL_EPISODES}",
        f"vmap-{int(USE_VMAP)}",
        f"lr-{train_cfg.learning_rate}",
        f"wd-{train_cfg.weight_decay}",
        f"epochs-{train_cfg.num_epochs}",
        f"pat-{train_cfg.patience}",
    ]
    return _slugify_path_component("_".join(parts))


def _is_completed_meta_split(split_dir: Path) -> bool:
    required_files = [
        split_dir / "best_meta_modules.pt",
        split_dir / "meta_metrics.json",
        split_dir / "meta_history.json",
    ]
    return all(path.exists() for path in required_files)


def _load_existing_meta_metrics(split_dir: Path) -> dict[str, Any] | None:
    metrics_path = split_dir / "meta_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except (json.JSONDecodeError, OSError):
        return None
    return None


def _fetch_class_weights(
    loader: object, split: object, num_classes: int
) -> torch.Tensor | None:
    get_weights = getattr(loader, "get_class_weights", None)
    if get_weights is None:
        return None

    weights_obj = None
    try:
        n_params = len(inspect.signature(get_weights).parameters)
    except (TypeError, ValueError):
        n_params = -1

    if n_params in (0, -1):
        try:
            weights_obj = get_weights()
        except TypeError:
            pass

    if weights_obj is None and n_params in (1, -1):
        train_indices = getattr(split, "train_indices", None)
        if train_indices is not None:
            try:
                weights_obj = get_weights(train_indices)
            except TypeError:
                pass

    if weights_obj is None:
        return None

    if isinstance(weights_obj, dict):
        weights = torch.ones(num_classes, dtype=torch.float32)
        for class_id in range(num_classes):
            raw_w = float(weights_obj.get(class_id, 1.0))
            # whar_datasets uses -1 for classes absent in windows -> ignore via zero weight.
            weights[class_id] = 0.0 if raw_w < 0.0 else raw_w
        return weights

    weights = torch.as_tensor(weights_obj, dtype=torch.float32).view(-1)
    if weights.numel() != num_classes:
        raise ValueError(
            f"Class weights length mismatch: expected {num_classes}, got {weights.numel()}."
        )
    weights = torch.where(weights < 0.0, torch.zeros_like(weights), weights)
    return weights


def _infer_window_size(loader: Loader, indices: Sequence[int]) -> int:
    if len(indices) == 0:
        raise ValueError("Cannot infer window size from an empty index set.")
    sample = loader.get_sample(int(indices[0]))
    if not sample:
        raise ValueError("Could not infer window size: empty sample.")
    x = np.asarray(sample[0])
    if x.ndim == 2:
        return int(x.shape[0])
    if x.ndim == 3 and x.shape[0] == 1:
        return int(x.shape[1])
    raise ValueError(f"Unexpected sample shape for window inference: {tuple(x.shape)}")


def _infer_subject_id(loader: Loader, indices: Sequence[int], fallback: int) -> int:
    if len(indices) == 0:
        return fallback

    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    session_meta = (
        loader.session_df[["session_id", "subject_id"]]
        .drop_duplicates("session_id")
        .set_index("session_id")
    )
    merged = subset.join(session_meta, on="session_id", how="left")
    subjects = merged["subject_id"].dropna().astype(int).unique().tolist()
    if len(subjects) == 1:
        return int(subjects[0])
    return fallback


def _subject_ids_for_indices(loader: Loader, indices: Sequence[int]) -> list[int]:
    if len(indices) == 0:
        return []
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    session_meta = (
        loader.session_df[["session_id", "subject_id"]]
        .drop_duplicates("session_id")
        .set_index("session_id")
    )
    merged = subset.join(session_meta, on="session_id", how="left")
    if merged["subject_id"].isna().any():
        raise ValueError(
            "Missing subject_id while inferring subject partition debug info."
        )
    return sorted(set(int(x) for x in merged["subject_id"].tolist()))


def _activity_support_by_subject(
    loader: Loader, indices: Sequence[int]
) -> tuple[dict[int, dict[int, int]], list[int]]:
    subset = loader.window_df.loc[list(indices), ["session_id"]].copy()
    subset["window_index"] = subset.index.astype(int)

    session_meta = loader.session_df[
        ["session_id", "subject_id", "activity_id"]
    ].drop_duplicates("session_id")
    merged = subset.merge(session_meta, on="session_id", how="left")

    if merged["subject_id"].isna().any() or merged["activity_id"].isna().any():
        raise ValueError(
            "Missing subject/activity metadata while preparing episodic splits."
        )

    counts = (
        merged.groupby(["subject_id", "activity_id"])
        .size()
        .rename("count")
        .reset_index()
    )
    activity_ids = sorted(int(x) for x in counts["activity_id"].unique().tolist())

    support: dict[int, dict[int, int]] = {}
    for row in counts.itertuples(index=False):
        subject_id = int(row.subject_id)
        activity_id = int(row.activity_id)
        count = int(row.count)
        support.setdefault(subject_id, {})[activity_id] = count

    return support, activity_ids


def _choose_activity_ids(
    loader: Loader,
    indices: Sequence[int],
    needed_per_subject_activity: int,
    min_subjects: int,
) -> list[int]:
    if len(indices) == 0:
        raise ValueError("Cannot choose activity ids from empty indices.")

    support, activities = _activity_support_by_subject(loader, indices)
    if not activities:
        raise ValueError("No activity ids found in selected indices.")

    candidate = activities.copy()
    while candidate:
        eligible_subjects = [
            sid
            for sid, per_activity in support.items()
            if all(
                per_activity.get(aid, 0) >= needed_per_subject_activity
                for aid in candidate
            )
        ]
        if len(eligible_subjects) >= min_subjects:
            return candidate

        # Prune least-supported activity and try again.
        support_counts = {
            aid: sum(
                1
                for per_activity in support.values()
                if per_activity.get(aid, 0) >= needed_per_subject_activity
            )
            for aid in candidate
        }
        drop_aid = min(candidate, key=lambda aid: (support_counts[aid], aid))
        candidate = [aid for aid in candidate if aid != drop_aid]

    raise ValueError(
        "Could not find a non-empty activity set with sufficient per-subject support for episodic sampling."
    )


def _build_set_encoder(
    kind: str,
    backbone: TinierHAR,
    num_classes: int,
) -> torch.nn.Module:
    kind_norm = kind.strip().lower()
    if kind_norm == "prototypical":
        return PrototypicalSetEncoder(
            backbone=backbone,
            num_classes=num_classes,
            set_encoder_config=DEFAULT_CONFIG.set_encoder,
        )
    if kind_norm == "attention":
        return AttentionSetEncoder(
            backbone=backbone,
            num_classes=num_classes,
            set_encoder_config=DEFAULT_CONFIG.set_encoder,
        )
    raise ValueError(f"Unsupported SET_ENCODER_KIND={kind!r}.")


def _run_meta_eval(
    trainer: SetToLoRAMetaTrainer,
    episodes: int,
    use_vmap: bool,
) -> dict[str, float]:
    trainer.base_model.eval()
    trainer.set_encoder.eval()
    trainer.hypernet.eval()

    losses: list[float] = []
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(episodes):
            x_support, y_support, x_query, y_query, _ = trainer._sample_episode()
            x_support = x_support.to(trainer.device)
            y_support = y_support.to(trainer.device)
            x_query = x_query.to(trainer.device)
            y_query = y_query.to(trainer.device)

            c_subject = trainer.set_encoder(x_support, y_support)
            lora_weights = trainer.hypernet(c_subject)
            batched_params = trainer._build_batched_params(
                x_query.size(0), lora_weights
            )

            if use_vmap:
                try:
                    logits = trainer._forward_queries_vmap(batched_params, x_query)
                except RuntimeError as exc:
                    if "aten::gru.input" not in str(exc):
                        raise
                    logits = trainer._forward_queries_loop(batched_params, x_query)
            else:
                logits = trainer._forward_queries_loop(batched_params, x_query)

            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = y_query.reshape(-1)
            loss = F.cross_entropy(
                logits_flat, targets_flat, weight=trainer.class_weights
            )

            losses.append(float(loss.item()))
            all_preds.append(logits.argmax(dim=-1).reshape(-1).cpu())
            all_targets.append(targets_flat.cpu())

    preds_t = torch.cat(all_preds) if all_preds else torch.empty((0,), dtype=torch.long)
    targets_t = (
        torch.cat(all_targets) if all_targets else torch.empty((0,), dtype=torch.long)
    )
    macro_f1 = (
        f1_score(targets_t.numpy(), preds_t.numpy(), average="macro", zero_division=0)
        if preds_t.numel() > 0
        else 0.0
    )

    return {
        "loss": sum(losses) / max(1, len(losses)),
        "macro_f1": float(macro_f1),
    }


def main() -> None:
    cfg = get_dataset_cfg(DATASET_ID, datasets_dir=str(ROOT / "datasets"))
    train_cfg = DEFAULT_CONFIG.training
    variant_name = _resolve_variant_name(train_cfg)
    output_dir = OUTPUT_ROOT_DIR / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_pipeline = PreProcessingPipeline(cfg)
    _, session_df, window_df = pre_pipeline.run()

    splitter = MetaLOSOSplitter(cfg)
    splits = splitter.get_splits(session_df, window_df)

    summary_rows: list[dict[str, Any]] = []

    for split_idx, split in enumerate(splits):
        subject_id = split_idx
        split_dir = output_dir / f"subject_{subject_id}"
        split_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n=== LOSO META split {split_idx + 1}/{len(splits)} | subject={subject_id} ==="
        )

        if _is_completed_meta_split(split_dir):
            existing = _load_existing_meta_metrics(split_dir)
            if existing is not None:
                summary_rows.append(existing)
                print(
                    "Found existing completed meta artifacts, skipping training/evaluation "
                    f"for subject {subject_id}."
                )
                continue
            print(
                "Found completed meta artifacts but could not parse meta_metrics.json; "
                f"retraining subject {subject_id}."
            )

        post_pipeline = PostProcessingPipeline(
            cfg,
            pre_pipeline,
            window_df,
            split.train_indices,
        )
        samples = post_pipeline.run()

        loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)

        # Infer true held-out subject id from test windows if unique.
        inferred_subject = _infer_subject_id(
            loader, split.test_indices, fallback=subject_id
        )
        train_subject_ids = _subject_ids_for_indices(loader, split.train_indices)
        val_subject_ids = _subject_ids_for_indices(loader, split.val_indices)
        test_subject_ids = _subject_ids_for_indices(loader, split.test_indices)
        print(
            "Meta split subjects "
            f"(train/val/test)={len(train_subject_ids)}/{len(val_subject_ids)}/{len(test_subject_ids)} "
            f"| train={train_subject_ids} val={val_subject_ids} test={test_subject_ids}"
        )

        num_channels = cfg.num_of_channels
        num_classes = cfg.num_of_activities
        window_size = _infer_window_size(loader, split.train_indices)

        base_model = TinierHAR(
            num_channels=num_channels,
            num_classes=num_classes,
            window_size=window_size,
            backbone_config=DEFAULT_CONFIG.backbone,
        )

        backbone_ckpt = BACKBONE_DIR / f"subject_{subject_id}" / "best_tinierhar.pt"
        if not backbone_ckpt.exists():
            raise FileNotFoundError(
                "Missing pretrained backbone checkpoint for split "
                f"{split_idx}: {backbone_ckpt}"
            )

        map_location = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        base_model.load_state_dict(torch.load(backbone_ckpt, map_location=map_location))
        for p in base_model.parameters():
            p.requires_grad = False
        base_model.eval()

        set_encoder = _build_set_encoder(SET_ENCODER_KIND, base_model, num_classes)
        hypernet = HyperNet(
            num_channels=num_channels,
            num_classes=num_classes,
            set_encoder_config=DEFAULT_CONFIG.set_encoder,
            backbone_config=DEFAULT_CONFIG.backbone,
            hypernet_config=DEFAULT_CONFIG.hypernet,
        )

        class_weights = _fetch_class_weights(loader, split, num_classes)
        if class_weights is not None:
            print(
                f"Using class weights from loader.get_class_weights: shape={tuple(class_weights.shape)}"
            )
        else:
            print("No class weights returned, using unweighted cross-entropy.")

        needed = SUPPORT_PER_CLASS + QUERY_PER_CLASS
        train_activity_ids = _choose_activity_ids(
            loader,
            split.train_indices,
            needed_per_subject_activity=needed,
            min_subjects=TRAIN_SUBJECTS_PER_EPISODE,
        )
        val_activity_ids = _choose_activity_ids(
            loader,
            split.val_indices,
            needed_per_subject_activity=needed,
            min_subjects=1,
        )
        test_activity_ids = _choose_activity_ids(
            loader,
            split.test_indices,
            needed_per_subject_activity=needed,
            min_subjects=1,
        )

        print(
            "Episode activity counts "
            f"(train/val/test)={len(train_activity_ids)}/{len(val_activity_ids)}/{len(test_activity_ids)}"
        )

        train_meta_cfg = MetaTrainerConfig(
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            batch_subjects=TRAIN_SUBJECTS_PER_EPISODE,
            support_per_class=SUPPORT_PER_CLASS,
            query_per_class=QUERY_PER_CLASS,
            use_vmap=USE_VMAP,
            seed=split_idx,
        )

        val_meta_cfg = MetaTrainerConfig(
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            batch_subjects=1,
            support_per_class=SUPPORT_PER_CLASS,
            query_per_class=QUERY_PER_CLASS,
            use_vmap=USE_VMAP,
            seed=10_000 + split_idx,
            device=train_meta_cfg.device,
        )

        test_meta_cfg = MetaTrainerConfig(
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            batch_subjects=1,
            support_per_class=SUPPORT_PER_CLASS,
            query_per_class=QUERY_PER_CLASS,
            use_vmap=USE_VMAP,
            seed=20_000 + split_idx,
            device=train_meta_cfg.device,
        )

        optimizer = torch.optim.Adam(
            list(set_encoder.parameters()) + list(hypernet.parameters()),
            lr=train_meta_cfg.learning_rate,
            weight_decay=train_meta_cfg.weight_decay,
        )

        train_trainer = SetToLoRAMetaTrainer(
            base_model=base_model,
            set_encoder=set_encoder,
            hypernet=hypernet,
            loader=loader,
            num_classes=num_classes,
            config=train_meta_cfg,
            optimizer=optimizer,
            class_weights=class_weights,
            indices=split.train_indices,
            activity_ids=train_activity_ids,
        )
        val_trainer = SetToLoRAMetaTrainer(
            base_model=base_model,
            set_encoder=set_encoder,
            hypernet=hypernet,
            loader=loader,
            num_classes=num_classes,
            config=val_meta_cfg,
            optimizer=optimizer,
            class_weights=class_weights,
            indices=split.val_indices,
            activity_ids=val_activity_ids,
        )
        test_trainer = SetToLoRAMetaTrainer(
            base_model=base_model,
            set_encoder=set_encoder,
            hypernet=hypernet,
            loader=loader,
            num_classes=num_classes,
            config=test_meta_cfg,
            optimizer=optimizer,
            class_weights=class_weights,
            indices=split.test_indices,
            activity_ids=test_activity_ids,
        )

        best_val_loss = float("inf")
        best_epoch = -1
        best_train_loss = float("inf")
        best_train_f1 = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        best_ckpt_path = split_dir / "best_meta_modules.pt"

        for epoch in range(1, train_cfg.num_epochs + 1):
            train_losses: list[float] = []
            train_f1s: list[float] = []

            progress = tqdm(
                range(TRAIN_EPISODES_PER_EPOCH),
                desc=f"MetaTrain {epoch}/{train_cfg.num_epochs}",
                leave=False,
            )
            for _ in progress:
                step_metrics = train_trainer.train_step(use_vmap=USE_VMAP)
                train_losses.append(float(step_metrics["loss"]))
                train_f1s.append(float(step_metrics["macro_f1"]))
                progress.set_postfix(
                    loss=f"{sum(train_losses) / max(1, len(train_losses)):.4f}",
                    f1=f"{sum(train_f1s) / max(1, len(train_f1s)):.4f}",
                )

            train_loss = sum(train_losses) / max(1, len(train_losses))
            train_macro_f1 = sum(train_f1s) / max(1, len(train_f1s))
            val_metrics = _run_meta_eval(val_trainer, EVAL_EPISODES, use_vmap=USE_VMAP)
            val_loss = float(val_metrics["loss"])
            val_macro_f1 = float(val_metrics["macro_f1"])

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                best_train_loss = train_loss
                best_train_f1 = train_macro_f1
                best_val_f1 = val_macro_f1
                patience_counter = 0
                torch.save(
                    {
                        "set_encoder": set_encoder.state_dict(),
                        "hypernet": hypernet.state_dict(),
                        "meta_config": asdict(train_meta_cfg),
                        "train_activity_ids": train_activity_ids,
                        "val_activity_ids": val_activity_ids,
                        "test_activity_ids": test_activity_ids,
                        "backbone_checkpoint": str(backbone_ckpt),
                    },
                    best_ckpt_path,
                )
            else:
                patience_counter += 1

            print(
                f"[Meta Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f} train_macro_f1={train_macro_f1:.4f} "
                f"val_loss={val_loss:.4f} val_macro_f1={val_macro_f1:.4f} "
                f"best_val_loss={best_val_loss:.4f} patience={patience_counter}/{train_cfg.patience}"
            )

            if patience_counter >= train_cfg.patience:
                print(
                    f"Meta early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}."
                )
                break

        if not best_ckpt_path.exists():
            raise RuntimeError("No best meta checkpoint was saved.")

        ckpt = torch.load(best_ckpt_path, map_location=train_trainer.device)
        set_encoder.load_state_dict(ckpt["set_encoder"])
        hypernet.load_state_dict(ckpt["hypernet"])

        test_metrics = _run_meta_eval(test_trainer, EVAL_EPISODES, use_vmap=USE_VMAP)

        result = SplitMetaResult(
            split_index=split_idx,
            subject_id=inferred_subject,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            train_loss_at_best=best_train_loss,
            train_macro_f1_at_best=best_train_f1,
            val_macro_f1_at_best=best_val_f1,
            test_loss=float(test_metrics["loss"]),
            test_macro_f1=float(test_metrics["macro_f1"]),
            checkpoint_path=str(best_ckpt_path),
        )

        result_dict = asdict(result)
        with (split_dir / "meta_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)

        with (split_dir / "meta_history.json").open("w", encoding="utf-8") as f:
            json.dump(train_trainer.state.history, f, indent=2)

        summary_rows.append(result_dict)

        print(
            f"Subject {inferred_subject}: test_loss={result.test_loss:.4f}, "
            f"test_macro_f1={result.test_macro_f1:.4f}"
        )

    mean_macro_f1 = sum(float(r["test_macro_f1"]) for r in summary_rows) / max(
        1, len(summary_rows)
    )
    mean_loss = sum(float(r["test_loss"]) for r in summary_rows) / max(
        1, len(summary_rows)
    )
    summary = {
        "num_splits": len(summary_rows),
        "mean_test_macro_f1": mean_macro_f1,
        "mean_test_loss": mean_loss,
        "splits": summary_rows,
        "backbone_dir": str(BACKBONE_DIR),
        "meta_output_root_dir": str(OUTPUT_ROOT_DIR),
        "meta_output_dir": str(output_dir),
        "variant_name": variant_name,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LOSO meta-training finished ===")
    print(f"Mean meta-test macro F1: {mean_macro_f1:.4f}")
    print(f"Mean meta-test loss: {mean_loss:.4f}")
    print(f"Variant: {variant_name}")
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()

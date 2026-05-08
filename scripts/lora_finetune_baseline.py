from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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


META_TRAINER_MODULE = _load_module_from_path(
    "hyper_har_meta_trainer", SRC / "hyper_har" / "training" / "meta-trainer.py"
)
MetaTrainerConfig = META_TRAINER_MODULE.MetaTrainerConfig
SetToLoRAMetaTrainer = META_TRAINER_MODULE.SetToLoRAMetaTrainer


DATASET_ID = WHARDatasetID.WEAR
BACKBONE_DIR = ROOT / "artifacts" / "loso_cv"
OUTPUT_DIR = ROOT / "artifacts" / "lora_finetune_loso_cv"

SUPPORT_PER_CLASS_CHOICES = (2, 4, 8, 12, 16, 20)
QUERY_PER_CLASS = 16
EVAL_EPISODES = 128
FINETUNE_STEPS = 30
FINETUNE_LR = 1e-2
FINETUNE_WEIGHT_DECAY = 0.0
LORA_RANK = 4
LORA_ALPHA = 1.0
ENABLE_CONV1_ADAPTER = False
ENABLE_CONV_LAST_ADAPTER = False
GRAD_CLIP_NORM = 1.0


@dataclass
class FineTuneSubjectResult:
    split_index: int
    subject_id: int
    episodes: int
    support_per_class_choices: list[int]
    query_per_class: int
    finetune_steps: int
    finetune_lr: float
    base_loss: float
    base_macro_f1: float
    finetuned_loss: float
    finetuned_macro_f1: float
    macro_f1_improvement: float


class _NoOpSetEncoder(torch.nn.Module):
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("_NoOpSetEncoder is only used to satisfy trainer wiring.")


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
    return int(subjects[0]) if len(subjects) == 1 else fallback


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
        raise ValueError("Missing subject/activity metadata.")

    counts = (
        merged.groupby(["subject_id", "activity_id"])
        .size()
        .rename("count")
        .reset_index()
    )
    activity_ids = sorted(int(x) for x in counts["activity_id"].unique().tolist())
    support: dict[int, dict[int, int]] = {}
    for row in counts.itertuples(index=False):
        support.setdefault(int(row.subject_id), {})[int(row.activity_id)] = int(
            row.count
        )
    return support, activity_ids


def _choose_activity_ids(
    loader: Loader,
    indices: Sequence[int],
    needed_per_subject_activity: int,
    min_subjects: int,
) -> list[int]:
    support, activities = _activity_support_by_subject(loader, indices)
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

    raise ValueError("No activity set has enough support for LoRA fine-tuning.")


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
            weights[class_id] = 0.0 if raw_w < 0.0 else raw_w
        return weights

    weights = torch.as_tensor(weights_obj, dtype=torch.float32).view(-1)
    if weights.numel() != num_classes:
        raise ValueError(
            f"Class weights length mismatch: expected {num_classes}, got {weights.numel()}."
        )
    return torch.where(weights < 0.0, torch.zeros_like(weights), weights)


def _init_lora_params(
    hypernet: HyperNet, batch_size: int, device: torch.device
) -> dict[str, tuple[torch.nn.Parameter, torch.nn.Parameter]]:
    adapters: dict[str, tuple[torch.nn.Parameter, torch.nn.Parameter]] = {}
    for name, (out_dim, in_dim) in hypernet.target_shapes.items():
        if "conv" in name:
            a_shape = (batch_size, hypernet.lora_rank, in_dim, 1, 1)
            b_shape = (batch_size, out_dim, hypernet.lora_rank, 1, 1)
        else:
            a_shape = (batch_size, hypernet.lora_rank, in_dim)
            b_shape = (batch_size, out_dim, hypernet.lora_rank)
        A = torch.nn.Parameter(torch.empty(a_shape, device=device))
        B = torch.nn.Parameter(torch.zeros(b_shape, device=device))
        torch.nn.init.normal_(A, mean=0.0, std=1e-3)
        adapters[name] = (A, B)
    return adapters


def _adapter_parameters(
    adapters: Mapping[str, tuple[torch.nn.Parameter, torch.nn.Parameter]],
) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for A, B in adapters.values():
        params.extend([A, B])
    return params


def _forward_adapted(
    trainer: SetToLoRAMetaTrainer,
    adapters: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    x: torch.Tensor,
) -> torch.Tensor:
    batched_params = trainer._build_batched_params(x.size(0), adapters)
    return trainer._forward_queries_loop(batched_params, x)


def _base_logits(trainer: SetToLoRAMetaTrainer, x: torch.Tensor) -> torch.Tensor:
    return trainer.base_model(x.reshape(-1, *x.shape[2:])).reshape(
        x.size(0), x.size(1), -1
    )


def _run_finetune_eval(
    trainer: SetToLoRAMetaTrainer,
    hypernet: HyperNet,
    episodes: int,
    support_per_class_choices: Sequence[int],
    steps: int,
    lr: float,
    weight_decay: float,
) -> dict[str, float]:
    trainer.base_model.eval()
    hypernet.eval()

    base_losses: list[float] = []
    tuned_losses: list[float] = []
    all_targets: list[torch.Tensor] = []
    base_preds_all: list[torch.Tensor] = []
    tuned_preds_all: list[torch.Tensor] = []

    choices = [int(k) for k in support_per_class_choices if int(k) > 0]
    if not choices:
        raise ValueError("support_per_class_choices must contain positive integers.")

    for episode_idx in tqdm(range(episodes), desc="LoRA fine-tune episodes", leave=False):
        k = choices[episode_idx % len(choices)]
        x_support, y_support, x_query, y_query, _ = trainer._sample_episode(
            support_per_class=k
        )
        x_support = x_support.to(trainer.device)
        y_support = y_support.to(trainer.device)
        x_query = x_query.to(trainer.device)
        y_query = y_query.to(trainer.device)

        adapters = _init_lora_params(hypernet, x_support.size(0), trainer.device)
        optimizer = torch.optim.AdamW(
            _adapter_parameters(adapters), lr=lr, weight_decay=weight_decay
        )

        for _step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            support_logits = _forward_adapted(trainer, adapters, x_support)
            support_loss = F.cross_entropy(
                support_logits.reshape(-1, support_logits.size(-1)),
                y_support.reshape(-1),
                weight=trainer.class_weights,
            )
            support_loss.backward()
            if GRAD_CLIP_NORM > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    _adapter_parameters(adapters), GRAD_CLIP_NORM
                )
            optimizer.step()

        with torch.no_grad():
            targets_flat = y_query.reshape(-1)
            base_logits = _base_logits(trainer, x_query)
            tuned_logits = _forward_adapted(trainer, adapters, x_query)

            base_loss = F.cross_entropy(
                base_logits.reshape(-1, base_logits.size(-1)),
                targets_flat,
                weight=trainer.class_weights,
            )
            tuned_loss = F.cross_entropy(
                tuned_logits.reshape(-1, tuned_logits.size(-1)),
                targets_flat,
                weight=trainer.class_weights,
            )
            base_losses.append(float(base_loss.item()))
            tuned_losses.append(float(tuned_loss.item()))
            all_targets.append(targets_flat.cpu())
            base_preds_all.append(base_logits.argmax(dim=-1).reshape(-1).cpu())
            tuned_preds_all.append(tuned_logits.argmax(dim=-1).reshape(-1).cpu())

    targets_t = torch.cat(all_targets)
    base_preds_t = torch.cat(base_preds_all)
    tuned_preds_t = torch.cat(tuned_preds_all)
    base_macro_f1 = f1_score(
        targets_t.numpy(), base_preds_t.numpy(), average="macro", zero_division=0
    )
    tuned_macro_f1 = f1_score(
        targets_t.numpy(), tuned_preds_t.numpy(), average="macro", zero_division=0
    )

    return {
        "base_loss": sum(base_losses) / max(1, len(base_losses)),
        "base_macro_f1": float(base_macro_f1),
        "finetuned_loss": sum(tuned_losses) / max(1, len(tuned_losses)),
        "finetuned_macro_f1": float(tuned_macro_f1),
        "macro_f1_improvement": float(tuned_macro_f1 - base_macro_f1),
    }


def main() -> None:
    cfg = get_dataset_cfg(DATASET_ID, datasets_dir=str(ROOT / "datasets"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pre_pipeline = PreProcessingPipeline(cfg)
    _, session_df, window_df = pre_pipeline.run()
    splitter = MetaLOSOSplitter(cfg)
    folds = splitter.get_folds(session_df, window_df)
    summary_rows: list[dict[str, Any]] = []

    for split_idx, fold in enumerate(folds):
        split = fold.meta_split
        subject_id = int(fold.test_subject_id)
        split_dir = OUTPUT_DIR / f"subject_{subject_id}"
        split_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = split_dir / "lora_finetune_metrics.json"

        print(
            f"\n=== LoRA fine-tune split {split_idx + 1}/{len(folds)} | subject={subject_id} ==="
        )
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            summary_rows.append(existing)
            print(f"Found existing metrics, skipping subject {subject_id}.")
            continue

        post_pipeline = PostProcessingPipeline(
            cfg, pre_pipeline, window_df, split.train_indices
        )
        samples = post_pipeline.run()
        loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)

        inferred_subject = _infer_subject_id(
            loader, split.test_indices, fallback=subject_id
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
            raise FileNotFoundError(f"Missing backbone checkpoint: {backbone_ckpt}")

        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        base_model.load_state_dict(torch.load(backbone_ckpt, map_location=device))
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad = False

        hypernet = HyperNet(
            num_channels=num_channels,
            num_classes=num_classes,
            set_encoder_output_dim=(num_classes + 1)
            * DEFAULT_CONFIG.set_encoder.hidden_dim,
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            enable_conv1_adapter=ENABLE_CONV1_ADAPTER,
            enable_conv_last_adapter=ENABLE_CONV_LAST_ADAPTER,
            set_encoder_config=DEFAULT_CONFIG.set_encoder,
            backbone_config=DEFAULT_CONFIG.backbone,
            hypernet_config=DEFAULT_CONFIG.hypernet,
        )

        activity_ids = _choose_activity_ids(
            loader,
            split.test_indices,
            needed_per_subject_activity=max(SUPPORT_PER_CLASS_CHOICES) + QUERY_PER_CLASS,
            min_subjects=1,
        )
        eval_cfg = MetaTrainerConfig(
            batch_subjects=1,
            support_per_class=max(SUPPORT_PER_CLASS_CHOICES),
            support_per_class_choices=SUPPORT_PER_CLASS_CHOICES,
            query_per_class=QUERY_PER_CLASS,
            seed=30_000 + split_idx,
            device=device,
        )
        class_weights = _fetch_class_weights(loader, split, num_classes)

        trainer = SetToLoRAMetaTrainer(
            base_model=base_model,
            set_encoder=_NoOpSetEncoder(),
            hypernet=hypernet,
            loader=loader,
            num_classes=num_classes,
            config=eval_cfg,
            class_weights=class_weights,
            indices=split.test_indices,
            activity_ids=activity_ids,
        )

        metrics = _run_finetune_eval(
            trainer=trainer,
            hypernet=hypernet,
            episodes=EVAL_EPISODES,
            support_per_class_choices=SUPPORT_PER_CLASS_CHOICES,
            steps=FINETUNE_STEPS,
            lr=FINETUNE_LR,
            weight_decay=FINETUNE_WEIGHT_DECAY,
        )
        result = FineTuneSubjectResult(
            split_index=split_idx,
            subject_id=inferred_subject,
            episodes=EVAL_EPISODES,
            support_per_class_choices=list(SUPPORT_PER_CLASS_CHOICES),
            query_per_class=QUERY_PER_CLASS,
            finetune_steps=FINETUNE_STEPS,
            finetune_lr=FINETUNE_LR,
            base_loss=float(metrics["base_loss"]),
            base_macro_f1=float(metrics["base_macro_f1"]),
            finetuned_loss=float(metrics["finetuned_loss"]),
            finetuned_macro_f1=float(metrics["finetuned_macro_f1"]),
            macro_f1_improvement=float(metrics["macro_f1_improvement"]),
        )
        result_dict = asdict(result)
        result_dict["test_subject_id"] = int(fold.test_subject_id)
        result_dict["activity_ids"] = activity_ids
        result_dict["lora_rank"] = LORA_RANK
        result_dict["lora_alpha"] = LORA_ALPHA
        result_dict["adapter_modules"] = hypernet.module_names
        result_dict["backbone_checkpoint"] = str(backbone_ckpt)

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)
        summary_rows.append(result_dict)

        print(
            f"Subject {inferred_subject}: base_f1={result.base_macro_f1:.4f}, "
            f"finetuned_f1={result.finetuned_macro_f1:.4f}, "
            f"improvement={result.macro_f1_improvement:+.4f}"
        )

    mean_base_macro_f1 = sum(float(r["base_macro_f1"]) for r in summary_rows) / max(
        1, len(summary_rows)
    )
    mean_finetuned_macro_f1 = sum(
        float(r["finetuned_macro_f1"]) for r in summary_rows
    ) / max(1, len(summary_rows))
    mean_improvement = sum(
        float(r["macro_f1_improvement"]) for r in summary_rows
    ) / max(1, len(summary_rows))

    summary = {
        "num_splits": len(summary_rows),
        "mean_base_macro_f1": mean_base_macro_f1,
        "mean_finetuned_macro_f1": mean_finetuned_macro_f1,
        "mean_macro_f1_improvement": mean_improvement,
        "support_per_class_choices": list(SUPPORT_PER_CLASS_CHOICES),
        "query_per_class": QUERY_PER_CLASS,
        "episodes": EVAL_EPISODES,
        "finetune_steps": FINETUNE_STEPS,
        "finetune_lr": FINETUNE_LR,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "enable_conv1_adapter": ENABLE_CONV1_ADAPTER,
        "enable_conv_last_adapter": ENABLE_CONV_LAST_ADAPTER,
        "splits": summary_rows,
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LoRA fine-tuning baseline finished ===")
    print(f"Mean episodic base macro F1: {mean_base_macro_f1:.4f}")
    print(f"Mean fine-tuned macro F1: {mean_finetuned_macro_f1:.4f}")
    print(f"Mean improvement: {mean_improvement:+.4f}")
    print(f"Saved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

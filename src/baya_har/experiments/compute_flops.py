import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from baya_har.config import DEFAULT_CONFIG
from baya_har.experiments.common import repo_relative_path
from baya_har.models.tinierhar import TinierHAR

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[3] / "artifacts" / ".matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(__file__).resolve().parents[3] / "artifacts" / ".cache"),
)


ROOT = Path(__file__).resolve().parents[3]


DEFAULT_DATASETS = ("hapt", "harth", "wear", "hhar")
DEFAULT_PROTOTYPE_K_VALUES = (1, 16)


@dataclass(frozen=True)
class DatasetShape:
    dataset_id: str
    window_size: int
    num_channels: int
    num_classes: int
    source: str


@dataclass(frozen=True)
class CountConfig:
    batch_size: int = 1
    multiply_add_flops: int = 2
    include_bias_adds: bool = True
    include_batchnorm_affine: bool = True
    include_relu: bool = True
    include_pool_comparisons: bool = True
    include_residual_adds: bool = True
    include_gru_gate_pointwise: bool = True
    include_attention_softmax: bool = True
    include_attention_pooling: bool = True


@dataclass(frozen=True)
class PrototypeCountConfig:
    em_iterations: int = 1
    em_likelihood_variance_source: str = "fixed"
    em_responsibility_variance_source: str = "fixed"
    em_uniform_class_prior: bool = True
    center_train_support_query: bool = True
    supervised_singleton_support_variance: str = "prior"
    supervised_normalize_support_mean: bool = False
    supervised_project_to_sphere: bool = False
    include_diagnostics: bool = False


@dataclass
class CountRow:
    dataset_id: str
    input_shape: str
    window_size: int
    num_channels: int
    num_classes: int
    conv_time: int
    gru_input_dim: int
    embedding_dim: int
    macs: int
    flops: int
    conv_macs: int
    conv_flops: int
    gru_macs: int
    gru_flops: int
    attention_macs: int
    attention_flops: int
    backbone_params: int
    shape_source: str


@dataclass
class PrototypeCountRow:
    dataset_id: str
    k: int
    num_classes: int
    embedding_dim: int
    support_embeddings: int
    map_macs: int
    map_flops: int
    map_em_macs: int
    map_em_flops: int
    map_em_centering_flops: int
    em_iterations: int
    map_note: str
    map_em_note: str


def _add(target: dict[str, int], key: str, macs: int = 0, flops: int = 0) -> None:
    target[f"{key}_macs"] = target.get(f"{key}_macs", 0) + int(macs)
    target[f"{key}_flops"] = target.get(f"{key}_flops", 0) + int(flops)


def _conv2d(
    counts: dict[str, int],
    key: str,
    shape: tuple[int, int, int, int],
    out_channels: int,
    kernel: tuple[int, int],
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    stride: tuple[int, int] = (1, 1),
    groups: int = 1,
    bias: bool = True,
    count_config: CountConfig = CountConfig(),
) -> tuple[int, int, int, int]:
    batch, in_channels, height, width = shape
    kernel_h, kernel_w = kernel
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    stride_h, stride_w = stride
    out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    output_elements = batch * out_channels * out_h * out_w
    macs = output_elements * (in_channels // groups) * kernel_h * kernel_w
    flops = count_config.multiply_add_flops * macs
    if bias and count_config.include_bias_adds:
        flops += output_elements
    _add(counts, key, macs, flops)
    return batch, out_channels, out_h, out_w


def _batchnorm2d(
    counts: dict[str, int],
    key: str,
    shape: tuple[int, int, int, int],
    count_config: CountConfig,
) -> None:
    if count_config.include_batchnorm_affine:
        _add(counts, key, flops=2 * _numel(shape))


def _relu(
    counts: dict[str, int],
    key: str,
    shape: tuple[int, ...],
    count_config: CountConfig,
) -> None:
    if count_config.include_relu:
        _add(counts, key, flops=_numel(shape))


def _maxpool2d(
    counts: dict[str, int],
    key: str,
    shape: tuple[int, int, int, int],
    kernel: tuple[int, int],
    count_config: CountConfig,
) -> tuple[int, int, int, int]:
    batch, channels, height, width = shape
    kernel_h, kernel_w = kernel
    out_h = height // kernel_h
    out_w = width // kernel_w
    out_shape = (batch, channels, out_h, out_w)
    if count_config.include_pool_comparisons:
        _add(counts, key, flops=_numel(out_shape) * (kernel_h * kernel_w - 1))
    return out_shape


def _linear(
    counts: dict[str, int],
    key: str,
    rows: int,
    in_features: int,
    out_features: int,
    bias: bool,
    count_config: CountConfig,
) -> None:
    output_elements = rows * out_features
    macs = output_elements * in_features
    flops = count_config.multiply_add_flops * macs
    if bias and count_config.include_bias_adds:
        flops += output_elements
    _add(counts, key, macs, flops)


def _numel(shape: Iterable[int]) -> int:
    out = 1
    for value in shape:
        out *= int(value)
    return int(out)


def _conv_block(
    counts: dict[str, int],
    shape: tuple[int, int, int, int],
    in_channels: int,
    out_channels: int,
    use_maxpool: bool,
    shortcut: bool,
    count_config: CountConfig,
) -> tuple[int, int, int, int]:
    padding = (5 - 1 + 1) // 2

    main = _conv2d(
        counts,
        "conv",
        shape,
        out_channels=in_channels,
        kernel=(5, 1),
        padding=(padding, 0),
        groups=in_channels,
        bias=True,
        count_config=count_config,
    )
    main = _conv2d(
        counts,
        "conv",
        main,
        out_channels=out_channels,
        kernel=(1, 1),
        bias=True,
        count_config=count_config,
    )
    _batchnorm2d(counts, "conv", main, count_config)
    _relu(counts, "conv", main, count_config)
    if use_maxpool:
        main = _maxpool2d(counts, "conv", main, (2, 1), count_config)

    if not shortcut:
        return main

    skip = shape
    if in_channels != out_channels:
        skip = _conv2d(
            counts,
            "conv",
            skip,
            out_channels=out_channels,
            kernel=(1, 1),
            bias=True,
            count_config=count_config,
        )
        _batchnorm2d(counts, "conv", skip, count_config)
    if use_maxpool:
        skip = _maxpool2d(counts, "conv", skip, (2, 1), count_config)
    if main != skip:
        raise RuntimeError(f"Residual shapes differ: main={main}, skip={skip}")
    if count_config.include_residual_adds:
        _add(counts, "conv", flops=_numel(main))
    return main


def _gru(
    counts: dict[str, int],
    seq_len: int,
    input_dim: int,
    hidden_dim: int,
    bidirectional: bool,
    count_config: CountConfig,
) -> int:
    directions = 2 if bidirectional else 1
    per_step_macs = 3 * hidden_dim * input_dim + 3 * hidden_dim * hidden_dim
    macs = directions * seq_len * per_step_macs
    flops = count_config.multiply_add_flops * macs
    if count_config.include_bias_adds:
        flops += directions * seq_len * 6 * hidden_dim
    if count_config.include_gru_gate_pointwise:
        gate_adds = 3 * hidden_dim
        activations = 3 * hidden_dim
        candidate_reset_mul = hidden_dim
        hidden_update = 4 * hidden_dim
        flops += (
            directions
            * seq_len
            * (gate_adds + activations + candidate_reset_mul + hidden_update)
        )
    _add(counts, "gru", macs, flops)
    return directions * hidden_dim


def _attention(
    counts: dict[str, int],
    seq_len: int,
    feature_dim: int,
    count_config: CountConfig,
) -> None:
    _linear(
        counts,
        "attention",
        rows=seq_len,
        in_features=feature_dim,
        out_features=1,
        bias=True,
        count_config=count_config,
    )
    if count_config.include_attention_softmax:
        # exp for each score, (seq_len - 1) additions for the denominator, and
        # one division per score. This intentionally keeps transcendental ops as
        # one counted operation so the convention stays simple and auditable.
        _add(counts, "attention", flops=seq_len + max(0, seq_len - 1) + seq_len)
    if count_config.include_attention_pooling:
        weighted_mults = seq_len * feature_dim
        sum_adds = max(0, seq_len - 1) * feature_dim
        _add(counts, "attention", flops=weighted_mults + sum_adds)


def count_tinierhar_backbone(
    shape: DatasetShape,
    count_config: CountConfig,
) -> CountRow:
    cfg = DEFAULT_CONFIG.backbone
    counts: dict[str, int] = {}
    tensor_shape = (
        int(count_config.batch_size),
        1,
        int(shape.window_size),
        int(shape.num_channels),
    )
    current = tensor_shape
    filters = int(cfg.nb_filters)
    current = _conv_block(
        counts,
        current,
        in_channels=1,
        out_channels=filters,
        use_maxpool=True,
        shortcut=True,
        count_config=count_config,
    )
    current = _conv_block(
        counts,
        current,
        in_channels=filters,
        out_channels=2 * filters,
        use_maxpool=True,
        shortcut=True,
        count_config=count_config,
    )
    for _ in range(int(cfg.nb_conv_blocks)):
        current = _conv_block(
            counts,
            current,
            in_channels=2 * filters,
            out_channels=2 * filters,
            use_maxpool=False,
            shortcut=True,
            count_config=count_config,
        )

    _batch, conv_channels, conv_time, conv_sensor_channels = current
    gru_input_dim = int(conv_channels * conv_sensor_channels)
    embedding_dim = _gru(
        counts,
        seq_len=int(conv_time),
        input_dim=gru_input_dim,
        hidden_dim=int(cfg.nb_units_gru),
        bidirectional=True,
        count_config=count_config,
    )
    _attention(
        counts,
        seq_len=int(conv_time),
        feature_dim=embedding_dim,
        count_config=count_config,
    )

    model = TinierHAR(
        num_channels=int(shape.num_channels),
        num_classes=int(shape.num_classes),
        window_size=int(shape.window_size),
        backbone_config=DEFAULT_CONFIG.backbone,
    )
    with torch.no_grad():
        dummy = torch.zeros(tensor_shape)
        observed = tuple(int(x) for x in model.conv_blocks(dummy).shape)
    if observed != current:
        raise RuntimeError(
            "Analytical conv shape does not match TinierHAR.conv_blocks: "
            f"analytical={current}, observed={observed}"
        )
    backbone_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if not name.startswith("classifier.")
    )

    macs = sum(value for key, value in counts.items() if key.endswith("_macs"))
    flops = sum(value for key, value in counts.items() if key.endswith("_flops"))
    return CountRow(
        dataset_id=shape.dataset_id,
        input_shape=f"({count_config.batch_size}, 1, {shape.window_size}, {shape.num_channels})",
        window_size=int(shape.window_size),
        num_channels=int(shape.num_channels),
        num_classes=int(shape.num_classes),
        conv_time=int(conv_time),
        gru_input_dim=int(gru_input_dim),
        embedding_dim=int(embedding_dim),
        macs=int(macs),
        flops=int(flops),
        conv_macs=int(counts.get("conv_macs", 0)),
        conv_flops=int(counts.get("conv_flops", 0)),
        gru_macs=int(counts.get("gru_macs", 0)),
        gru_flops=int(counts.get("gru_flops", 0)),
        attention_macs=int(counts.get("attention_macs", 0)),
        attention_flops=int(counts.get("attention_flops", 0)),
        backbone_params=int(backbone_params),
        shape_source=shape.source,
    )


def _l2_normalize_flops(rows: int, dim: int) -> int:
    if rows <= 0:
        return 0
    # square, sum, sqrt, divide for each row vector.
    return rows * (dim + max(0, dim - 1) + 1 + dim)


def _supervised_map_prototype_count(
    num_classes: int,
    embedding_dim: int,
    k: int,
    config: PrototypeCountConfig,
) -> tuple[int, int]:
    if k < 1:
        raise ValueError(f"MAP prototype estimation requires k >= 1, got {k}.")
    if config.supervised_singleton_support_variance not in {"prior", "floor"}:
        raise ValueError(
            "supervised_singleton_support_variance must be 'prior' or 'floor'."
        )

    c = int(num_classes)
    d = int(embedding_dim)
    k = int(k)
    flops_per_class = 0

    # support_mean = cls_support.mean(dim=0)
    flops_per_class += max(0, k - 1) * d + d

    if config.supervised_normalize_support_mean:
        flops_per_class += _l2_normalize_flops(1, d)

    # prior_var = prior_variances[activity_id].clamp_min(...)
    flops_per_class += d

    if k > 1:
        # support_var = cls_support.var(...).clamp_min(...)
        flops_per_class += k * d  # subtract support mean
        flops_per_class += k * d  # square centered values
        flops_per_class += max(0, k - 1) * d  # variance sum
        flops_per_class += d  # divide by k - 1
        flops_per_class += d  # clamp_min
    elif config.supervised_singleton_support_variance == "floor":
        # torch.full_like has no arithmetic FLOPs.
        pass

    # prior/support precision, precision sum, and posterior mean.
    flops_per_class += d  # prior_precision = 1 / prior_var
    flops_per_class += d  # support_precision = n / support_var
    flops_per_class += d  # precision_sum = prior_precision + support_precision
    flops_per_class += 2 * d  # weighted prior/support means
    flops_per_class += d  # add weighted means
    flops_per_class += d  # divide by precision_sum

    if config.supervised_project_to_sphere:
        # The script computes the norm diagnostic before projection; this is
        # intentionally omitted unless include_diagnostics=True.
        flops_per_class += _l2_normalize_flops(1, d)

    if config.include_diagnostics:
        flops_per_class += 2 * d  # prior/support weights
        flops_per_class += 3 * d + max(0, d - 1)  # three vector means
        flops_per_class += d + max(0, d - 1) + 1  # prototype norm

    return 0, c * flops_per_class


def _euclidean_logits_count(
    num_support: int,
    num_classes: int,
    embedding_dim: int,
) -> int:
    pairs = int(num_support) * int(num_classes)
    d = int(embedding_dim)
    # prototype_logits(..., "euclidean") calls torch.cdist and then squares the
    # returned distance before dividing by temperature. Count semantic ops:
    # subtract, square, sum, sqrt, square, divide, negate.
    return pairs * (d + d + max(0, d - 1) + 1 + 1 + 1 + 1)


def _softmax_flops(rows: int, cols: int) -> int:
    return int(rows) * (int(cols) + max(0, int(cols) - 1) + int(cols))


def _map_em_prototype_count(
    num_classes: int,
    embedding_dim: int,
    k: int,
    config: PrototypeCountConfig,
    count_config: CountConfig,
) -> tuple[int, int, int]:
    if k < 1:
        raise ValueError(f"MAP-EM prototype estimation requires k >= 1, got {k}.")
    if config.em_iterations < 1:
        raise ValueError("em_iterations must be >= 1.")
    if config.em_likelihood_variance_source not in {"fixed", "responsibility"}:
        raise ValueError(
            "em_likelihood_variance_source must be 'fixed' or 'responsibility'."
        )
    if config.em_responsibility_variance_source not in {"fixed", "support"}:
        raise ValueError(
            "em_responsibility_variance_source must be 'fixed' or 'support'."
        )

    c = int(num_classes)
    d = int(embedding_dim)
    n = c * int(k)
    macs = 0
    flops = 0
    centering_flops = 0

    # prior_var = prior_variances[activity_tensor].clamp_min(...)
    flops += c * d

    if config.center_train_support_query:
        # support_center = support_emb.mean(dim=0)
        centering_flops += max(0, n - 1) * d + d
        # support_emb = support_emb - support_center
        centering_flops += n * d
        flops += centering_flops

    for _ in range(int(config.em_iterations)):
        if config.em_responsibility_variance_source == "support":
            # Gaussian diagonal logits:
            # centered, square, divide by variance, log variance, add, sum,
            # multiply by -0.5.
            flops += n * c * d  # subtract prototypes
            flops += n * c * d  # square
            flops += n * c * d  # divide by responsibility variance
            flops += c * d  # log responsibility variance
            flops += n * c * d  # add log variance
            flops += n * c * max(0, d - 1)  # sum over d
            flops += n * c  # multiply by -0.5
        else:
            flops += _euclidean_logits_count(n, c, d)

        if not config.em_uniform_class_prior:
            flops += n * c

        flops += _softmax_flops(n, c)
        flops += c * max(0, n - 1)  # soft_counts = responsibilities.sum(dim=0)
        flops += c  # safe_counts = soft_counts.clamp_min(...)

        # soft_means = responsibilities.T @ support_emb
        soft_mean_macs = c * d * n
        macs += soft_mean_macs
        flops += count_config.multiply_add_flops * soft_mean_macs
        flops += c * d  # soft_means / safe_counts

        if (
            config.em_likelihood_variance_source == "responsibility"
            or config.em_responsibility_variance_source == "support"
        ):
            flops += n * c * d  # centered
            flops += n * c * d  # square
            flops += n * c * d  # multiply by responsibilities
            flops += c * d * max(0, n - 1)  # sum over support embeddings
            flops += c * d  # divide by safe_counts
            flops += c * d  # clamp_min

        # prior_precision, update_var, support_precision, precision_sum,
        # posterior prototypes.
        flops += c * d  # prior_precision = 1 / prior_var
        flops += c * d  # support_precision = safe_counts / update_var
        flops += c * d  # precision_sum
        flops += 2 * c * d  # weighted prior/support means
        flops += c * d  # add weighted means
        flops += c * d  # divide by precision_sum

    if config.include_diagnostics:
        flops += 2 * c * d  # prior/support weight divisions
        flops += n * c  # clamp responsibilities for entropy
        flops += n * c  # log responsibilities
        flops += n * c  # multiply p * log(p)
        flops += n * max(0, c - 1)  # entropy sum
        flops += n  # entropy negation
        flops += n * max(0, c - 1)  # confidence max
        flops += n + c + c * d  # rough means/min/max diagnostics
        flops += c * d + c * max(0, d - 1) + c  # prototype norms
        flops += c * d + c * d + c * max(0, d - 1) + c  # shift norms

    return int(macs), int(flops), int(centering_flops)


def count_prototype_estimation(
    shape: DatasetShape,
    embedding_dim: int,
    k_values: Iterable[int],
    count_config: CountConfig,
    prototype_config: PrototypeCountConfig,
) -> list[PrototypeCountRow]:
    rows: list[PrototypeCountRow] = []
    for k in k_values:
        k = int(k)
        if k < 1:
            continue
        map_macs, map_flops = _supervised_map_prototype_count(
            num_classes=int(shape.num_classes),
            embedding_dim=int(embedding_dim),
            k=k,
            config=prototype_config,
        )
        map_em_macs, map_em_flops, map_em_centering_flops = _map_em_prototype_count(
            num_classes=int(shape.num_classes),
            embedding_dim=int(embedding_dim),
            k=k,
            config=prototype_config,
            count_config=count_config,
        )
        rows.append(
            PrototypeCountRow(
                dataset_id=shape.dataset_id,
                k=k,
                num_classes=int(shape.num_classes),
                embedding_dim=int(embedding_dim),
                support_embeddings=int(shape.num_classes) * k,
                map_macs=map_macs,
                map_flops=map_flops,
                map_em_macs=map_em_macs,
                map_em_flops=map_em_flops,
                map_em_centering_flops=map_em_centering_flops,
                em_iterations=int(prototype_config.em_iterations),
                map_note="supervised Bayesian MAP update from script 07",
                map_em_note=(
                    "unlabeled MAP-EM update from script 09; fixed likelihood "
                    "and fixed responsibility variance defaults; includes "
                    "support centering for center_train_support_query=True"
                ),
            )
        )
    return rows


def _load_shape_from_checkpoint(
    dataset_id: str,
    artifacts_root: Path,
    ce_stage_name: str,
) -> DatasetShape | None:
    stage_dir = artifacts_root / dataset_id / ce_stage_name
    paths = sorted(stage_dir.glob("*/best_model_with_meta.pt"))
    if not paths:
        return None

    shapes: set[tuple[int, int, int]] = set()
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        meta = dict(checkpoint.get("model_meta", {}))
        required = ("window_size", "num_channels", "num_classes")
        if not all(key in meta for key in required):
            continue
        shapes.add(
            (
                int(meta["window_size"]),
                int(meta["num_channels"]),
                int(meta["num_classes"]),
            )
        )
    if not shapes:
        return None
    if len(shapes) != 1:
        raise ValueError(
            f"Found inconsistent checkpoint shapes for {dataset_id}: {sorted(shapes)}"
        )
    window_size, num_channels, num_classes = next(iter(shapes))
    return DatasetShape(
        dataset_id=dataset_id,
        window_size=window_size,
        num_channels=num_channels,
        num_classes=num_classes,
        source=f"{repo_relative_path(stage_dir)}/*/best_model_with_meta.pt",
    )


def _load_shape_from_preprocessing(
    dataset_id: str,
    datasets_dir: Path,
) -> DatasetShape:
    from whar_datasets import PreProcessingPipeline, WHARDatasetID

    from .common import (
        DEFAULT_SEED,
        DEFAULT_SELECTED_ACTIVITIES,
        DEFAULT_SPLIT_STRATEGY,
        DEFAULT_TEST_SUBJECTS,
        DEFAULT_VAL_PERCENTAGE,
        DEFAULT_VAL_SUBJECTS,
        DEFAULT_WINDOW_OVERLAP,
        SharedConfig,
        build_loader,
        build_or_load_loso_folds,
        prepare_cfg,
        reconcile_activity_config,
        resolve_output_root,
        sample_window_array,
        split_indices_for_fold,
    )

    cfg = prepare_cfg(
        dataset_id=WHARDatasetID(dataset_id),
        datasets_dir=datasets_dir,
        selected_activities=DEFAULT_SELECTED_ACTIVITIES,
        window_overlap=DEFAULT_WINDOW_OVERLAP,
    )
    pre = PreProcessingPipeline(cfg)
    _raw_df, session_df, window_df = pre.run()
    reconcile_activity_config(cfg, session_df)
    shared_cfg = SharedConfig(
        dataset_id=dataset_id,
        datasets_dir=str(datasets_dir),
        selected_activities=DEFAULT_SELECTED_ACTIVITIES,
        window_overlap=DEFAULT_WINDOW_OVERLAP,
        val_subjects=DEFAULT_VAL_SUBJECTS,
        test_subjects=DEFAULT_TEST_SUBJECTS,
        seed=DEFAULT_SEED,
        split_strategy=DEFAULT_SPLIT_STRATEGY,
        val_percentage=DEFAULT_VAL_PERCENTAGE,
    )
    manifest_path = (
        resolve_output_root(None, dataset_id)
        / "shared_splits"
        / "loso_subject_folds.json"
    )
    fold = build_or_load_loso_folds(session_df, window_df, shared_cfg, manifest_path)[0]
    split = split_indices_for_fold(session_df, window_df, fold)
    loader = build_loader(cfg, session_df, pre, window_df, split.train_indices)
    sample = sample_window_array(loader, split.train_indices[0])
    return DatasetShape(
        dataset_id=dataset_id,
        window_size=int(sample.shape[0]),
        num_channels=int(sample.shape[1]),
        num_classes=int(cfg.num_of_activities),
        source=f"preprocessing:{fold.fold_id}",
    )


def resolve_dataset_shape(
    dataset_id: str,
    artifacts_root: Path,
    ce_stage_name: str,
    datasets_dir: Path,
    metadata_source: str,
    fallback_to_preprocessing: bool,
) -> DatasetShape:
    if metadata_source == "checkpoint":
        shape = _load_shape_from_checkpoint(dataset_id, artifacts_root, ce_stage_name)
        if shape is not None:
            return shape
        if not fallback_to_preprocessing:
            raise FileNotFoundError(
                f"No usable checkpoint metadata found under "
                f"{artifacts_root / dataset_id / ce_stage_name}."
            )
    return _load_shape_from_preprocessing(dataset_id, datasets_dir)


def write_outputs(
    rows: list[CountRow],
    prototype_rows: list[PrototypeCountRow],
    output_dir: Path,
    count_config: CountConfig,
    prototype_config: PrototypeCountConfig,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tinierhar_backbone_flops.json"
    csv_path = output_dir / "tinierhar_backbone_flops.csv"
    prototype_csv_path = output_dir / "tinierhar_prototype_estimation_flops.csv"

    payload = {
        "description": (
            "TinierHAR.encode FLOPs for the backbone path used by "
            "07_eval_bayesian_support_prototypes_loso.py, plus prototype-estimation "
            "FLOPs for the supervised MAP update in script 07 and the unlabeled "
            "MAP-EM update in script 09. Backbone classifier/projection heads are "
            "excluded; prototype counts assume embeddings are already computed."
        ),
        "counting_convention": asdict(count_config),
        "prototype_counting_convention": asdict(prototype_config),
        "backbone_config": asdict(DEFAULT_CONFIG.backbone),
        "backbone_rows": [asdict(row) for row in rows],
        "prototype_rows": [asdict(row) for row in prototype_rows],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    prototype_fieldnames = (
        list(asdict(prototype_rows[0]).keys()) if prototype_rows else []
    )
    with prototype_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prototype_fieldnames)
        writer.writeheader()
        for row in prototype_rows:
            writer.writerow(asdict(row))
    return json_path, csv_path, prototype_csv_path


def print_backbone_table(rows: list[CountRow]) -> None:
    headers = (
        "dataset",
        "input",
        "conv_t",
        "gru_in",
        "params",
        "MACs",
        "FLOPs",
    )
    table = [
        (
            row.dataset_id,
            row.input_shape,
            str(row.conv_time),
            str(row.gru_input_dim),
            f"{row.backbone_params:,}",
            f"{row.macs:,}",
            f"{row.flops:,}",
        )
        for row in rows
    ]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in table))
        for idx in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in table:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def print_prototype_table(rows: list[PrototypeCountRow]) -> None:
    headers = (
        "dataset",
        "k",
        "C",
        "d",
        "N",
        "MAP FLOPs",
        "MAP-EM MACs",
        "Center FLOPs",
        "MAP-EM FLOPs",
    )
    table = [
        (
            row.dataset_id,
            str(row.k),
            str(row.num_classes),
            str(row.embedding_dim),
            str(row.support_embeddings),
            f"{row.map_flops:,}",
            f"{row.map_em_macs:,}",
            f"{row.map_em_centering_flops:,}",
            f"{row.map_em_flops:,}",
        )
        for row in rows
    ]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in table))
        for idx in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in table:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute TinierHAR backbone FLOPs, excluding the classification head, "
            "for the MAP and MAP-EM setup used in the BayaHAR paper."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset ids to evaluate (default: hapt harth wear hhar).",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=ROOT / "datasets",
        help="Dataset directory used only if preprocessing fallback is needed.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=ROOT / "artifacts" / "datasets",
        help="Root containing per-dataset experiment artifacts.",
    )
    parser.add_argument(
        "--ce-stage-name",
        default="classifier",
        help="Classifier stage name used for checkpoint metadata.",
    )
    parser.add_argument(
        "--metadata-source",
        choices=("checkpoint", "preprocessing"),
        default="checkpoint",
        help="Where to infer exact window/channel/class metadata from.",
    )
    parser.add_argument(
        "--no-preprocess-fallback",
        action="store_true",
        help="Fail instead of running preprocessing when checkpoint metadata is missing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "tables" / "computational_cost",
        help="Directory for JSON and CSV outputs.",
    )
    parser.add_argument(
        "--prototype-k-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_PROTOTYPE_K_VALUES),
        help=(
            "K-shot values for prototype-estimation counts. Defaults to the "
            "positive k values used by the Bayesian prototype scripts."
        ),
    )
    parser.add_argument(
        "--em-iterations",
        type=int,
        default=1,
        help="Number of MAP-EM iterations to count (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count_config = CountConfig()
    prototype_config = PrototypeCountConfig(em_iterations=int(args.em_iterations))
    rows: list[CountRow] = []
    prototype_rows: list[PrototypeCountRow] = []
    for dataset_id in args.datasets:
        dataset_id = str(dataset_id).lower()
        shape = resolve_dataset_shape(
            dataset_id=dataset_id,
            artifacts_root=args.artifacts_root,
            ce_stage_name=args.ce_stage_name,
            datasets_dir=args.datasets_dir,
            metadata_source=args.metadata_source,
            fallback_to_preprocessing=not bool(args.no_preprocess_fallback),
        )
        backbone_row = count_tinierhar_backbone(shape, count_config)
        rows.append(backbone_row)
        prototype_rows.extend(
            count_prototype_estimation(
                shape=shape,
                embedding_dim=int(backbone_row.embedding_dim),
                k_values=args.prototype_k_values,
                count_config=count_config,
                prototype_config=prototype_config,
            )
        )

    print("Backbone encode cost")
    print_backbone_table(rows)
    print("\nPrototype-estimation cost, embeddings already given")
    print_prototype_table(prototype_rows)
    json_path, csv_path, prototype_csv_path = write_outputs(
        rows,
        prototype_rows,
        args.output_dir,
        count_config,
        prototype_config,
    )
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {prototype_csv_path}")


if __name__ == "__main__":
    main()

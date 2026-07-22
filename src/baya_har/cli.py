import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from baya_har.experiments import (
    evaluate_logistic,
    evaluate_map,
    evaluate_map_em,
    evaluate_oftta,
    evaluate_original,
    evaluate_pda,
    evaluate_prior,
    evaluate_protonet,
    plot_adaptation_curves,
    plot_map_em_tsne,
    plot_overview,
    train_classifier,
)
from baya_har.experiments.common import ARTIFACTS_ROOT, ROOT

DATASETS = ("hhar", "wear", "harth", "hapt")
ADAPTATION_STAGES: dict[str, Any] = {
    "protonet": evaluate_protonet,
    "map": evaluate_map,
    "map-em": evaluate_map_em,
    "pda": evaluate_pda,
    "oftta": evaluate_oftta,
    "logistic": evaluate_logistic,
}
DEFAULT_METHODS = ("original", "prior", *ADAPTATION_STAGES)


def _k_values(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError("shot ranges must be ascending")
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    result = tuple(dict.fromkeys(values))
    if not result or any(shot < 0 for shot in result):
        raise argparse.ArgumentTypeError("shots must be non-negative")
    return result


def _dataset_artifact_dir(args: argparse.Namespace, dataset: str) -> Path:
    return Path(args.artifacts_dir) / "datasets" / dataset


def _base_config(module: Any, args: argparse.Namespace, dataset: str) -> Any:
    overrides: dict[str, Any] = {
        "dataset_id": dataset,
        "datasets_dir": str(args.datasets_dir),
        "output_root": str(_dataset_artifact_dir(args, dataset)),
    }
    for name in ("device", "max_folds", "force_rerun"):
        arg_name = "force" if name == "force_rerun" else name
        value = getattr(args, arg_name, None)
        if value is not None and hasattr(module.RUN_CONFIG, name):
            overrides[name] = value
    if hasattr(module.RUN_CONFIG, "k_values") and hasattr(args, "shots"):
        overrides["k_values"] = args.shots
    if hasattr(module.RUN_CONFIG, "episodes_per_k") and hasattr(args, "episodes"):
        overrides["episodes_per_k"] = args.episodes
    return replace(module.RUN_CONFIG, **overrides)


def _run_training(args: argparse.Namespace) -> None:
    for dataset in args.datasets:
        print(f"\n== Train classifier: {dataset} ==")
        train_classifier.run(_base_config(train_classifier, args, dataset))


def _run_evaluation(args: argparse.Namespace) -> None:
    for dataset in args.datasets:
        print(f"\n== Evaluate: {dataset} ==")
        if "original" in args.methods:
            evaluate_original.run(_base_config(evaluate_original, args, dataset))
        if "prior" in args.methods:
            evaluate_prior.run(_base_config(evaluate_prior, args, dataset))
        for name, module in ADAPTATION_STAGES.items():
            if name in args.methods:
                module.run(_base_config(module, args, dataset))


def _run_figures(args: argparse.Namespace) -> None:
    dataset_root = Path(args.artifacts_dir) / "datasets"
    use_paper = args.results_source == "paper"
    overview_source_csv = (
        Path(args.artifacts_dir) / "results" / "paper_overview_results.csv"
        if use_paper
        else None
    )
    curve_source_csv = (
        Path(args.artifacts_dir) / "results" / "paper_results.csv"
        if use_paper
        else None
    )
    plot_overview.run(
        replace(
            plot_overview.RUN_CONFIG,
            output_root=str(dataset_root),
            results_source=args.results_source,
            source_records_csv=(
                str(overview_source_csv) if overview_source_csv is not None else None
            ),
        )
    )
    plot_adaptation_curves.run(
        replace(
            plot_adaptation_curves.RUN_CONFIG,
            output_root=str(dataset_root),
            results_source=args.results_source,
            source_records_csv=(
                str(curve_source_csv) if curve_source_csv is not None else None
            ),
        )
    )
    if not args.skip_tsne:
        plot_map_em_tsne.run(_base_config(plot_map_em_tsne, args, "harth"))


def _run_flops(args: argparse.Namespace) -> None:
    from baya_har.experiments.compute_flops import (
        CountConfig,
        PrototypeCountConfig,
        count_prototype_estimation,
        count_tinierhar_backbone,
        resolve_dataset_shape,
        write_outputs,
    )

    count_config = CountConfig()
    prototype_config = PrototypeCountConfig(em_iterations=1)
    rows = []
    prototype_rows = []
    dataset_root = Path(args.artifacts_dir) / "datasets"
    for dataset in args.datasets:
        shape = resolve_dataset_shape(
            dataset_id=dataset,
            artifacts_root=dataset_root,
            ce_stage_name="classifier",
            datasets_dir=Path(args.datasets_dir),
            metadata_source="checkpoint",
            fallback_to_preprocessing=True,
        )
        row = count_tinierhar_backbone(shape, count_config)
        rows.append(row)
        prototype_rows.extend(
            count_prototype_estimation(
                shape=shape,
                embedding_dim=row.embedding_dim,
                k_values=(1, 16),
                count_config=count_config,
                prototype_config=prototype_config,
            )
        )
    write_outputs(
        rows,
        prototype_rows,
        Path(args.artifacts_dir) / "tables" / "computational_cost",
        count_config,
        prototype_config,
    )


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--datasets-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--force", action="store_true", default=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="baya-har",
        description="Run the camera-ready BayaHAR experiment pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train LOSO TinierHAR classifiers.")
    _add_shared_arguments(train)

    evaluate = subparsers.add_parser("evaluate", help="Run paper evaluations.")
    _add_shared_arguments(evaluate)
    evaluate.add_argument(
        "--methods", nargs="+", choices=DEFAULT_METHODS, default=list(DEFAULT_METHODS)
    )
    evaluate.add_argument("--shots", type=_k_values, default=tuple(range(17)))
    evaluate.add_argument("--episodes", type=int, default=100)

    figures = subparsers.add_parser("figures", help="Regenerate paper figures 2-4.")
    _add_shared_arguments(figures)
    figures.add_argument(
        "--results-source",
        choices=("paper", "live"),
        default="paper",
        help="Use frozen paper aggregates or live per-dataset evaluation outputs.",
    )
    figures.add_argument("--skip-tsne", action="store_true")

    flops = subparsers.add_parser("flops", help="Regenerate computational-cost tables.")
    _add_shared_arguments(flops)

    reproduce = subparsers.add_parser(
        "reproduce", help="Run training, evaluation, figures, and FLOP accounting."
    )
    _add_shared_arguments(reproduce)
    reproduce.add_argument("--shots", type=_k_values, default=tuple(range(17)))
    reproduce.add_argument("--episodes", type=int, default=100)
    reproduce.add_argument(
        "--results-source",
        choices=("paper", "live"),
        default="paper",
        help="Use frozen paper aggregates or live per-dataset evaluation outputs.",
    )
    reproduce.add_argument("--skip-tsne", action="store_true")
    reproduce.set_defaults(methods=list(DEFAULT_METHODS))
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        _run_training(args)
    elif args.command == "evaluate":
        _run_evaluation(args)
    elif args.command == "figures":
        _run_figures(args)
    elif args.command == "flops":
        _run_flops(args)
    elif args.command == "reproduce":
        _run_training(args)
        _run_evaluation(args)
        _run_figures(args)
        _run_flops(args)


if __name__ == "__main__":
    main()

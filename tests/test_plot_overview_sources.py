import json
from dataclasses import replace

from baya_har.experiments.plot_overview import RUN_CONFIG, _collect


def test_paper_source_uses_only_complete_frozen_records(tmp_path) -> None:
    source_path = tmp_path / "paper_overview_results.csv"
    source_path.write_text(
        "dataset_id,method_key,n,value,fold_std\n"
        "toy,original,0,0.1,0.01\n"
        "toy,fixed,16,0.2,0.02\n"
        "toy,map_em_centered,16,0.3,0.03\n"
        "toy,bayesian,16,0.4,0.04\n",
        encoding="utf-8",
    )
    config = replace(
        RUN_CONFIG,
        output_root=str(tmp_path / "missing-live-results"),
        results_source="paper",
        source_records_csv=str(source_path),
        dataset_ids=("toy",),
    )

    frame, warnings = _collect(config)

    assert warnings == []
    assert frame["macro_f1"].tolist() == [0.1, 0.2, 0.3, 0.4]
    assert frame["meta"].tolist() == [
        "baseline",
        "fixed:n=16:source22",
        "map_em_centered:n=16:source22",
        "bayesian:n=16:source22",
    ]


def test_live_source_does_not_read_paper_records_and_separates_map_from_map_em(
    tmp_path,
) -> None:
    dataset_dir = tmp_path / "datasets" / "toy"
    (dataset_dir / "original").mkdir(parents=True)
    (dataset_dir / "prior_euclidean").mkdir()
    (dataset_dir / "map_em_euclidean_centered").mkdir()
    (dataset_dir / "map_euclidean").mkdir()
    summary = {"folds": [{"test_macro_f1": 0.1}, {"test_macro_f1": 0.3}]}
    (dataset_dir / "original" / "summary.json").write_text(json.dumps(summary))
    (dataset_dir / "prior_euclidean" / "summary.json").write_text(
        json.dumps(summary)
    )
    curve_header = "k,macro_f1_mean,macro_f1_subject_std\n"
    (dataset_dir / "map_em_euclidean_centered" / "overall_by_k_results.csv").write_text(
        curve_header + "16,0.7,0.07\n"
    )
    (dataset_dir / "map_euclidean" / "overall_by_k_results.csv").write_text(
        curve_header + "16,0.8,0.08\n"
    )
    config = replace(
        RUN_CONFIG,
        output_root=str(tmp_path / "datasets"),
        results_source="live",
        source_records_csv=None,
        dataset_ids=("toy",),
    )

    frame, warnings = _collect(config)

    assert warnings == []
    values = dict(zip(frame["method"], frame["macro_f1"]))
    assert values["MAP-EM Proto (16-Shot)"] == 0.7
    assert values["MAP Proto (16-Shot)"] == 0.8
    assert values["Original Classifier"] == 0.2

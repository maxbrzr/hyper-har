from baya_har.experiments.plot_adaptation_curves import (
    RUN_CONFIG,
    _choose_stage_dir,
    _method_specs_for_mode,
)


def test_map_curve_does_not_select_map_em_results(tmp_path) -> None:
    (tmp_path / "map_em_euclidean_centered").mkdir()
    expected = tmp_path / "map_euclidean"
    expected.mkdir()
    map_spec = next(
        spec
        for spec in _method_specs_for_mode(RUN_CONFIG, "supervised")
        if spec.key == "bayesian"
    )

    assert _choose_stage_dir(tmp_path, map_spec) == expected

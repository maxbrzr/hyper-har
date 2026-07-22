import pytest

from baya_har.metrics import observed_class_macro_f1, present_class_labels


def test_macro_f1_ignores_classes_absent_from_targets() -> None:
    assert observed_class_macro_f1([0, 1], [0, 1]) == pytest.approx(1.0)


def test_prediction_of_absent_class_is_included_as_zero_f1_class() -> None:
    assert observed_class_macro_f1([0, 1], [0, 2]) == pytest.approx(1.0 / 3.0)


def test_empty_targets_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty targets"):
        present_class_labels([])


def test_shared_evaluation_keeps_global_confusion_matrix_without_f1_penalty() -> None:
    from baya_har.experiments.common import classification_metrics

    metrics = classification_metrics([0, 1], [0, 1], num_classes=4)

    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert len(metrics["confusion_matrix"]) == 4

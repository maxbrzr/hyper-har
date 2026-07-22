from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score


def present_class_labels(y_true: Sequence[int] | np.ndarray) -> list[int]:
    """Return the sorted classes represented in an evaluation target."""
    true_np = np.asarray(y_true, dtype=np.int64)
    if true_np.size == 0:
        raise ValueError("Cannot compute classification metrics for empty targets.")
    return [int(label) for label in np.unique(true_np).tolist()]


def observed_class_macro_f1(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> float:
    """Compute macro-F1 over classes observed in targets or predictions."""
    true_np = np.asarray(y_true, dtype=np.int64)
    pred_np = np.asarray(y_pred, dtype=np.int64)
    present_class_labels(true_np)
    return float(
        f1_score(
            true_np,
            pred_np,
            average="macro",
            zero_division=0,
        )
    )

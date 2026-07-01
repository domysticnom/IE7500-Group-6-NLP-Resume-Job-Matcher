"""Shared evaluation metrics — both approaches call this identically."""
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ..scoring.base import LABELS


def evaluate(y_true, y_pred) -> dict:
    """Accuracy, macro-F1, per-class report, and confusion matrix over LABELS."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "report": classification_report(y_true, y_pred, labels=LABELS, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }

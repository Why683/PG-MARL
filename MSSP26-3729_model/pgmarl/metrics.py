from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(labels: np.ndarray, predictions: np.ndarray,
                           num_classes: int = 5) -> Dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(num_classes)), zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "support": support.astype(int).tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(num_classes))).astype(int).tolist(),
    }


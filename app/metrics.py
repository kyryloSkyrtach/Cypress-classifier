"""Evaluation metrics — pure Python, no scikit-learn."""

from __future__ import annotations

from app.features import IDX_TO_LABEL


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Fraction of correctly classified samples."""
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred, strict=False)) / len(y_true)


def confusion_matrix(y_true: list[int], y_pred: list[int], n_classes: int = 3) -> list[list[int]]:
    """Return n_classes × n_classes confusion matrix (rows=true, cols=pred)."""
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred, strict=False):
        cm[t][p] += 1
    return cm


def classification_report(y_true: list[int], y_pred: list[int], n_classes: int = 3) -> str:
    """Return a formatted string with per-class and macro metrics."""
    cm = confusion_matrix(y_true, y_pred, n_classes)

    precisions, recalls, f1s = [], [], []
    lines = [f"{'Class':<16} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Support':>9}"]
    lines.append("-" * 57)

    for c in range(n_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n_classes)) - tp
        fn = sum(cm[c][cc] for cc in range(n_classes)) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        label = IDX_TO_LABEL.get(c, str(c))
        lines.append(f"{label:<16} {precision:>9.3f} {recall:>9.3f} {f1:>9.3f} {support:>9}")

    lines.append("-" * 57)
    macro_p = sum(precisions) / n_classes
    macro_r = sum(recalls) / n_classes
    macro_f1 = sum(f1s) / n_classes
    acc = accuracy(y_true, y_pred)
    total = len(y_true)
    lines.append(f"{'macro avg':<16} {macro_p:>9.3f} {macro_r:>9.3f} {macro_f1:>9.3f} {total:>9}")
    lines.append(f"\nAccuracy: {acc:.4f}  ({sum(t == p for t, p in zip(y_true, y_pred, strict=False))}/{total})")
    return "\n".join(lines)


def print_confusion_matrix(cm: list[list[int]], n_classes: int = 3) -> None:
    """Pretty-print the confusion matrix with class labels."""
    labels = [IDX_TO_LABEL.get(c, str(c)) for c in range(n_classes)]
    col_w = max(len(lb) for lb in labels) + 2
    header = " " * (col_w + 2) + "".join(f"{lb:>{col_w}}" for lb in labels)
    print(header)
    print(" " * (col_w + 2) + "-" * (col_w * n_classes))
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>{col_w}}" for v in row)
        print(f"{labels[i]:<{col_w}} |{row_str}")

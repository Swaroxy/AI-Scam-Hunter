"""Evaluation helpers shared by every model (baseline, LoRA, RAG)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from .config import LABELS


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """Metrics for the positive ('suspicious') class plus macro-F1 and, if scores given, ROC/PR AUC."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    _, _, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    out = {"precision": p, "recall": r, "f1": f, "macro_f1": mf, "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))}
    if y_score is not None and len(set(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    return {k: round(float(v), 4) for k, v in out.items()}


def plot_confusion(y_true, y_pred, title: str = "", ax=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=[LABELS[0], LABELS[1]])
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    return cm


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(results).T[["precision", "recall", "f1", "macro_f1", "accuracy"] +
                                   [c for c in ("roc_auc", "pr_auc") if any(c in r for r in results.values())]]


def choose_threshold(y_val, scores, min_recall: float = 0.95) -> float:
    """Pick the highest threshold on the VALIDATION set that still reaches `min_recall`.

    For scam detection, missing a scam (false negative) is usually costlier than a false alarm,
    so we tune the threshold for recall on validation data -- never on the test set.
    """
    y_val, scores = np.asarray(y_val), np.asarray(scores)
    best = 0.5
    for t in np.linspace(0.05, 0.95, 91):
        pred = (scores >= t).astype(int)
        tp, fn = ((pred == 1) & (y_val == 1)).sum(), ((pred == 0) & (y_val == 1)).sum()
        if tp / max(tp + fn, 1) >= min_recall:
            best = float(t)
    return round(best, 2)


def error_analysis(df: pd.DataFrame, y_pred, n: int = 8) -> pd.DataFrame:
    """Return misclassified rows (with category) so we can read what the model gets wrong."""
    wrong = df.assign(pred=np.asarray(y_pred))
    wrong = wrong[wrong["pred"] != wrong["label"]]
    return wrong[["category", "label", "pred", "text"]].head(n)

"""Dataset-quality helpers: deduplication, group-aware splitting, leakage checks, label audit."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from .config import SEED


def normalize_text(text: str) -> str:
    """Case/whitespace/punctuation-insensitive key used for duplicate detection."""
    t = re.sub(r"\s+", " ", text.lower()).strip()
    return re.sub(r"[^\w\s]+$", "", t).strip()


# ----------------------------------------------------------------------------- dedup
def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact/normalised duplicates; report label conflicts (same text, different label)."""
    df = df.copy()
    df["_key"] = df["text"].map(normalize_text)
    conflicts = df.groupby("_key")["label"].nunique()
    n_conflicts = int((conflicts > 1).sum())
    before = len(df)
    df = df.drop_duplicates("_key", keep="first").drop(columns="_key").reset_index(drop=True)
    return df, {"before": before, "after": len(df), "removed": before - len(df), "label_conflicts": n_conflicts}


# ----------------------------------------------------------------------------- splitting
def assign_splits(df: pd.DataFrame, seed: int = SEED, val_groups: int = 1, test_groups: int = 1) -> pd.DataFrame:
    """Group-aware split: within each category, whole template groups go to ONE split only.

    A random row-level split would put near-identical template variants in both train and
    test and inflate every metric (template leakage). Here the test set contains templates
    the model has never seen, a much more honest estimate of generalisation.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = "train"
    for _, sub in df.groupby("category"):
        groups = sorted(sub["group"].unique())
        rng.shuffle(groups)
        test_g = groups[:test_groups]
        val_g = groups[test_groups:test_groups + val_groups]
        df.loc[df["group"].isin(test_g), "split"] = "test"
        df.loc[df["group"].isin(val_g), "split"] = "val"
    return df


# ----------------------------------------------------------------------------- leakage
def near_duplicate_count(a: pd.Series, b: pd.Series, threshold: float = 0.9) -> int:
    """Count texts in `b` whose TF-IDF cosine similarity to any text in `a` exceeds threshold."""
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(pd.concat([a, b]))
    sims = (vec.transform(b) @ vec.transform(a).T).max(axis=1).toarray().ravel()
    return int((sims >= threshold).sum())


def leakage_report(df: pd.DataFrame, threshold: float = 0.9) -> dict:
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    keys = {s: set(p["text"].map(normalize_text)) for s, p in parts.items()}
    groups = {s: set(p["group"]) for s, p in parts.items()}
    return {
        "exact_overlap_train_test": len(keys["train"] & keys["test"]),
        "exact_overlap_train_val": len(keys["train"] & keys["val"]),
        "group_overlap_train_test": len(groups["train"] & groups["test"]),
        "group_overlap_train_val": len(groups["train"] & groups["val"]),
        f"near_dups_test_vs_train(>={threshold})": near_duplicate_count(parts["train"]["text"], parts["test"]["text"], threshold),
    }


# ----------------------------------------------------------------------------- imbalance
def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("split")["label"].agg(n="count", suspicious="sum")
    out["suspicious_pct"] = (100 * out["suspicious"] / out["n"]).round(1)
    return out


def class_weights(labels) -> dict[int, float]:
    """Inverse-frequency weights (same formula as sklearn's class_weight='balanced')."""
    labels = np.asarray(labels)
    n, k = len(labels), len(np.unique(labels))
    return {int(c): n / (k * (labels == c).sum()) for c in np.unique(labels)}


# ----------------------------------------------------------------------------- label audit
def audit_labels(texts: pd.Series, labels: pd.Series, n_splits: int = 5, flag_below: float = 0.2) -> pd.DataFrame:
    """Flag rows whose given label gets a very low out-of-fold probability (possible mislabels).

    This is a lightweight cousin of 'confident learning': a simple model trained on the other
    folds disagrees strongly with the label. Flagged rows should be reviewed by a human.
    """
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                         LogisticRegression(max_iter=1000, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    proba = cross_val_predict(pipe, texts, labels, cv=cv, method="predict_proba")
    p_given = proba[np.arange(len(labels)), labels.to_numpy()]
    out = pd.DataFrame({"text": texts.to_numpy(), "label": labels.to_numpy(), "p_given_label": p_given.round(3)})
    return out[out["p_given_label"] < flag_below].sort_values("p_given_label")

"""Stage 1 -- evaluation-first baseline: TF-IDF + Logistic Regression."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import SEED
from .evaluate import compute_metrics


def build_baseline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, lowercase=True)),
        ("clf", LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=SEED)),
    ])


def majority_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """The floor: always predict the most common class. Any real model must beat this."""
    dummy = DummyClassifier(strategy="most_frequent").fit(train["text"], train["label"])
    return compute_metrics(test["label"], dummy.predict(test["text"]))


def top_features(pipe: Pipeline, k: int = 12) -> tuple[list[str], list[str]]:
    """Words the linear model associates most with 'suspicious' and with 'legitimate'."""
    names = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coef = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    return names[order[::-1][:k]].tolist(), names[order[:k]].tolist()


def run_baseline(train: pd.DataFrame, test: pd.DataFrame, C: float = 1.0):
    pipe = build_baseline(C).fit(train["text"], train["label"])
    scores = pipe.predict_proba(test["text"])[:, 1]
    preds = (scores >= 0.5).astype(int)
    return pipe, preds, scores, compute_metrics(test["label"], preds, scores)

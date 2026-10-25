"""Model loading for the API: one `Scorer` interface, two backends.

  baseline (default): TF-IDF + LogReg, trained at startup from data/messages_clean.csv (takes ~1s)
  lora:               your trained adapter in artifacts/scam_hunter_lora (needs torch/transformers/peft)

Choose with the SCAM_MODEL environment variable ("baseline" | "lora"). If "lora" is requested
but unavailable, we fall back to the baseline and say so in /health instead of crashing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .baseline import build_baseline
from .config import ARTIFACTS_DIR, CLEAN_CSV, RAW_CSV
from .data_quality import assign_splits, remove_duplicates

LORA_DIR = ARTIFACTS_DIR / "scam_hunter_lora"


@dataclass
class Scorer:
    name: str
    predict_proba: Callable[[Sequence[str]], np.ndarray]  # texts -> P(suspicious)
    note: str = ""


def _training_frame() -> pd.DataFrame:
    if not CLEAN_CSV.exists():
        from .data_gen import generate_dataset
        raw = generate_dataset()
        raw.to_csv(RAW_CSV, index=False)
        assign_splits(remove_duplicates(raw)[0]).to_csv(CLEAN_CSV, index=False)
    return pd.read_csv(CLEAN_CSV)


def load_baseline() -> Scorer:
    df = _training_frame()
    train = df[df["split"] == "train"]
    pipe = build_baseline().fit(train["text"], train["label"])
    return Scorer("tfidf_logreg", lambda texts: pipe.predict_proba(list(texts))[:, 1])


def load_lora() -> Scorer:
    from .finetune import load_adapter, predict_proba  # heavy imports only when needed
    model, tok = load_adapter(str(LORA_DIR))
    return Scorer("distilbert_lora", lambda texts: predict_proba(model, tok, texts))


def load_scorer(kind: str | None = None) -> Scorer:
    kind = (kind or os.getenv("SCAM_MODEL", "baseline")).lower()
    if kind == "lora":
        try:
            return load_lora()
        except Exception as exc:  # missing adapter, no torch, etc.
            scorer = load_baseline()
            scorer.note = f"LoRA requested but unavailable ({type(exc).__name__}); serving baseline instead."
            return scorer
    return load_baseline()

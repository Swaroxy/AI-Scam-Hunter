"""Retrieval (RAG-style) comparison using FAISS.

Two uses of retrieval, both contrasted with fine-tuning in the notebook:
  1. RagClassifier  - classify by retrieving the k most similar *labelled* training messages and voting.
  2. retrieve_guidance - fetch short scam-pattern notes to ground an explanation (no training at all).

Embeddings: sentence-transformers if available (needs a model download), otherwise an offline
TF-IDF + SVD fallback so the demo always runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEED

KNOWLEDGE_BASE = [
    {"title": "Credential phishing", "text": "Messages claiming your bank account is locked or suspended and asking you to log in through a link or reply with a password, PIN or card number are phishing. Real banks do not ask for these by message."},
    {"title": "Prize and lottery lures", "text": "Unexpected prize or lottery wins that require a fee, personal details or fast action are scams. You cannot win a draw you never entered."},
    {"title": "Delivery fee scams", "text": "Fake courier notices ask for a small redelivery or customs fee via a link. Check tracking on the courier's official site instead of using the link."},
    {"title": "Tech support scams", "text": "Unsolicited alerts about viruses or compromised devices that urge you to call a number or install a remote access tool are scams."},
    {"title": "Gift card and boss impersonation", "text": "Requests from a boss or colleague to buy gift cards, change payroll details or wire money secretly and urgently are a classic fraud. Verify by calling the person on a known number."},
    {"title": "Advance fee and investment fraud", "text": "Guaranteed high returns, zero risk, unclaimed inheritances and advance fees to release funds are hallmarks of fraud."},
    {"title": "Benign security notices", "text": "Genuine verification-code and sign-in notices usually ask for no action, tell you never to share the code, and do not contain login links."},
]


class Embedder:
    """Turns text into L2-normalised vectors. backend: 'auto' | 'sbert' | 'tfidf'."""

    def __init__(self, backend: str = "auto", sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.backend = backend
        self._model = None
        if backend in ("auto", "sbert"):
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(sbert_model)
                self.backend = "sbert"
            except Exception as exc:  # no package / no network
                if backend == "sbert":
                    raise
                self.backend = "tfidf"
                self._why = repr(exc)[:80]

    def fit(self, texts) -> "Embedder":
        if self.backend == "tfidf":
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(texts)
            n_comp = max(2, min(128, len(texts) - 1, len(self._tfidf.vocabulary_) - 1))
            self._svd = TruncatedSVD(n_components=n_comp, random_state=SEED).fit(self._tfidf.transform(texts))
        return self

    def encode(self, texts) -> np.ndarray:
        texts = list(texts)
        if self.backend == "sbert":
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        else:
            vecs = self._svd.transform(self._tfidf.transform(texts))
            vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        return np.ascontiguousarray(vecs, dtype="float32")


def _index(vecs: np.ndarray):
    import faiss
    idx = faiss.IndexFlatIP(vecs.shape[1])  # inner product == cosine for normalised vectors
    idx.add(vecs)
    return idx


class RagClassifier:
    """k-NN over labelled training messages stored in a FAISS index. No gradient training."""

    def __init__(self, k: int = 5, backend: str = "auto"):
        self.k, self.embedder = k, Embedder(backend)

    def fit(self, train: pd.DataFrame) -> "RagClassifier":
        self.embedder.fit(train["text"])
        self.labels = train["label"].to_numpy()
        self.texts = train["text"].to_numpy()
        self.index = _index(self.embedder.encode(train["text"]))
        return self

    def neighbours(self, text: str, k: int | None = None):
        sims, ids = self.index.search(self.embedder.encode([text]), k or self.k)
        return [(float(s), int(self.labels[i]), self.texts[i]) for s, i in zip(sims[0], ids[0])]

    def predict_proba(self, texts) -> np.ndarray:
        sims, ids = self.index.search(self.embedder.encode(texts), self.k)
        w = np.clip(sims, 0, None) + 1e-6
        return (w * self.labels[ids]).sum(axis=1) / w.sum(axis=1)


def retrieve_guidance(text: str, embedder: Embedder | None = None, k: int = 2) -> list[dict]:
    """Retrieve the most relevant scam-pattern notes for a message (grounding for explanations)."""
    embedder = embedder or Embedder("auto")
    docs = [d["title"] + ". " + d["text"] for d in KNOWLEDGE_BASE]
    embedder.fit(docs + [text])
    index = _index(embedder.encode(docs))
    _, ids = index.search(embedder.encode([text]), k)
    return [KNOWLEDGE_BASE[i] for i in ids[0]]

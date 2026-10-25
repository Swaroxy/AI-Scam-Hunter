"""FastAPI service for AI Scam Hunter.

Run:   uvicorn scam_hunter.api:app --reload
Docs:  http://127.0.0.1:8000/docs
Model: SCAM_MODEL=lora uvicorn scam_hunter.api:app     (after training in the notebook)

Privacy: messages are scored in memory and never stored or logged by this service.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .report import DISCLAIMER, build_report
from .serving import Scorer, load_scorer

MAX_CHARS = 5000
MAX_BATCH = 50


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS, description="Email or SMS text to screen.")


class BatchRequest(BaseModel):
    messages: list[AnalyzeRequest] = Field(..., min_length=1, max_length=MAX_BATCH)


class WarningSign(BaseModel):
    type: str
    evidence: str


class RiskReport(BaseModel):
    verdict: str = Field(description="suspicious | uncertain | no_strong_indicators (never 'safe')")
    risk_score: int = Field(ge=0, le=100)
    model_probability_suspicious: float
    warning_signs: list[WarningSign]
    recommended_action: str
    disclaimer: str = DISCLAIMER


class BatchResponse(BaseModel):
    count: int
    reports: list[RiskReport]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.scorer = load_scorer()  # loaded once at startup
    yield


app = FastAPI(
    title="AI Scam Hunter",
    version="1.0.0",
    description="Screens fictional/example emails and SMS for scam indicators. "
                "A screening aid only: it never guarantees a message is safe.",
    lifespan=lifespan,
)


def _scorer() -> Scorer:
    scorer = getattr(app.state, "scorer", None)
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return scorer


@app.get("/health")
def health():
    s = _scorer()
    return {"status": "ok", "model": s.name, "note": s.note}


@app.post("/analyze", response_model=RiskReport)
def analyze(req: AnalyzeRequest):
    p = float(_scorer().predict_proba([req.text])[0])
    return build_report(req.text, p)


@app.post("/analyze/batch", response_model=BatchResponse)
def analyze_batch(req: BatchRequest):
    texts = [m.text for m in req.messages]
    probs = _scorer().predict_proba(texts)
    reports = [build_report(t, float(p)) for t, p in zip(texts, probs)]
    return {"count": len(reports), "reports": reports}

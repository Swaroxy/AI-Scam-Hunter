"""API tests. Run: python -m pytest -q   (needs fastapi + httpx)"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from scam_hunter.api import app  # noqa: E402

SCAM = "Dear customer, your account will be suspended in 6 hours. Verify now: http://northwind-verify-login.example/123"
HAM = "Hi Sam, are we still on for dinner on Fri 14 Nov? I found a new place near the station."


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs lifespan -> loads the model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_analyze_scam_vs_normal(client):
    scam = client.post("/analyze", json={"text": SCAM}).json()
    ham = client.post("/analyze", json={"text": HAM}).json()
    assert scam["verdict"] == "suspicious" and scam["warning_signs"]
    assert ham["verdict"] != "suspicious"
    assert scam["risk_score"] > ham["risk_score"]


def test_never_claims_safe(client):
    for text in (SCAM, HAM, "ok"):
        body = client.post("/analyze", json={"text": text}).json()
        assert body["verdict"] in {"suspicious", "uncertain", "no_strong_indicators"}
        assert "not mean a message is safe" in body["disclaimer"]


def test_batch(client):
    r = client.post("/analyze/batch", json={"messages": [{"text": SCAM}, {"text": HAM}]})
    assert r.status_code == 200 and r.json()["count"] == 2


def test_validation_errors(client):
    assert client.post("/analyze", json={"text": ""}).status_code == 422
    assert client.post("/analyze", json={"text": "x" * 6000}).status_code == 422
    assert client.post("/analyze", json={}).status_code == 422
    assert client.post("/analyze/batch", json={"messages": []}).status_code == 422

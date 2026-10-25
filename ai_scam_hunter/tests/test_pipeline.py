"""Fast sanity tests (no transformers needed). Run: python -m pytest -q"""
import pandas as pd
import pytest

from scam_hunter.data_gen import generate_dataset
from scam_hunter.data_quality import assign_splits, leakage_report, normalize_text, remove_duplicates
from scam_hunter.report import build_report, validate_report


@pytest.fixture(scope="module")
def data():
    clean, _ = remove_duplicates(generate_dataset(seed=0))
    return assign_splits(clean, seed=0)


def test_dataset_is_reproducible():
    assert generate_dataset(seed=1).equals(generate_dataset(seed=1))


def test_deduplication_removes_injected_duplicates():
    raw = generate_dataset(seed=0)
    clean, info = remove_duplicates(raw)
    assert info["removed"] > 0 and clean["text"].map(normalize_text).is_unique
    assert info["label_conflicts"] == 0


def test_no_leakage_between_splits(data):
    rep = leakage_report(data)
    assert rep["exact_overlap_train_test"] == 0 and rep["group_overlap_train_test"] == 0
    assert set(data["split"]) == {"train", "val", "test"}
    assert all(data[data.split == s]["label"].nunique() == 2 for s in ("train", "val", "test"))


def test_report_schema_and_never_safe():
    for p in (0.0, 0.2, 0.5, 0.9, 1.0):
        r = build_report("Hello, lunch at noon?", p)
        validate_report(r)
        assert r["verdict"] != "safe" and "not mean a message is safe" in r["disclaimer"]


def test_strong_rule_signals_override_low_model_score():
    text = "Send the gift cards today and keep this between us. Confirm your password."
    assert build_report(text, 0.05)["verdict"] == "uncertain"


def test_validate_rejects_bad_reports():
    with pytest.raises(ValueError):
        validate_report({"verdict": "safe", "risk_score": 1, "warning_signs": [], "recommended_action": ""})

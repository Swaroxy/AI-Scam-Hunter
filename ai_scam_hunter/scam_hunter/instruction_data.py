"""Instruction-tuning data: (instruction, input, output) triples with JSON outputs.

The same labelled messages from stage 1 are re-expressed as instruction-following examples,
so a small generative model can be taught to emit the structured report directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .report import DISCLAIMER, detect_signals

INSTRUCTION = ("Analyze the message below. Decide whether it is suspicious or legitimate, list the warning "
               "signs (quoting the matching text from the message when there is one), and answer with a JSON risk report only.")


def target_report(row: pd.Series) -> dict:
    """Ground-truth report built from dataset labels (not from a model)."""
    if int(row["label"]) == 1:
        annotated = set(json.loads(row["signs"]))
        evidence = {s["type"]: s["evidence"] for s in detect_signals(row["text"])}
        signs = [{"type": t, "evidence": evidence.get(t, "")} for t in annotated]
        return {"verdict": "suspicious", "risk_score": 90, "warning_signs": signs,
                "recommended_action": "Do not click links or send money/codes; verify via an official channel."}
    return {"verdict": "no_strong_indicators", "risk_score": 10, "warning_signs": [],
            "recommended_action": "No strong warning signs, but verify unexpected requests via an official channel."}


def to_instruction_records(df: pd.DataFrame) -> list[dict]:
    return [{"instruction": INSTRUCTION, "input": r["text"],
             "output": json.dumps(target_report(r), ensure_ascii=False)} for _, r in df.iterrows()]


def format_prompt(rec: dict) -> str:
    """Prompt template used for BOTH training and inference (keep them identical!)."""
    return f"### Instruction:\n{rec['instruction']}\n\n### Input:\n{rec['input']}\n\n### Response:\n"


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

"""Structured JSON risk reports: model probability + transparent rule-based evidence.

Design rule: this system NEVER declares a message "safe". The lowest-risk verdict is
`no_strong_indicators`, and strong rule-based signals can override a low model score by
pushing the verdict to `uncertain`.
"""
from __future__ import annotations

import re

from .config import ORG_NAMES

DISCLAIMER = ("Automated screening only. A low score does not mean a message is safe; "
              "verify through the sender's official channel before acting.")

VERDICTS = ("suspicious", "uncertain", "no_strong_indicators")

_URL = re.compile(r"https?://[^\s]+", re.I)
_SIGNALS: dict[str, re.Pattern] = {
    "urgency": re.compile(r"\b(urgent(ly)?|immediately|within \d+ hours|act now|final notice|as soon as you|"
                          r"will be (suspended|disabled|closed|deleted|destroyed|returned))\b", re.I),
    "credential_request": re.compile(r"\b(password|pin|username|card number|verify your identity|confirm your (card )?details|"
                                     r"update your details|full name and address)\b", re.I),
    "payment_request": re.compile(r"\b(gift cards?|prepaid cards?|wire transfer|crypto|processing fee|handling fee|"
                                  r"advance fee|redelivery charge|customs fee|shipping (charge|adjustment)|deposit)\b", re.I),
    "prize_lure": re.compile(r"\b(congratulations|you have won|lucky winner|unclaimed|claim your|lottery|randomly selected)\b", re.I),
    "threat": re.compile(r"\b(suspended|locked|compromised|hackers|virus|destroyed|disabled)\b", re.I),
    "secrecy": re.compile(r"\b(keep this between us|do not discuss|confidential|quietly)\b", re.I),
    "remote_access": re.compile(r"\b(remote tool|connect to your computer|support tool)\b", re.I),
    "too_good_to_be_true": re.compile(r"(guarantees?|zero risk|doubled their money|\d+% returns|inheritance|beneficiary)", re.I),
}
_STRONG = {"credential_request", "payment_request", "remote_access", "secrecy"}


def _risky_link(url: str) -> bool:
    u = url.lower()
    return u.startswith("http://") or bool(re.search(r"(verify|secure|login|claim|update|pay)-", u))


def detect_signals(text: str) -> list[dict]:
    """Return [{'type', 'evidence'}] where evidence is the exact text that triggered the rule."""
    found = []
    for name, pat in _SIGNALS.items():
        m = pat.search(text)
        if m:
            found.append({"type": name, "evidence": m.group(0)})
    for url in _URL.findall(text):
        if _risky_link(url):
            found.append({"type": "suspicious_link", "evidence": url.rstrip(".,)")})
            break
    kinds = {s["type"] for s in found}
    if any(org.lower() in text.lower() for org in ORG_NAMES) and kinds & {"suspicious_link", "credential_request"}:
        found.append({"type": "impersonation", "evidence": "mentions an organisation while asking for a link click or credentials"})
    return found


def build_report(text: str, p_suspicious: float, high: float = 0.7, low: float = 0.3) -> dict:
    """Combine a model probability with rule-based evidence into a JSON-serialisable report."""
    signs = detect_signals(text)
    n_strong = sum(s["type"] in _STRONG for s in signs)
    if p_suspicious >= high:
        verdict = "suspicious"
    elif p_suspicious <= low and n_strong < 2:
        verdict = "no_strong_indicators"
    else:
        verdict = "uncertain"  # also reached when the model says 'low' but the rules disagree
    risk = round(100 * (0.75 * p_suspicious + 0.25 * min(1.0, len(signs) / 4)))
    action = {
        "suspicious": "Do not click links, call numbers, or send money/codes. Contact the sender through an official channel you already trust.",
        "uncertain": "Treat with caution. Verify independently before acting on anything in this message.",
        "no_strong_indicators": "No strong warning signs found, but this is not a guarantee. Verify unexpected requests through official channels.",
    }[verdict]
    return {
        "verdict": verdict,
        "risk_score": int(risk),
        "model_probability_suspicious": round(float(p_suspicious), 3),
        "warning_signs": signs,
        "recommended_action": action,
        "disclaimer": DISCLAIMER,
    }


def validate_report(report: dict) -> None:
    """Raise ValueError if a (possibly LLM-generated) report breaks the schema."""
    need = {"verdict", "risk_score", "warning_signs", "recommended_action"}
    if missing := need - set(report):
        raise ValueError(f"missing keys: {sorted(missing)}")
    if report["verdict"] not in VERDICTS:
        raise ValueError(f"bad verdict: {report['verdict']!r}")
    if not (isinstance(report["risk_score"], (int, float)) and 0 <= report["risk_score"] <= 100):
        raise ValueError("risk_score must be 0-100")
    if not isinstance(report["warning_signs"], list):
        raise ValueError("warning_signs must be a list")

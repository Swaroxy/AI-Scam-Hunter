# 🛡️ AI Scam Hunter

An end-to-end, reproducible AI pipeline that classifies **fictional** emails and SMS as *suspicious* or *legitimate*, points out the warning signs, and produces a **structured JSON risk report**. Built to teach (and show off) an evaluation-first approach to applied AI.

> **Safety:** all data is synthetic (`.example` domains, `555` numbers). The system is a screening aid and **never** guarantees a message is safe. Its lowest-risk verdict is `no_strong_indicators`.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter lab AI_Scam_Hunter.ipynb                       # run all cells
python -m pytest -q                                    # fast sanity tests
```

The notebook runs without the transformer stack (the LoRA section is skipped with a message). For the full run you need `torch`, `transformers`, `peft` and internet access once, to download `distilbert-base-uncased`. CPU is fine (a few minutes); a GPU is faster.

## What's inside

| Learning objective | Where |
|---|---|
| Fine-tuning vs. RAG | Notebook §1 (decision guide), §7 (FAISS k-NN + guidance retrieval) |
| Instruction tuning, structured output | §5, `instruction_data.py`, optional `finetune_causal.py` |
| LoRA / PEFT | §6, `finetune.py` (DistilBERT + LoRA, class-weighted loss) |
| Dataset quality | §3: imbalance, dedup, group-based split (leakage), label audit |
| Evaluation-first | §4 baseline (TF-IDF + LogReg), §8 comparison: precision/recall/F1 + confusion matrices |

```
AI_Scam_Hunter.ipynb        the narrative notebook
scam_hunter/
  config.py                 paths, seeds, model + LoRA settings
  data_gen.py               fictional dataset generator (hard negatives, subtle scams, injected duplicates)
  data_quality.py           dedup, group split, leakage report, class weights, label audit
  baseline.py  evaluate.py  TF-IDF+LR baseline, metrics, confusion matrices, threshold tuning
  stress_set.py             40 hand-written out-of-template messages (never used for tuning)
  instruction_data.py       (instruction, input, JSON output) records
  finetune.py               LoRA sequence classifier (Hugging Face Trainer + PEFT)
  finetune_causal.py        optional: LoRA instruction-tuning of a small causal LM
  rag_demo.py               FAISS retrieval: k-NN classifier + scam-pattern guidance
  report.py                 JSON risk report, rule-based evidence, guardrails, schema validation
tests/test_pipeline.py
```

## Design decisions worth explaining in an interview

- **Leakage:** messages are split by *template group*, so the test set contains templates the model never saw; exact/near-duplicate overlap is checked and reported.
- **Two test sets:** template data is easy (the baseline scores ~0.99 F1), so a hand-written **stress set** (new wording, leetspeak, benign messages with scammy words) is the more informative comparison. It is tiny (40 messages), so treat differences as indicative.
- **Imbalance:** report precision/recall/F1 on the *suspicious* class, weight the loss, tune the threshold for recall on validation data only.
- **Guardrails:** no "safe" verdict; strong rule signals can raise a low model score to `uncertain`; mid-range probabilities abstain as `uncertain`.

## Status of results

Numbers on the template test set and stress set for the baseline and the RAG comparison are produced by the notebook. The **LoRA numbers are produced when you run §6 on a machine with the transformer stack and Hugging Face access**; they are not pre-filled here, so no result in this repo is claimed that hasn't been computed.

## Next steps

Larger/real licensed data and a bigger stress set · char n-gram baseline · probability calibration · measure JSON-validity of the instruction-tuned generator · FastAPI `POST /analyze` · small UI.

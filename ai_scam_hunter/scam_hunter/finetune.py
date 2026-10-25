"""Stage 2 -- LoRA/PEFT fine-tuning of a pretrained transformer for classification.

Needs: torch, transformers, datasets, peft, accelerate and a one-off model download from the
Hugging Face Hub. Runs on CPU (a few minutes for ~600 short messages) or GPU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

from .config import (LABELS, LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGETS, MAX_LEN, MODEL_NAME, SEED)
from .data_quality import class_weights


def count_params(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "trainable_pct": round(100 * trainable / total, 3)}


def _to_dataset(df: pd.DataFrame, tok) -> Dataset:
    ds = Dataset.from_dict({"text": df["text"].tolist(), "labels": df["label"].astype(int).tolist()})
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=["text"])


class WeightedTrainer(Trainer):
    """Cross-entropy with class weights to counter class imbalance."""

    def __init__(self, *args, class_weights_tensor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cw = class_weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self._cw.to(outputs.logits.device) if self._cw is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def _metrics(eval_pred):
    logits, labels = eval_pred
    return {"f1": f1_score(labels, logits.argmax(-1), pos_label=1)}


def build_model(use_lora: bool = True, model_name: str = MODEL_NAME, r: int = LORA_R):
    base = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=LABELS, label2id={v: k for k, v in LABELS.items()})
    if not use_lora:
        return base
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                     target_modules=LORA_TARGETS, modules_to_save=["pre_classifier", "classifier"])
    return get_peft_model(base, cfg)


def train_classifier(train: pd.DataFrame, val: pd.DataFrame, output_dir: str, use_lora: bool = True,
                     epochs: int = 4, lr: float = 3e-4, batch_size: int = 16, model_name: str = MODEL_NAME):
    """Fine-tune; returns (trainer, tokenizer, param_counts). Use lr~3e-4 for LoRA, ~3e-5 for full FT."""
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = build_model(use_lora, model_name)
    counts = count_params(model)
    cw = class_weights(train["label"])
    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=32,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1", save_total_limit=1, seed=SEED, report_to="none",
        logging_steps=20, weight_decay=0.01, warmup_ratio=0.1)
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=_to_dataset(train, tok), eval_dataset=_to_dataset(val, tok),
        data_collator=DataCollatorWithPadding(tok), compute_metrics=_metrics,
        class_weights_tensor=torch.tensor([cw[0], cw[1]], dtype=torch.float))
    trainer.train()
    return trainer, tok, counts


@torch.no_grad()
def predict_proba(model, tok, texts, batch_size: int = 32) -> np.ndarray:
    """P(suspicious) for each text."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        out.append(torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def save_adapter(model, tok, path: str) -> None:
    model.save_pretrained(path)  # for LoRA: saves only the small adapter + classifier head
    tok.save_pretrained(path)


def load_adapter(path: str, model_name: str = MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(path)
    base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return PeftModel.from_pretrained(base, path), tok

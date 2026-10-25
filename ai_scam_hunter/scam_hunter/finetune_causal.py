"""Stage 3 (optional) -- instruction-tune a small causal LM with LoRA to emit the JSON report.

GPU strongly recommended. Not required for the core project: the classifier + rule-based
report in `report.py` already produces valid JSON. This stage shows how SFT with prompt
masking works (loss only on the response tokens).
"""
from __future__ import annotations

import json

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from .config import CAUSAL_MODEL_NAME, SEED
from .instruction_data import format_prompt
from .report import validate_report


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, records, tok, max_len: int = 384):
        self.items = []
        for rec in records:
            prompt_ids = tok(format_prompt(rec), add_special_tokens=False)["input_ids"]
            resp_ids = tok(rec["output"] + tok.eos_token, add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + resp_ids)[:max_len]
            labels = ([-100] * len(prompt_ids) + resp_ids)[:max_len]  # mask the prompt
            self.items.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(pad_id: int):
    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids = torch.tensor([b["input_ids"] + [pad_id] * (m - len(b["input_ids"])) for b in batch])
        lab = torch.tensor([b["labels"] + [-100] * (m - len(b["labels"])) for b in batch])
        att = torch.tensor([[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch])
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return collate


def train_instruction_model(train_records, output_dir: str, model_name: str = CAUSAL_MODEL_NAME,
                            epochs: int = 2, lr: float = 2e-4, batch_size: int = 4):
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, cfg)
    args = TrainingArguments(output_dir=output_dir, num_train_epochs=epochs, learning_rate=lr,
                             per_device_train_batch_size=batch_size, logging_steps=20,
                             save_strategy="no", report_to="none", seed=SEED, remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=SFTDataset(train_records, tok),
                      data_collator=make_collator(tok.pad_token_id))
    trainer.train()
    return model, tok


@torch.no_grad()
def generate_report(model, tok, message: str, instruction: str) -> dict | None:
    """Generate a report and validate it. Returns None when output is not valid JSON/schema."""
    prompt = format_prompt({"instruction": instruction, "input": message})
    enc = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    out = model.generate(**enc, max_new_tokens=220, do_sample=False, pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        report = json.loads(text.strip())
        validate_report(report)
        return report
    except (json.JSONDecodeError, ValueError):
        return None

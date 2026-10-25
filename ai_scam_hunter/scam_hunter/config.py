"""Central configuration so every module and the notebook agree on paths and settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RAW_CSV = DATA_DIR / "messages_raw.csv"
CLEAN_CSV = DATA_DIR / "messages_clean.csv"

SEED = 42
LABELS = {0: "legitimate", 1: "suspicious"}
POSITIVE_LABEL = 1  # "suspicious" is the class we care most about catching

# Transformer + LoRA settings (small on purpose: runs on a laptop CPU, faster on GPU)
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGETS = ["q_lin", "v_lin"]  # DistilBERT attention projections

# Optional instruction-tuned generator (stage 3, GPU recommended)
CAUSAL_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Fictional organisations used in the synthetic data
ORG_NAMES = [
    "Northwind Bank", "Contoso Credit Union", "Fabrikam Pay", "Litware Wallet",
    "Tailspin Couriers", "Fabrikam Shop", "Woodgrove Market", "Proseware Store",
    "Adventure Works Mobile", "Litware",
]

for _d in (DATA_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

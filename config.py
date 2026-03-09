import os
import torch
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR      = Path(os.getenv("DATA_DIR", "data"))
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw sources
CSV_COMMENTED  = RAW_DIR / "palettes_commented.csv"
CSV_TAGGED = RAW_DIR / "palettes_tagged.csv"

# Processed splits
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV   = PROCESSED_DIR / "val.csv"
TEST_CSV  = PROCESSED_DIR / "test.csv"

# Cached embeddings (gitignored, computed on first run)
EMB_TRAIN = DATA_DIR / "embeddings_train.npy"
EMB_VAL   = DATA_DIR / "embeddings_val.npy"
EMB_TEST  = DATA_DIR / "embeddings_test.npy"

# ── Model architecture ────────────────────────────────────────────
EMBED_DIM  = 512   # CLIP ViT-B/32 output dimension
HIDDEN_DIM = 512
N_COLORS   = 5
COLOR_DIM  = 3     # Oklab: (L, a, b)

# Flat list of Oklab column names: L0,a0,b0 … L4,a4,b4
OKLAB_COLS = [f"{c}{i}" for i in range(N_COLORS) for c in ("L", "a", "b")]

# ── Training ──────────────────────────────────────────────────────
BATCH_SIZE = 256
EPOCHS     = 200
LR         = 3e-4

# ── Device ────────────────────────────────────────────────────────
DEVICE = (
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)

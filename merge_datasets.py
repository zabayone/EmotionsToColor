"""
Merges the two palette datasets, assigns sample weights, normalises
numeric precision, and splits into train / val / test CSVs.

Weights:
    - colorhex (LLM descriptions): 1.0
    - colorhunt (tag-based):        4.0
"""

import pandas as pd
from pathlib import Path
from config import OKLAB_COLS, PROCESSED_DIR, CSV_COMMENTED, CSV_TAGGED


def build_text_llm(row) -> str:
    desc = str(row.get("description", "")).strip()
    return desc if len(desc) > 5 else ""


def build_text_tags(row) -> str:
    tags = [t.strip() for t in str(row.get("tags", "")).split(",") if t.strip()]
    return "color palette: " + ", ".join(tags) if tags else ""


df_old = pd.read_csv(CSV_COMMENTED)
df_new = pd.read_csv(CSV_TAGGED)

df_old["text"]   = df_old.apply(build_text_llm,  axis=1)
df_old["weight"] = 1.0

df_new["text"]   = df_new.apply(build_text_tags, axis=1)
df_new["weight"] = 4.0

KEEP = OKLAB_COLS + ["text", "weight"]
df_old = df_old[df_old["text"].str.len() > 5][KEEP].copy()
df_new = df_new[df_new["text"].str.len() > 5][KEEP].copy()

print(f"colorhex  : {len(df_old):>6} palettes")
print(f"colorhunt : {len(df_new):>6} palettes")

df = pd.concat([df_old, df_new], ignore_index=True)
df = df.drop_duplicates(subset=OKLAB_COLS)

# Uniform numeric precision across both sources
for col in OKLAB_COLS:
    df[col] = df[col].round(6)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n       = len(df)
n_train = int(n * 0.80)
n_val   = int(n * 0.10)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
df.iloc[:n_train          ].to_csv(PROCESSED_DIR / "train.csv", index=False)
df.iloc[n_train:n_train+n_val].to_csv(PROCESSED_DIR / "val.csv",   index=False)
df.iloc[n_train+n_val:    ].to_csv(PROCESSED_DIR / "test.csv",  index=False)

print(f"Combined  : {n:>6} palettes (deduplicated)")
print(f"Train / Val / Test: {n_train} / {n_val} / {n - n_train - n_val}")

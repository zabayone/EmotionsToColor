import warnings
warnings.filterwarnings("ignore", category=Warning, module="urllib3")

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import open_clip

from config import OKLAB_COLS, N_COLORS, COLOR_DIM, DEVICE

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED  = "openai"
EMBED_BATCH_SIZE = 256


def build_text(row) -> str:
    """Returns the best available text description for a palette row."""
    for col in ("text", "description"):
        val = str(row.get(col, "")).strip()
        if len(val) > 5:
            return val
    return "color palette"


def compute_clip_embeddings(texts: list[str], device: str) -> np.ndarray:
    """Encodes a list of strings with CLIP and returns L2-normalised embeddings."""
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    tokenizer   = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model.eval().to(device)

    batches = []
    n_batches = (len(texts) - 1) // EMBED_BATCH_SIZE + 1
    with torch.no_grad():
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            tokens = tokenizer(texts[i:i + EMBED_BATCH_SIZE]).to(device)
            emb    = model.encode_text(tokens).float()
            emb    = emb / emb.norm(dim=-1, keepdim=True)
            batches.append(emb.cpu().numpy())
            print(f"  Embedding batch {i // EMBED_BATCH_SIZE + 1}/{n_batches}", end="\r")
    print()
    return np.concatenate(batches, axis=0)


class TextPaletteDataset(Dataset):
    """
    Loads a palette CSV and provides (embedding, palette, weight) triplets.

    Embeddings are computed once and cached to disk as .npy files.
    All tensors live in CPU RAM and are moved to DEVICE in __getitem__.
    """

    def __init__(self, csv_path: str | Path, embeddings_path: str | Path):
        self.df = pd.read_csv(csv_path)
        embeddings_path = Path(embeddings_path)
        print(f"Loaded {len(self.df)} palettes from '{csv_path}'")

        if embeddings_path.exists():
            print(f"Loading cached embeddings from '{embeddings_path}'")
            emb = np.load(embeddings_path)
        else:
            print("Computing CLIP embeddings (this runs once)...")
            texts = [build_text(row) for _, row in self.df.iterrows()]
            emb   = compute_clip_embeddings(texts, DEVICE)
            np.save(embeddings_path, emb)
            print(f"Embeddings saved to '{embeddings_path}'")

        self.embeddings = torch.tensor(emb, dtype=torch.float32)

        oklab_flat   = self.df[OKLAB_COLS].values.astype(np.float32)
        self.palettes = torch.tensor(oklab_flat).view(-1, N_COLORS, COLOR_DIM)

        weights = self.df["weight"].fillna(1.0).values if "weight" in self.df.columns \
                  else np.ones(len(self.df))
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.embeddings[idx].to(DEVICE),
            self.palettes[idx].to(DEVICE),
            self.weights[idx].to(DEVICE),
        )

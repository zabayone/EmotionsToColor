# EmotionsToColor

A lightweight neural model that generates 5-colour palettes from natural language descriptions, grounded in the perceptual **Oklab** colour space and evaluated against Russell's circumplex model of affect.

Built as part of a Master's thesis in Acoustic Engineering at Politecnico di Milano.

---

## How it works

```
Text prompt
    │
    ▼
CLIP ViT-B/32 (OpenAI)   →  512-dim embedding
    │
    ▼
Text2PaletteModel         →  5 × Oklab (L, a, b)
    │
    ▼
Emotional anchor blend    →  circumplex-aligned palette
    │
    ▼
enforce_diversity          →  final palette (hex)
```

The model is trained on ~35 000 colour palettes paired with text descriptions (LLM-generated) and tag-based annotations. Tag-based samples receive 4× sample weight during training.

---

## Results

Evaluated on 9 emotion classes from Russell's circumplex model:

| Metric | Raw model | After anchor | Target |
|---|---|---|---|
| Intra-class Consistency | 0.048 | — | < 0.08 |
| Inter-class Discrimination | 0.322 | 0.273 | > 0.15 |
| Circumplex Pearson r | 0.281 | **0.719** | > 0.50 |
| Intra-palette Diversity | 0.352 | 0.176 | > 0.18 |

---

## Repo structure

```
text2palette/
├── data/
│   ├── raw/
│   │   ├── colorhex_palettes_described.csv
│   │   └── colorhunt_palettes.csv
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── config.py
├── dataset.py
├── model.py
├── train.py
├── merge_datasets.py
├── inference.py
├── evaluate.py
├── requirements.txt
└── .gitignore
```

---

## Quickstart

```bash
# 1 — Install dependencies
pip install -r requirements.txt

# 2 — Download the pretrained model
# → Get best_palette_gen.pt from the GitHub Releases page
# → Place it in data/best_palette_gen.pt

# 3 — Run inference
python inference.py
```

To retrain from scratch:
```bash
# Regenerate processed splits (optional, already included)
python merge_datasets.py

# Train (200 epochs, ~24s/epoch on Apple M1 Max)
python train.py

# Evaluate
python evaluate.py
```

---

## Datasets

The processed CSVs are derived from two public colour palette sources and preprocessed into Oklab (L, a, b) coordinates using the `colour-science` library. Raw sources are included in `data/raw/` for full reproducibility.

Embeddings (`.npy`) are not included — they are computed automatically on first run and cached locally.

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Apple MPS, CUDA, or CPU

See `requirements.txt` for the full list.

---

## License

Code: MIT  
Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

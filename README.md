# EmotionsToColor

A lightweight neural model that generates 5-colour palettes from natural language descriptions, grounded in the perceptual **Oklab** colour space and evaluated against Russell's circumplex model of affect.

Built as part of a Master's thesis in Acoustic Engineering at Politecnico di Milano.

---

## Scope

This project is part of my Master's thesis in Acoustic Engineering at Politecnico di Milano.

The goal is to generate a VR environment driven by a **Music Emotion Recognition (MER)** model 
that analyses piano performances using audio signals, motion tracking, and EMG data.
The MER model classifies performances into 9 emotional states based on Russell's 
circumplex model of affect:

| Class | Description |
|---|---|
| Neutral | Emotional balance, absence of strong feelings |
| Alert and excited | Heightened awareness, anticipation, energized |
| Elated and happy | Intense joy and satisfaction, sense of fulfillment |
| Contented and serene | Peaceful well-being, no immediate desires |
| Relaxed and calm | Physical and mental ease, free from tension |
| Melancholic and bored | Reflective sadness, lack of stimulation |
| Sad and depressed | Deep unhappiness or despair |
| Stressed and upset | Mental strain, frustration or discomfort |
| Nervous and tense | Apprehension and unease, physical tension |

The first stage of environment generation is **colour palette assignment**.
This model generates perceptually coherent 5-colour palettes from emotion class labels
and streams them to Unity via the **OSC protocol**.

**Why Oklab?**  
Oklab is a perceptually uniform colour space — equal numerical distances correspond to 
equal perceived colour differences. This guarantees smooth, artefact-free transitions 
between emotional scenes and consistent rendering across devices.


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

## References & Credits

**Colour space**
- Björn Ottosson — [Oklab perceptual colour space](https://bottosson.github.io/posts/oklab/)
- [`colour-science`](https://www.colour-science.org/) — Oklab conversion library

**Embeddings**
- OpenAI / LAION — [CLIP ViT-B/32](https://github.com/mlfoundations/open_clip)

**Emotion model**
- Russell, J. A. (1980). *A circumplex model of affect*. Journal of Personality and 
  Social Psychology, 39(6), 1161–1178.

**Datasets**
- Colour palette data sourced from public online repositories,
  preprocessed into Oklab coordinates and paired with text annotations.

  **Related work**
- Bahng et al. (2018). *Coloring with Words: Guiding Image Colorization 
  Through Text-based Palette Generation.*
  [Text2Colors](https://github.com/awesome-davian/Text2Colors) — ECCV 2018.

## License

Code: MIT  
Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

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
enforce_diversity          →  spread colours apart
    │
    ▼
sort_palette_by_luminance  →  darkest → lightest (Unity-ready)
    │
    ▼
oklab_to_hex               →  final palette (#rrggbb × 5)
```

The model is trained on ~35 000 colour palettes paired with text descriptions (LLM-generated) and tag-based annotations. Tag-based samples receive 4× sample weight during training.

---

## Model & Training

### Architecture

Text2Palette uses a **two-stage architecture**:

1. **CLIP ViT-B/32** (frozen) — encodes the text prompt into a 512-dimensional embedding.
   CLIP was chosen over alternatives (e.g. Sentence-BERT) because it was trained on
   image-text pairs, giving it a strong prior on the visual semantics of colour-related
   language (*"warm sunset"*, *"cold winter"*, *"neon night"*).

2. **Text2PaletteModel** — a lightweight MLP that maps the CLIP embedding to 5 colours
   in Oklab space. The architecture consists of:
   - A shared encoder (2 × Linear → GELU → LayerNorm → Dropout 0.1)
   - Five independent colour heads (one per output colour), each predicting (L, a, b)

   Output activations enforce valid Oklab ranges:
   - L ∈ [0, 1] via **Sigmoid**
   - a, b ∈ [−0.5, 0.5] via **Tanh × 0.5**

The model has ~800K parameters and runs in < 5ms per inference on CPU.

---

### Training

The model is trained end-to-end with a **composite loss** of three terms:

```
L = L_huber + 0.3 · L_triplet + 1.0 · L_diversity + 0.5 · L_spread
```

| Term | Role |
|---|---|
| **Huber** | Reconstruction — predicted palette close to ground truth |
| **Triplet** | Ranking — palette of prompt A closer to its target than to prompt B's target |
| **Diversity penalty** | Prevents colour collapse — pushes colours apart if closer than 0.12 |
| **Lightness spread** | Forces L values to span [0.1, 0.9] evenly across the 5 colours |

**Why Huber instead of MSE?**  
Huber is less sensitive to outliers in colour space — some training palettes contain
very dark or very saturated colours that would dominate a squared loss.

**Why Triplet loss?**  
It teaches the model *relative* semantics: the palette for *"happy"* should be closer
to other happy palettes than to sad ones. This improves generalisation to unseen prompts
without requiring explicit emotion labels during training.

**Sample weighting:** tag-based palettes (ColorHunt) receive 4× weight relative to
LLM-described palettes (ColorHex), reflecting higher confidence in their semantic labels.

Training runs for 200 epochs with AdamW (lr=3e-4, cosine annealing to 1e-5)
on ~28 000 training samples. On Apple M1 Max: ~24s/epoch.

---

### Emotional Anchor Blend

The model is trained on generic colour-text pairs and has no explicit emotion supervision.
To align output palettes with the 9 target emotion classes, a **post-hoc anchor blend**
is applied at inference time:

```
palette_final = (1 − α) · palette_model + α · anchor_class
```
Anchor palettes are hand-crafted Oklab targets grounded in colour psychology and
Russell's circumplex model of affect. The blend weight α is tuned per class (0.30–0.65)
based on the measured anchor gap between model output and theoretical target.

This raises the Circumplex Pearson correlation from **0.28 → 0.72** without any
additional training.

---

### Palette Ordering

After diversity enforcement the 5 colours are sorted by Oklab **L** (lightness) in
ascending order — darkest to lightest:

| Index | Role (Unity convention) |
|-------|------------------------|
| 0 | Darkest — shadows, backgrounds |
| 1 | Dark mid-tone |
| 2 | Mid-tone |
| 3 | Light mid-tone |
| 4 | Lightest — highlights, foreground accents |

This deterministic ordering lets Unity shaders and scripts address palette slots by
perceived brightness without needing to sort at runtime.

### Inference Temperature

The inference pipeline adds Gaussian noise to the CLIP embedding before each forward
pass so that repeated calls with the same prompt return perceptibly different palettes:

```python
T_SCALED = 0.25 / (EMBED_DIM ** 0.5)   # ≈ 0.011  →  ~14° angular deviation
emb = emb + torch.randn_like(emb) * T_SCALED
emb = emb / emb.norm(dim=-1, keepdim=True)
```

Pass `temperature=0.0` to `generate()` to get a deterministic output (used internally
by `evaluate.py`).

---

## Results

Evaluated on 9 emotion classes from Russell's circumplex model of affect.
Full pipeline: CLIP embedding → Text2PaletteModel → emotional anchor blend → enforce_diversity → sort_palette_by_luminance.

| Metric | Raw model | After anchor | Target |
|---|---|---|---|
| Intra-class Consistency | 0.051 | — | < 0.08 |
| Inter-class Discrimination | 0.322 | 0.273 | > 0.15 |
| Circumplex Pearson r | 0.281 | **0.719** | > 0.50 |
| Intra-palette Diversity | 0.352 | 0.176 ¹ | > 0.18 |

> **Note — consistency metric uses `T = 0.05 / √512 ≈ 0.002`** (evaluation temperature).
> Interactive inference uses the higher default `T = 0.25 / √512 ≈ 0.011`, which gives
> perceptibly different palettes across runs (~14° angular deviation on the embedding sphere).

¹ Diversity is partially recovered by `enforce_diversity(min_dist=0.15)` applied as
  the final step in the pipeline. The reduction post-anchor is expected and semantically
  correct — emotionally coherent classes (e.g. *sad and depressed*) naturally produce
  less contrasted palettes.
  
---

  The following image is an example generated on the 9 emotional classes taken into consideration:
  
<img width="1297" height="1465" alt="examples" src="https://github.com/user-attachments/assets/b39d23ff-b87a-4c50-b2c4-d9eaa2cb728f" />

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

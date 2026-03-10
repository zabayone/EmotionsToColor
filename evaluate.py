"""
Objective evaluation of Text2PaletteModel on the 9 target emotion classes.

Metrics
-------
1. Intra-class Consistency   — same class → similar palettes (low variance)
2. Inter-class Discrimination — different classes → distant palettes
3. Circumplex Ordering        — Oklab distances correlate with Russell valence/arousal
4. Intra-palette Diversity    — the 5 colours within each palette are visually distinct
5. Anchor Alignment           — distance between model output and theoretical anchor
6. Post-anchor Evaluation     — metrics after the emotion-anchor blend is applied
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import open_clip
from itertools import combinations

from model import Text2PaletteModel
from config import DEVICE, DATA_DIR, EMBED_DIM, ANCHOR_SCALE
from inference import enrich_prompt

# ── Emotion classes ───────────────────────────────────────────────
# Prompts are enriched with synonyms to reduce CLIP embedding variance.
CLASSES: dict[str, str] = {
    "neutral":               "neutral balanced",
    "alert and excited":     "alert excited energized anticipation",
    "elated and happy":      "elated happy joyful fulfilled",
    "contented and serene":  "contented serene peaceful satisfied",
    "relaxed and calm":      "relaxed calm ease tranquil",
    "melancholic and bored": "melancholic bored pensive reflective sad",
    "sad and depressed":     "sad depressed hopeless despair",
    "stressed and upset":    "stressed upset frustrated strain",
    "nervous and tense":     "nervous tense apprehensive uneasy",
}

# Russell's circumplex coordinates: (valence, arousal) in [-1, 1]
CIRCUMPLEX: dict[str, tuple[float, float]] = {
    "neutral":               ( 0.0,  0.0),
    "alert and excited":     ( 0.6,  0.9),
    "elated and happy":      ( 1.0,  0.7),
    "contented and serene":  ( 0.9,  0.1),
    "relaxed and calm":      ( 0.7, -0.5),
    "melancholic and bored": (-0.5, -0.6),
    "sad and depressed":     (-1.0, -0.4),
    "stressed and upset":    (-0.7,  0.6),
    "nervous and tense":     (-0.4,  0.8),
}

# Theoretical Oklab anchors — (5, 3) array per class
EMOTIONAL_ANCHORS: dict[str, np.ndarray] = {
    "neutral": np.array([
        ( 0.5139, -0.0055, -0.0214), ( 0.5783, -0.0068, -0.0196),
        ( 0.6715, -0.0073, -0.0207), ( 0.7552, -0.0075, -0.0187), ( 0.8467, -0.0066, -0.0165),
    ], dtype=np.float32),
    "alert and excited": np.array([
        ( 0.6839,  0.1718,  0.1152), ( 0.7236,  0.1228,  0.1407),
        ( 0.8124,  0.0401,  0.1655), ( 0.6541,  0.2035,  0.1120), ( 0.8803,  0.0092,  0.1344),
    ], dtype=np.float32),
    "elated and happy": np.array([
        ( 0.8916, -0.0082,  0.1566), ( 0.8403,  0.0178,  0.1715),
        ( 0.9240, -0.0079,  0.1143), ( 0.7828,  0.0678,  0.1537), ( 0.8851,  0.0089,  0.1287),
    ], dtype=np.float32),
    "contented and serene": np.array([
        ( 0.8082, -0.0864,  0.0018), ( 0.8719, -0.0628, -0.0025),
        ( 0.9163, -0.0382, -0.0241), ( 0.8005, -0.0465, -0.0620), ( 0.7427, -0.0551, -0.0681),
    ], dtype=np.float32),
    "relaxed and calm": np.array([
        ( 0.68, -0.04, -0.08), ( 0.74, -0.03, -0.06),
        ( 0.62, -0.05, -0.10), ( 0.78, -0.02, -0.05), ( 0.65, -0.04, -0.09),
    ], dtype=np.float32),
    "melancholic and bored": np.array([
        ( 0.48, -0.03, -0.10), ( 0.42, -0.02, -0.08),
        ( 0.55, -0.04, -0.06), ( 0.38, -0.01, -0.12), ( 0.52, -0.03, -0.07),
    ], dtype=np.float32),
    "sad and depressed": np.array([
        ( 0.28, -0.04, -0.16), ( 0.22, -0.03, -0.14),
        ( 0.35, -0.05, -0.12), ( 0.18, -0.02, -0.18), ( 0.32, -0.04, -0.15),
    ], dtype=np.float32),
    "stressed and upset": np.array([
        ( 0.4225,  0.1346,  0.0673), ( 0.5386,  0.1731,  0.0871),
        ( 0.6083,  0.1862,  0.0949), ( 0.7045,  0.1492,  0.1217), ( 0.3141,  0.1009,  0.0435),
    ], dtype=np.float32),
    "nervous and tense": np.array([
        ( 0.3763,  0.0379, -0.1477), ( 0.4853,  0.0435, -0.1511),
        ( 0.5544, -0.0151, -0.1751), ( 0.6618,  0.0483, -0.1815), ( 0.2923,  0.0080, -0.0987),
    ], dtype=np.float32),
}

# Alpha values per class (tuned from anchor-gap analysis)
ANCHOR_ALPHA: dict[str, float] = {
    "neutral":               0.25,
    "alert and excited":     0.50,
    "elated and happy":      0.45,
    "contented and serene":  0.40,
    "relaxed and calm":      0.35,
    "melancholic and bored": 0.45,
    "sad and depressed":     0.60,
    "stressed and upset":    0.45,
    "nervous and tense":     0.45,
}

# Global scaling factor applied to all per-class anchor blend weights.
# Imported from config — single source of truth.

T_SCALED = 0.05 / (EMBED_DIM ** 0.5)
N_RUNS   = 10

# ── Model setup ───────────────────────────────────────────────────
clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer        = open_clip.get_tokenizer("ViT-B-32")
clip_model.eval().to(DEVICE)

model = Text2PaletteModel().to(DEVICE)
model.load_state_dict(torch.load(DATA_DIR / "best_palette_gen.pt", map_location=DEVICE))
model.eval()


# ── Helpers ───────────────────────────────────────────────────────

COLOR_ENRICHMENT_WEIGHT: float = 0.25  # must match inference.py default


def generate(prompt: str, temperature: float = 0.0) -> np.ndarray:
    with torch.no_grad():
        tokens_plain = tokenizer([prompt]).to(DEVICE)
        emb = clip_model.encode_text(tokens_plain).float()
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)

        if COLOR_ENRICHMENT_WEIGHT > 0:
            enriched = enrich_prompt(prompt)
            if enriched != prompt:
                tokens_rich = tokenizer([enriched]).to(DEVICE)
                emb_rich = clip_model.encode_text(tokens_rich).float()
                emb_rich = emb_rich / (emb_rich.norm(dim=-1, keepdim=True) + 1e-8)
                emb = (1 - COLOR_ENRICHMENT_WEIGHT) * emb + COLOR_ENRICHMENT_WEIGHT * emb_rich
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)

        if temperature > 0:
            emb = emb + torch.randn_like(emb) * temperature
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
    return model(emb).squeeze(0).cpu().detach().numpy()


def palette_dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2, axis=1).mean())


def circumplex_dist(c1: tuple, c2: tuple) -> float:
    return float(np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2))


def apply_anchor(palette: np.ndarray, cls: str) -> np.ndarray:
    alpha = ANCHOR_ALPHA.get(cls, 0.55)
    effective_alpha = ANCHOR_SCALE * alpha
    return (1 - effective_alpha) * palette + effective_alpha * EMOTIONAL_ANCHORS[cls]


# ── Pre-compute base palettes ─────────────────────────────────────
print("Generating base palettes for all 9 classes...\n")
class_names   = list(CLASSES.keys())
BASE_PALETTES = {cls: generate(CLASSES[cls]) for cls in class_names}


# ── 1. Intra-class Consistency ────────────────────────────────────
print("=" * 60)
print(f"1. INTRA-CLASS CONSISTENCY  (T_scaled={T_SCALED:.4f}, N={N_RUNS})")
print("   Target: < 0.08  |  Noisy: > 0.20")

consistency: dict[str, float] = {}
for cls, prompt in CLASSES.items():
    runs  = [generate(prompt, temperature=T_SCALED) for _ in range(N_RUNS)]
    score = np.mean([palette_dist(runs[i], runs[j])
                     for i, j in combinations(range(N_RUNS), 2)])
    consistency[cls] = score
    flag = "✅" if score < 0.08 else ("⚠️" if score < 0.15 else "❌ noisy")
    print(f"  {cls:30s} {score:.4f}  {flag}")
print(f"\n  Mean: {np.mean(list(consistency.values())):.4f}")


# ── 2. Inter-class Discrimination ────────────────────────────────
print("\n" + "=" * 60)
print("2. INTER-CLASS DISCRIMINATION  (temperature=0)")
print("   Target: > 0.15  |  Poor: < 0.08")

disc_scores, worst_pairs = [], []
for c1, c2 in combinations(class_names, 2):
    d = palette_dist(BASE_PALETTES[c1], BASE_PALETTES[c2])
    disc_scores.append(d)
    if d < 0.10:
        worst_pairs.append((c1, c2, d))

print(f"  Mean inter-class distance : {np.mean(disc_scores):.4f}")
print(f"  Min                       : {np.min(disc_scores):.4f}")
print(f"  Max                       : {np.max(disc_scores):.4f}")

if worst_pairs:
    print("\n  ⚠️  Pairs below threshold (< 0.10):")
    for c1, c2, d in worst_pairs:
        print(f"     {c1:25s} ↔ {c2:25s}: {d:.4f}")
else:
    print("\n  ✅ All pairs well separated")

print("\n  Inter-class distance matrix (Oklab):")
header = "".join(f"{n[:6]:>8}" for n in class_names)
print(f"  {'':30s}{header}")
for c1 in class_names:
    row = f"  {c1:30s}"
    for c2 in class_names:
        row += f"{'—':>8}" if c1 == c2 else f"{palette_dist(BASE_PALETTES[c1], BASE_PALETTES[c2]):>8.3f}"
    print(row)


# ── 3. Circumplex Ordering ────────────────────────────────────────
print("\n" + "=" * 60)
print("3. CIRCUMPLEX ORDERING")
print("   Do Oklab distances correlate with Russell valence/arousal distances?")
print("   Target: Pearson r > 0.5")

oklab_dists = [palette_dist(BASE_PALETTES[c1], BASE_PALETTES[c2])
               for c1, c2 in combinations(class_names, 2)]
circ_dists  = [circumplex_dist(CIRCUMPLEX[c1], CIRCUMPLEX[c2])
               for c1, c2 in combinations(class_names, 2)]

corr = float(np.corrcoef(oklab_dists, circ_dists)[0, 1])
flag = "✅" if corr > 0.5 else ("⚠️" if corr > 0.3 else "❌")
print(f"\n  Pearson r (raw model): {corr:.4f}  {flag}")

print("\n  Top 3 most distant pairs (Oklab):")
for d, (c1, c2) in sorted(zip(oklab_dists, combinations(class_names, 2)), key=lambda x: -x[0])[:3]:
    print(f"    {c1:25s} ↔ {c2:20s}  Oklab={d:.3f}  Circ={circumplex_dist(CIRCUMPLEX[c1], CIRCUMPLEX[c2]):.3f}")


# ── 4. Intra-palette Diversity ────────────────────────────────────
print("\n" + "=" * 60)
print("4. INTRA-PALETTE DIVERSITY  (temperature=0)")
print("   Target: > 0.18  |  Collapse: < 0.08")

diversity_scores: dict[str, float] = {}
for cls in class_names:
    pal   = BASE_PALETTES[cls]
    score = np.mean([np.linalg.norm(pal[i] - pal[j]) for i, j in combinations(range(5), 2)])
    diversity_scores[cls] = score
    flag = "✅" if score > 0.18 else ("⚠️" if score > 0.10 else "❌ collapse")
    print(f"  {cls:30s} {score:.4f}  {flag}")
print(f"\n  Mean: {np.mean(list(diversity_scores.values())):.4f}")


# ── 5. Anchor Alignment ───────────────────────────────────────────
print("\n" + "=" * 60)
print("5. ANCHOR ALIGNMENT  (raw model vs theoretical anchor)")
print("   Low  → model already close, anchor optional")
print("   High → anchor critical to correct emotional semantics")

anchor_gaps: dict[str, float] = {}
for cls in class_names:
    gap = palette_dist(BASE_PALETTES[cls], EMOTIONAL_ANCHORS[cls])
    anchor_gaps[cls] = gap
    flag = "✅ close" if gap < 0.15 else ("⚠️ medium" if gap < 0.30 else "❌ far → anchor critical")
    print(f"  {cls:30s} gap={gap:.4f}  {flag}")
print(f"\n  Mean gap: {np.mean(list(anchor_gaps.values())):.4f}")


# ── 6. Post-anchor Evaluation ─────────────────────────────────────
print("\n" + "=" * 60)
print("6. POST-ANCHOR EVALUATION  (before → after anchor blend)")

ANCHORED_PALETTES = {cls: apply_anchor(BASE_PALETTES[cls], cls) for cls in class_names}

# 6a — Gap reduction
print("\n  6a. Anchor gap reduction:")
for cls in class_names:
    before = anchor_gaps[cls]
    after  = palette_dist(ANCHORED_PALETTES[cls], EMOTIONAL_ANCHORS[cls])
    print(f"  {cls:30s} {before:.4f} → {after:.4f}  (↓{before - after:.4f})")

# 6b — Discrimination
anchored_disc  = [palette_dist(ANCHORED_PALETTES[c1], ANCHORED_PALETTES[c2])
                  for c1, c2 in combinations(class_names, 2)]
anchored_worst = [(c1, c2, d) for d, (c1, c2)
                  in zip(anchored_disc, combinations(class_names, 2)) if d < 0.10]
print(f"\n  6b. Discrimination  before={np.mean(disc_scores):.4f}  after={np.mean(anchored_disc):.4f}")
if anchored_worst:
    for c1, c2, d in anchored_worst:
        print(f"      ⚠️  {c1} ↔ {c2}: {d:.4f}")
else:
    print("      ✅ All pairs still well separated")

# 6c — Circumplex correlation
corr_anchored = float(np.corrcoef(anchored_disc, circ_dists)[0, 1])
flag = "✅" if corr_anchored > 0.5 else ("⚠️" if corr_anchored > 0.3 else "❌")
print(f"\n  6c. Circumplex r  before={corr:.4f}  after={corr_anchored:.4f}  {flag}"
      f"  (Δ={corr_anchored - corr:+.4f})")

# 6d — Diversity
anchored_div = [np.mean([np.linalg.norm(ANCHORED_PALETTES[cls][i] - ANCHORED_PALETTES[cls][j])
                         for i, j in combinations(range(5), 2)])
                for cls in class_names]
print(f"\n  6d. Diversity  before={np.mean(list(diversity_scores.values())):.4f}"
      f"  after={np.mean(anchored_div):.4f}")


# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY  (raw model  →  after anchor)")
print(f"  Consistency    (target < 0.08):  {np.mean(list(consistency.values())):.4f}  (unchanged)")
print(f"  Discrimination (target > 0.15):  {np.mean(disc_scores):.4f}  →  {np.mean(anchored_disc):.4f}")
print(f"  Circumplex r   (target > 0.50):  {corr:.4f}  →  {corr_anchored:.4f}")
print(f"  Diversity      (target > 0.18):  {np.mean(list(diversity_scores.values())):.4f}  →  {np.mean(anchored_div):.4f}")

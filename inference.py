"""
inference.py — Text-to-palette generation pipeline.

Pipeline:
    1. CLIP encodes the text prompt into a 512-dim embedding
    2. Text2PaletteModel predicts a raw 5-colour palette in Oklab space
    3. Emotional anchor blends the palette toward a class-specific target
       (based on Russell's circumplex model of affect)
    4. enforce_diversity pushes colours apart to guarantee visual variety
    5. sort_palette_by_luminance orders colours dark → light for Unity indexing
    6. Oklab → sRGB → hex conversion for display
"""

import warnings
warnings.filterwarnings("ignore", category=Warning, module="urllib3")

import torch
import numpy as np
import open_clip

from model import Text2PaletteModel
from config import DEVICE, DATA_DIR, EMBED_DIM, ANCHOR_SCALE

# ── Constants ─────────────────────────────────────────────────────
# Temperature for embedding-space noise.  Scaled to embedding dimension so
# that the expected angular deviation on the unit sphere stays constant
# regardless of model size.  0.25 / √512 ≈ 0.011 → ~14° rotation, which
# gives perceptibly different palettes across runs while remaining
# semantically coherent (same emotion class).
T_SCALED = 0.25 / (EMBED_DIM ** 0.5)   # temperature scaled to embedding norm


# ── Emotional anchors (Russell circumplex) ────────────────────────
# Each anchor is a (5, 3) array of target Oklab colours for that emotion class.
# Blend weight (alpha) is tuned per class based on model anchor-gap analysis.
# ANCHOR_SCALE (imported from config) is applied globally to all weights.

EMOTIONAL_ANCHORS: dict[str, tuple[np.ndarray, float]] = {
    "neutral": (np.array([
        (0.5139, -0.0055, -0.0214), (0.5783, -0.0068, -0.0196),
        (0.6715, -0.0073, -0.0207), (0.7552, -0.0075, -0.0187), (0.8467, -0.0066, -0.0165),
    ], dtype=np.float32), 0.25),

    "alert and excited": (np.array([
        (0.6839,  0.1718,  0.1152), (0.7236,  0.1228,  0.1407),
        (0.8124,  0.0401,  0.1655), (0.6541,  0.2035,  0.1120), (0.8803,  0.0092,  0.1344),
    ], dtype=np.float32), 0.50),

    "elated and happy": (np.array([
        (0.8916, -0.0082,  0.1566), (0.8403,  0.0178,  0.1715),
        (0.9240, -0.0079,  0.1143), (0.7828,  0.0678,  0.1537), (0.8851,  0.0089,  0.1287),
    ], dtype=np.float32), 0.45),

    "contented and serene": (np.array([
        (0.8082, -0.0864,  0.0018), (0.8719, -0.0628, -0.0025),
        (0.9163, -0.0382, -0.0241), (0.8005, -0.0465, -0.0620), (0.7427, -0.0551, -0.0681),
    ], dtype=np.float32), 0.40),

    "relaxed and calm": (np.array([
        (0.68, -0.04, -0.08), (0.74, -0.03, -0.06),
        (0.62, -0.05, -0.10), (0.78, -0.02, -0.05), (0.65, -0.04, -0.09),
    ], dtype=np.float32), 0.35),

    "melancholic and bored": (np.array([
        (0.48, -0.03, -0.10), (0.42, -0.02, -0.08),
        (0.55, -0.04, -0.06), (0.38, -0.01, -0.12), (0.52, -0.03, -0.07),
    ], dtype=np.float32), 0.45),

    "sad and depressed": (np.array([
        (0.28, -0.04, -0.16), (0.22, -0.03, -0.14),
        (0.35, -0.05, -0.12), (0.18, -0.02, -0.18), (0.32, -0.04, -0.15),
    ], dtype=np.float32), 0.60),

    "stressed and upset": (np.array([
        (0.4225,  0.1346,  0.0673), (0.5386,  0.1731,  0.0871),
        (0.6083,  0.1862,  0.0949), (0.7045,  0.1492,  0.1217), (0.3141,  0.1009,  0.0435),
    ], dtype=np.float32), 0.45),

    "nervous and tense": (np.array([
        (0.3763,  0.0379, -0.1477), (0.4853,  0.0435, -0.1511),
        (0.5544, -0.0151, -0.1751), (0.6618,  0.0483, -0.1815), (0.2923,  0.0080, -0.0987),
    ], dtype=np.float32), 0.45),
}

# ── Prompt colour enrichment ──────────────────────────────────────
# Natural-language colour descriptors appended to the CLIP input to steer the
# embedding toward the correct chromatic zone *before* model inference, rather
# than relying solely on post-hoc anchor blending.
# The original prompt label is kept for anchor-key matching; only the CLIP
# encoding sees the extended version.

PROMPT_COLOR_DESCRIPTIONS: dict[str, str] = {
    "neutral":                "muted gray-blue, cool balanced tones, neutral palette",
    "alert and excited":      "vivid orange-red, warm energetic hues, saturated amber",
    "elated and happy":       "bright sunny yellow, cheerful golden light, warm luminous",
    "contented and serene":   "soft aqua seafoam, calm sky blue, fresh cool teal",
    "relaxed and calm":       "gentle cool blue, soothing soft tones, quiet periwinkle",
    "melancholic and bored":  "dull muted violet, pale gray-lavender, desaturated cool",
    "sad and depressed":      "dark navy blue, deep cold indigo, heavy dark shadows",
    "stressed and upset":     "dark crimson red, intense deep red, tense saturated",
    "nervous and tense":      "cold electric violet-blue, deep indigo, tense dark",
}


def enrich_prompt(prompt: str) -> str:
    """
    Returns a colour-enriched version of the prompt for CLIP encoding.
    Matches *prompt* against EMOTIONAL_ANCHORS keys by shared-word overlap
    (same strategy as apply_anchor).  If a match is found the corresponding
    colour descriptor from PROMPT_COLOR_DESCRIPTIONS is appended.
    Falls back to the original prompt for free-form inputs.
    """
    prompt_words = set(prompt.lower().split())
    best_key, best_score = None, 0
    for key in EMOTIONAL_ANCHORS:
        score = len(set(key.split()) & prompt_words)
        if score > best_score:
            best_score, best_key = score, key
    if best_key is None or best_score == 0:
        return prompt
    descriptor = PROMPT_COLOR_DESCRIPTIONS.get(best_key, "")
    return f"{prompt}, {descriptor}" if descriptor else prompt


# ── Post-processing ───────────────────────────────────────────────

def apply_anchor(palette: np.ndarray, prompt: str) -> np.ndarray:
    """
    Blends the model output toward the best-matching emotional anchor.
    Matching is done by counting shared words between prompt and anchor key.
    Returns the palette unchanged if no anchor matches.
    """
    prompt_words = set(prompt.lower().split())
    best_key, best_score = None, 0
    for key in EMOTIONAL_ANCHORS:
        score = len(set(key.split()) & prompt_words)
        if score > best_score:
            best_score, best_key = score, key
    if best_key is None:
        return palette
    anchor, alpha = EMOTIONAL_ANCHORS[best_key]
    effective_alpha = ANCHOR_SCALE * alpha
    return (1 - effective_alpha) * palette + effective_alpha * anchor


def enforce_diversity(palette: np.ndarray, min_dist: float = 0.15) -> np.ndarray:
    """
    Iterative repulsion: pushes colour pairs closer than min_dist apart.
    Preserves the palette centroid (emotion is not altered, only spread).
    """
    p = palette.copy()
    for _ in range(10):
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                diff = p[j] - p[i]
                d    = np.linalg.norm(diff)
                if 1e-6 < d < min_dist:
                    push   = diff / d * (min_dist - d) * 0.5
                    p[i]  -= push
                    p[j]  += push
    p[:, 0] = np.clip(p[:, 0],  0.05, 0.95)  # L
    p[:, 1] = np.clip(p[:, 1], -0.40, 0.40)  # a
    p[:, 2] = np.clip(p[:, 2], -0.40, 0.40)  # b
    return p


def sort_palette_by_luminance(palette: np.ndarray) -> np.ndarray:
    """
    Sorts the 5 colours by Oklab L channel, darkest → lightest.

    Consistent ordering lets downstream consumers (e.g. Unity shaders) address
    palette slots by perceived brightness:
        index 0 = darkest  (shadows / background)
        index 4 = lightest (highlights / foreground)
    """
    return palette[np.argsort(palette[:, 0])]


def oklab_to_hex(L: float, a: float, b: float) -> str:
    """Converts a single Oklab colour to a lowercase hex string."""
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r  =  4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g  = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    b_ = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_

    def gamma(c: float) -> float:
        c = np.clip(c, 0, 1)
        return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)

    return f"#{int(gamma(r) * 255):02x}{int(gamma(g) * 255):02x}{int(gamma(b_) * 255):02x}"


def print_palette(hexes: list[str], title: str) -> None:
    """Renders a colour palette as terminal colour blocks with hex codes."""
    blocks = "".join(
        f"\033[48;2;{int(h[1:3], 16)};{int(h[3:5], 16)};{int(h[5:7], 16)}m   \033[0m"
        for h in hexes
    )
    print(f"\n🎨 {title}")
    print(blocks)
    print("  " + "  ".join(hexes) + "\n")


# ── Inference pipeline ────────────────────────────────────────────

def load_models():
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer        = open_clip.get_tokenizer("ViT-B-32")
    clip_model.eval().to(DEVICE)

    model = Text2PaletteModel().to(DEVICE)
    model.load_state_dict(torch.load(DATA_DIR / "best_palette_gen.pt", map_location=DEVICE))
    model.eval()
    return model, clip_model, tokenizer


def generate(prompt: str, model, clip_model, tokenizer,
             temperature: float = T_SCALED,
             color_enrichment_weight: float = 0.25) -> list[str]:
    """Full pipeline: CLIP → model → anchor → diversity → hex.

    Args:
        prompt: Emotion description (e.g. "sad and depressed").
        temperature: Noise added to the CLIP embedding for palette variety.
        color_enrichment_weight: Interpolation weight in [0, 1] between the
            plain embedding (0) and the colour-enriched embedding (1).
            At 0 the colour hints are ignored entirely; at 1 the enriched
            prompt fully replaces the original.  Default 0.25 gives a subtle
            chromatic nudge while preserving emotion semantics.
    """
    with torch.no_grad():
        tokens_plain = tokenizer([prompt]).to(DEVICE)
        emb_plain = clip_model.encode_text(tokens_plain).float()
        emb_plain = emb_plain / (emb_plain.norm(dim=-1, keepdim=True) + 1e-8)

        if color_enrichment_weight > 0:
            enriched = enrich_prompt(prompt)
            if enriched != prompt:
                tokens_rich = tokenizer([enriched]).to(DEVICE)
                emb_rich = clip_model.encode_text(tokens_rich).float()
                emb_rich = emb_rich / (emb_rich.norm(dim=-1, keepdim=True) + 1e-8)
                emb = (1 - color_enrichment_weight) * emb_plain + color_enrichment_weight * emb_rich
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
            else:
                emb = emb_plain
        else:
            emb = emb_plain

        if temperature > 0:
            emb = emb + torch.randn_like(emb) * temperature
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        palette = model(emb).squeeze(0).cpu().numpy()

    palette = apply_anchor(palette, prompt)
    palette = enforce_diversity(palette)
    palette = sort_palette_by_luminance(palette)
    return [oklab_to_hex(*color) for color in palette]


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    _model, _clip, _tok = load_models()
    print(f"Model loaded on {DEVICE}. Type 'q' to quit.\n")

    while True:
        try:
            prompt = input("Description: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye! 👋")
            break
        if not prompt or prompt.lower() in ("q", "quit", "exit"):
            break
        hexes = generate(prompt, _model, _clip, _tok)
        print_palette(hexes, prompt)

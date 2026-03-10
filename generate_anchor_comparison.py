"""Generate a side-by-side comparison image for raw, anchor, and final palettes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from inference import (
    ANCHOR_SCALE,
    EMOTIONAL_ANCHORS,
    apply_anchor,
    enforce_diversity,
    load_models,
    oklab_to_hex,
    sort_palette_by_luminance,
)


CLASSES: list[str] = [
    "neutral",
    "alert and excited",
    "elated and happy",
    "contented and serene",
    "relaxed and calm",
    "melancholic and bored",
    "sad and depressed",
    "stressed and upset",
    "nervous and tense",
]


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def raw_palette(prompt: str, model, clip_model, tokenizer) -> np.ndarray:
    tokens = tokenizer([prompt]).to(next(model.parameters()).device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens).float()
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        palette = model(emb).squeeze(0).cpu().numpy()
    return palette


def to_hexes(palette: np.ndarray) -> list[str]:
    ordered = sort_palette_by_luminance(palette)
    return [oklab_to_hex(*color) for color in ordered]


def build_image(comparison: dict[str, dict[str, list[str]]], output_path: Path) -> None:
    background = (247, 245, 240)
    title_color = (25, 29, 34)
    text_color = (60, 66, 74)
    line_color = (214, 208, 198)
    panel_fill = (255, 255, 255)

    margin_x = 28
    margin_top = 26
    margin_bottom = 20
    label_width = 180
    mode_width = 124
    swatch_size = 42
    swatch_gap = 6
    block_gap = 18
    row_gap = 14
    title_height = 48

    modes = ["model", "anchor", "final"]
    palette_block_width = mode_width + 5 * swatch_size + 4 * swatch_gap
    row_width = label_width + len(modes) * palette_block_width + (len(modes) - 1) * block_gap
    image_width = margin_x * 2 + row_width
    image_height = margin_top + title_height + len(CLASSES) * swatch_size + (len(CLASSES) - 1) * row_gap + margin_bottom + 24

    image = Image.new("RGB", (image_width, image_height), background)
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()

    draw.text((margin_x, margin_top), "Palette comparison: model vs anchor vs final", fill=title_color, font=title_font)
    draw.text(
        (margin_x, margin_top + 16),
        f"Final = sort(enforce_diversity((1 - s*a) * model + s*a * anchor)), with s={ANCHOR_SCALE}",
        fill=text_color,
        font=body_font,
    )

    start_y = margin_top + title_height
    for index, mode in enumerate(modes):
        x = margin_x + label_width + index * (palette_block_width + block_gap)
        draw.text((x, start_y - 16), mode.title(), fill=text_color, font=body_font)

    for row, cls in enumerate(CLASSES):
        y = start_y + row * (swatch_size + row_gap)
        draw.text((margin_x, y + 14), cls.title(), fill=text_color, font=body_font)

        for column, mode in enumerate(modes):
            base_x = margin_x + label_width + column * (palette_block_width + block_gap)
            draw.rounded_rectangle(
                (base_x - 8, y - 6, base_x + palette_block_width - 16, y + swatch_size + 6),
                radius=10,
                fill=panel_fill,
                outline=line_color,
                width=1,
            )
            draw.text((base_x, y + 14), mode.title(), fill=text_color, font=body_font)
            for swatch_index, color in enumerate(comparison[cls][mode]):
                x = base_x + mode_width + swatch_index * (swatch_size + swatch_gap)
                draw.rounded_rectangle(
                    (x, y, x + swatch_size, y + swatch_size),
                    radius=8,
                    fill=hex_to_rgb(color),
                    outline=(255, 255, 255),
                    width=2,
                )

        if row < len(CLASSES) - 1:
            line_y = y + swatch_size + row_gap // 2
            draw.line((margin_x, line_y, image_width - margin_x, line_y), fill=line_color, width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw/anchor/final palette comparison image.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/anchor_comparison.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, clip_model, tokenizer = load_models()

    comparison: dict[str, dict[str, list[str]]] = {}
    for cls in CLASSES:
        raw = raw_palette(cls, model, clip_model, tokenizer)
        anchor = EMOTIONAL_ANCHORS[cls][0]
        final = sort_palette_by_luminance(enforce_diversity(apply_anchor(raw, cls)))
        comparison[cls] = {
            "model": to_hexes(raw),
            "anchor": to_hexes(anchor),
            "final": [oklab_to_hex(*color) for color in final],
        }

    build_image(comparison, args.output)
    print(f"Saved anchor comparison to {args.output}")


if __name__ == "__main__":
    main()
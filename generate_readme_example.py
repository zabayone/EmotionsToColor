"""Generate the README example PNG for the 9 target emotion classes.

The script runs the same inference pipeline used by ``inference.py`` and renders
the resulting palettes to a single PNG image suitable for the project README.

Default behaviour is deterministic (``temperature=0``) so the image is
reproducible across runs with the same model checkpoint.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from inference import generate, load_models


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_image(palettes: dict[str, list[str]], output_path: Path, temperature: float) -> None:
    background = (248, 246, 240)
    title_color = (28, 31, 36)
    text_color = (55, 60, 68)
    line_color = (215, 209, 198)

    margin_x = 44
    margin_top = 38
    margin_bottom = 28
    label_width = 270
    swatch_size = 84
    swatch_gap = 10
    row_gap = 18
    title_gap = 26
    footer_gap = 18

    row_width = label_width + 5 * swatch_size + 4 * swatch_gap
    image_width = margin_x * 2 + row_width
    image_height = (
        margin_top
        + 54
        + title_gap
        + len(CLASSES) * swatch_size
        + (len(CLASSES) - 1) * row_gap
        + footer_gap
        + margin_bottom
    )

    image = Image.new("RGB", (image_width, image_height), background)
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()

    title = "EmotionsToColor: 9 target emotion palettes"
    subtitle = (
        f"Generated with emotional anchor scaling (s=0.3), temperature={temperature:g}, "
        "5 colours per class"
    )
    draw.text((margin_x, margin_top), title, fill=title_color, font=title_font)
    draw.text((margin_x, margin_top + 18), subtitle, fill=text_color, font=body_font)

    start_y = margin_top + 54 + title_gap
    for index, cls in enumerate(CLASSES):
        y = start_y + index * (swatch_size + row_gap)
        draw.text((margin_x, y + 34), cls.title(), fill=text_color, font=body_font)

        for swatch_index, color in enumerate(palettes[cls]):
            x = margin_x + label_width + swatch_index * (swatch_size + swatch_gap)
            draw.rounded_rectangle(
                (x, y, x + swatch_size, y + swatch_size),
                radius=10,
                fill=hex_to_rgb(color),
                outline=(255, 255, 255),
                width=2,
            )

        if index < len(CLASSES) - 1:
            line_y = y + swatch_size + row_gap // 2
            draw.line(
                (margin_x, line_y, image_width - margin_x, line_y),
                fill=line_color,
                width=1,
            )

    footer_text = f"Output file: {output_path.name}"
    footer_y = image_height - margin_bottom - 12
    draw.text((margin_x, footer_y), footer_text, fill=text_color, font=body_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the README example PNG.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/readme_example.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Inference temperature. Use 0 for deterministic output.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used when temperature is non-zero.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    model, clip_model, tokenizer = load_models()
    palettes = {
        cls: generate(cls, model, clip_model, tokenizer, temperature=args.temperature)
        for cls in CLASSES
    }
    build_image(palettes, args.output, args.temperature)
    print(f"Saved README example to {args.output}")


if __name__ == "__main__":
    main()
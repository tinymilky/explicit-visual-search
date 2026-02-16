#!/usr/bin/env python3
"""
Visualize Gemini bbox annotations (per-row) on DocVQA images.

Input JSONL row example:
{
  "question": "...",
  "answer": "...",
  "image": "xxx.png",
  "width": 1692,
  "height": 2245,
  "bboxes": [{"role":"context","label":"...","bbox":[ymin,xmin,ymax,xmax]}, ...]
}

Notes:
- Gemini bbox format is [ymin, xmin, ymax, xmax] scaled to [0, 1000].
- We map back to pixels using: x = (xmin/1000)*W, y = (ymin/1000)*H.
- Each JSONL row outputs ONE unique image file even if rows share the same source image.

Usage:
  python viz_gemini_bboxes.py \
    --input annotated.jsonl \
    --image-root /path/to/images \
    --output-dir /path/to/out_viz
"""

import argparse
import json
import os
import textwrap
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_font(font_path: Optional[str], size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass
    # fallback
    return ImageFont.load_default()


def scaled_bbox_to_pixels(b: List[float], W: int, H: int) -> Tuple[int, int, int, int]:
    """
    b: [ymin, xmin, ymax, xmax] in [0,1000]
    returns: (x1, y1, x2, y2) in pixels
    """
    if not (isinstance(b, list) or isinstance(b, tuple)) or len(b) != 4:
        return (0, 0, 0, 0)
    ymin, xmin, ymax, xmax = b
    x1 = int(round((float(xmin) / 1000.0) * W))
    y1 = int(round((float(ymin) / 1000.0) * H))
    x2 = int(round((float(xmax) / 1000.0) * W))
    y2 = int(round((float(ymax) / 1000.0) * H))
    # clamp
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    # normalize ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def wrap_line(s: str, width_chars: int) -> List[str]:
    s = (s or "").strip()
    if not s:
        return [""]
    return textwrap.wrap(s, width=width_chars, break_long_words=True, replace_whitespace=False) or [""]


def visualize_one(
    img_path: str,
    question: str,
    answer: str,
    bboxes: List[Dict[str, Any]],
    out_path: str,
    font_path: Optional[str] = None,
) -> None:
    im = Image.open(img_path).convert("RGB")
    W, H = im.size

    # Fonts
    font_title = load_font(font_path, size=18)
    font_body = load_font(font_path, size=14)
    font_id = load_font(font_path, size=16)

    # Build header text (question/answer + legend)
    lines: List[str] = []
    lines += wrap_line(f"Q: {question}", 120)
    lines += wrap_line(f"A: {answer}", 120)
    lines.append("")  # spacer

    # Legend lines: "1) role | label"
    for i, bb in enumerate(bboxes, start=1):
        role = (bb.get("role", "") or "").strip()
        label = (bb.get("label", "") or "").strip()
        # keep legend compact
        legend = f"{i}) {role} | {label}"
        lines += wrap_line(legend, 120)

    # Header height estimation
    line_h = 20
    header_h = max(120, min(520, 20 + line_h * len(lines)))  # cap to avoid huge headers
    pad = 12

    canvas = Image.new("RGB", (W, H + header_h), (255, 255, 255))
    canvas.paste(im, (0, header_h))

    draw = ImageDraw.Draw(canvas)

    # Header background band (light gray)
    draw.rectangle([0, 0, W, header_h], fill=(245, 245, 245))

    # Draw header text
    y = pad
    for idx_line, t in enumerate(lines):
        f = font_title if idx_line in (0, 1) else font_body
        draw.text((pad, y), t, fill=(0, 0, 0), font=f)
        y += line_h

    # Draw boxes on the image region (offset by header_h)
    for i, bb in enumerate(bboxes, start=1):
        role = (bb.get("role", "") or "").strip().lower()
        raw = bb.get("bbox", None)

        x1, y1, x2, y2 = scaled_bbox_to_pixels(raw, W, H)
        y1o, y2o = y1 + header_h, y2 + header_h

        # Simple role-based color
        color = (0, 180, 0) if role == "answer" else (220, 0, 0)

        # Rectangle
        draw.rectangle([x1, y1o, x2, y2o], outline=color, width=3)

        # ID tag near top-left of box
        tag = str(i)
        tw, th = draw.textbbox((0, 0), tag, font=font_id)[2:]
        tag_x = x1
        tag_y = max(header_h, y1o - th - 4)
        draw.rectangle([tag_x, tag_y, tag_x + tw + 10, tag_y + th + 6], fill=color)
        draw.text((tag_x + 5, tag_y + 3), tag, fill=(255, 255, 255), font=font_id)

    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Annotated jsonl from Gemini")
    ap.add_argument("--image-root", required=True, help="Root directory containing images")
    ap.add_argument("--output-dir", required=True, help="Output directory for visualizations")
    ap.add_argument("--font", default=None, help="Optional path to a .ttf/.otf font file")
    ap.add_argument("--no-total", action="store_true", help="Do not pre-count lines for tqdm total")
    args = ap.parse_args()

    ensure_dir(args.output_dir)

    total = None
    if not args.no_total:
        with open(args.input, "rb") as f:
            total = sum(1 for _ in f)

    with open(args.input, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(tqdm(fin, total=total, desc="Visualizing", dynamic_ncols=True), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            image_name = row.get("image", "")
            question = row.get("question", "")
            answer = row.get("answer", "")
            bboxes = row.get("bboxes", []) or []

            img_path = os.path.join(args.image_root, image_name)
            if not os.path.isfile(img_path):
                # Skip missing images
                continue

            # Unique per row output filename
            base = os.path.splitext(os.path.basename(image_name))[0]
            out_name = f"{idx:06d}__{base}.png"
            out_path = os.path.join(args.output_dir, out_name)

            visualize_one(
                img_path=img_path,
                question=question,
                answer=answer,
                bboxes=bboxes,
                out_path=out_path,
                font_path=args.font,
            )

    print(f"Done. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()

'''
python viz_gemini_bboxes.py \
  --input ./tmp_annotated.jsonl \
  --image-root ./docvqa_imgs \
  --output-dir ./viz_out_tmp


python viz_gemini_bboxes.py --input ./tmp_annotated.jsonl --image-root ./docvqa_imgs --output-dir ./viz_out_tmp
'''
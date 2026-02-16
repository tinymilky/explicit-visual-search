#!/usr/bin/env python3
"""
Visualize InfoVQA bbox annotations (per-row) on infographic images.

Input JSONL row example (pixel bboxes):
{
  "question": "...",
  "answer": "...",
  "possible_answers": [...],
  "image": "20471.jpeg",
  "width": 596,
  "height": 5107,
  "bboxs": [[xmin,ymin,xmax,ymax], ...],
  "dataset": "infographicsvqa",
  "split": "train"
}

Notes:
- bboxs are [xmin, ymin, xmax, ymax] in ORIGINAL IMAGE PIXELS (floats allowed).
- We clamp boxes to the loaded image size.

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
    return ImageFont.load_default()


def pixel_bbox_to_pixels(b: List[float], W: int, H: int) -> Tuple[int, int, int, int]:
    """
    b: [xmin, ymin, xmax, ymax] in pixel coords (floats ok)
    returns: (x1, y1, x2, y2) in pixels (int), clamped to image bounds
    """
    if not (isinstance(b, list) or isinstance(b, tuple)) or len(b) != 4:
        return (0, 0, 0, 0)

    xmin, ymin, xmax, ymax = b
    x1 = int(round(float(xmin)))
    y1 = int(round(float(ymin)))
    x2 = int(round(float(xmax)))
    y2 = int(round(float(ymax)))

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
    possible_answers: List[str],
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

    # Header text (question/answer + possible answers + legend)
    lines: List[str] = []
    lines += wrap_line(f"Q: {question}", 120)
    lines += wrap_line(f"A: {answer}", 120)
    if possible_answers:
        lines += wrap_line(f"Possible: {', '.join(possible_answers)}", 120)
    lines.append("")

    # print(bboxes)
    # print("here")
    # quit()
    # Legend lines: "1) bbox"
    for i, bb in enumerate(bboxes, start=1):
        raw = bb.get("bbox", None)
        # raw = bb
        lines += wrap_line(f"{i}) {raw}", 120)

    # Header height estimation
    line_h = 20
    header_h = max(120, min(520, 20 + line_h * len(lines)))
    pad = 12

    canvas = Image.new("RGB", (W, H + header_h), (255, 255, 255))
    canvas.paste(im, (0, header_h))

    draw = ImageDraw.Draw(canvas)

    # Header background band
    draw.rectangle([0, 0, W, header_h], fill=(245, 245, 245))

    # Draw header text
    y = pad
    for idx_line, t in enumerate(lines):
        f = font_title if idx_line in (0, 1) else font_body
        draw.text((pad, y), t, fill=(0, 0, 0), font=f)
        y += line_h

    # Draw boxes on the image region (offset by header_h)
    for i, bb in enumerate(bboxes, start=1):
        raw = bb.get("bbox", None)
        # raw = bb
        # print(f"DEBUG: raw bbox for box {i}: {raw}")

        x1, y1, x2, y2 = pixel_bbox_to_pixels(raw, W, H)
        y1o, y2o = y1 + header_h, y2 + header_h

        color = (220, 0, 0)  # single color (no role info)

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
    ap.add_argument("--input", required=True, help="JSONL with pixel bboxs")
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
            possible_answers = row.get("possible_answers", []) or []

            img_path = os.path.join(args.image_root, image_name)
            if not os.path.isfile(img_path):
                continue

            # Accept either "bboxs" (your format) or "bboxes" (fallback)
            raw_boxes = row.get("bboxs", None)
            if raw_boxes is None:
                raw_boxes = row.get("bboxes", []) or []

            # Normalize to list[dict] with key "bbox"
            bboxes: List[Dict[str, Any]] = []
            if isinstance(raw_boxes, list):
                for b in raw_boxes:
                    if isinstance(b, (list, tuple)) and len(b) == 4:
                        bboxes.append({"bbox": list(b)})
                    elif isinstance(b, dict) and "bbox" in b:
                        bboxes.append({"bbox": b["bbox"]})

            base = os.path.splitext(os.path.basename(image_name))[0]
            out_name = f"{idx:06d}__{base}.png"
            out_path = os.path.join(args.output_dir, out_name)

            visualize_one(
                img_path=img_path,
                question=question,
                answer=answer,
                possible_answers=possible_answers,
                bboxes=bboxes,
                out_path=out_path,
                font_path=args.font,
            )

    print(f"Done. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()

'''
python viz_viscot_bboxes.py --input infographicsvqa_cot_tmp.jsonl --image-root ./infographicsvqa_images --output-dir ./viscot_viz_out_tmp
'''
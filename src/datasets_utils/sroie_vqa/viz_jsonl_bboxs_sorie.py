#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize bboxs from a JSONL file on images.

Input JSONL row example (pixel coords):
{
  "question": "...",
  "answer": "...",
  "image": "xxx.jpg",
  "width": 1700,
  "height": 2200,
  "bboxs": [[xmin, ymin, xmax, ymax], ...],
  "dataset": "dude",
  "split": "train"
}

Notes:
- bboxs are assumed to be [xmin, ymin, xmax, ymax] in pixel coordinates.
- If row["width"]/row["height"] differ from the actual loaded image size, boxes are rescaled.
- Output is one visualization image per row.

Usage:
  python viz_jsonl_bboxs.py \
    --input data.jsonl \
    --image-root /path/to/images \
    --output-dir /path/to/out_viz

Optional:
  --max-rows 200
  --no-header        # don't draw question/answer header panel
  --thickness 3
  --font /path/to/font.ttf
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple

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


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def to_int_box_xyxy(
    b: Any,
    W: int,
    H: int,
    orig_W: Optional[float],
    orig_H: Optional[float],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert [xmin,ymin,xmax,ymax] to integer pixel box in current image size.
    If orig_W/orig_H are provided and differ, rescale.
    """
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    try:
        xmin, ymin, xmax, ymax = map(float, b)
    except Exception:
        return None

    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin

    sx = 1.0
    sy = 1.0
    if orig_W and orig_H and orig_W > 0 and orig_H > 0 and (int(orig_W) != int(W) or int(orig_H) != int(H)):
        sx = float(W) / float(orig_W)
        sy = float(H) / float(orig_H)

    xmin *= sx
    xmax *= sx
    ymin *= sy
    ymax *= sy

    xmin = _clamp(xmin, 0.0, W - 1.0)
    xmax = _clamp(xmax, 0.0, W - 1.0)
    ymin = _clamp(ymin, 0.0, H - 1.0)
    ymax = _clamp(ymax, 0.0, H - 1.0)

    x1 = int(round(xmin))
    y1 = int(round(ymin))
    x2 = int(round(xmax))
    y2 = int(round(ymax))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if x2 == x1 or y2 == y1:
        # degenerate
        return None

    return x1, y1, x2, y2


def wrap_lines(s: str, width_chars: int) -> List[str]:
    s = (s or "").strip()
    if not s:
        return [""]
    return textwrap.wrap(s, width=width_chars, break_long_words=True, replace_whitespace=False) or [""]


def visualize_row(
    row: Dict[str, Any],
    img_path: str,
    out_path: str,
    *,
    thickness: int = 3,
    font_path: Optional[str] = None,
    draw_header: bool = True,
) -> None:
    im = Image.open(img_path).convert("RGB")
    W, H = im.size

    orig_W = None
    orig_H = None
    try:
        orig_W = float(row.get("width", None))
        orig_H = float(row.get("height", None))
    except Exception:
        orig_W, orig_H = None, None

    raw_boxes = row.get("bboxs", []) or []
    if not isinstance(raw_boxes, list):
        raw_boxes = []

    boxes: List[Tuple[int, int, int, int]] = []
    for b in raw_boxes:
        bx = to_int_box_xyxy(b, W, H, orig_W, orig_H)
        if bx is not None:
            boxes.append(bx)

    # Header content
    header_h = 0
    pad = 12
    font_title = load_font(font_path, 18)
    font_body = load_font(font_path, 14)
    font_id = load_font(font_path, 16)

    header_lines: List[str] = []
    if draw_header:
        q = row.get("question", "")
        a = row.get("answer", "")
        header_lines += wrap_lines(f"Q: {q}", 120)
        header_lines += wrap_lines(f"A: {a}", 120)
        header_lines += wrap_lines(f"image: {os.path.basename(img_path)} | boxes: {len(boxes)}", 120)
        header_lines.append("")
        for i, b in enumerate(raw_boxes, start=1):
            header_lines += wrap_lines(f"{i}) {b}", 120)

        line_h = 20
        header_h = max(100, min(520, pad * 2 + line_h * len(header_lines)))

    # Build canvas
    canvas = Image.new("RGB", (W, H + header_h), (255, 255, 255))
    canvas.paste(im, (0, header_h))
    draw = ImageDraw.Draw(canvas)

    # Header background + text
    if draw_header:
        draw.rectangle([0, 0, W, header_h], fill=(245, 245, 245))
        y = pad
        for li, t in enumerate(header_lines):
            f = font_title if li in (0, 1) else font_body
            draw.text((pad, y), t, fill=(0, 0, 0), font=f)
            y += 20

    # Draw bboxes
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        y1o, y2o = y1 + header_h, y2 + header_h
        color = (220, 0, 0)

        draw.rectangle([x1, y1o, x2, y2o], outline=color, width=max(1, thickness))

        tag = str(i)
        tb = draw.textbbox((0, 0), tag, font=font_id)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        tag_x = x1
        tag_y = max(header_h, y1o - th - 4)

        draw.rectangle([tag_x, tag_y, tag_x + tw + 10, tag_y + th + 6], fill=color)
        draw.text((tag_x + 5, tag_y + 3), tag, fill=(255, 255, 255), font=font_id)

    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL with bboxs")
    ap.add_argument("--image-root", required=True, help="Root directory containing images")
    ap.add_argument("--output-dir", required=True, help="Output directory for visualizations")
    ap.add_argument("--max-rows", type=int, default=0, help="If >0, process only first N rows")
    ap.add_argument("--thickness", type=int, default=3, help="Box outline thickness")
    ap.add_argument("--font", default=None, help="Optional .ttf/.otf font for nicer text")
    ap.add_argument("--no-header", action="store_true", help="Do not draw the header panel")
    ap.add_argument("--no-total", action="store_true", help="Do not pre-count lines for tqdm total")
    args = ap.parse_args()

    ensure_dir(args.output_dir)

    total = None
    if not args.no_total:
        try:
            with open(args.input, "rb") as f:
                total = sum(1 for _ in f)
        except Exception:
            total = None

    n = 0
    with open(args.input, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=total, desc="Visualizing JSONL bboxs", dynamic_ncols=True):
            if args.max_rows > 0 and n >= args.max_rows:
                break

            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue

            image_name = row.get("image", "")
            if not isinstance(image_name, str) or not image_name:
                continue

            img_path = os.path.join(args.image_root, image_name)
            if not os.path.isfile(img_path):
                continue

            n += 1
            base = os.path.splitext(os.path.basename(image_name))[0]
            out_name = f"{n:06d}__{base}.png"
            out_path = os.path.join(args.output_dir, out_name)

            visualize_row(
                row=row,
                img_path=img_path,
                out_path=out_path,
                thickness=args.thickness,
                font_path=args.font,
                draw_header=not args.no_header,
            )

    print(f"Done. Wrote {n} visualizations to: {args.output_dir}")


if __name__ == "__main__":
    main()

'''
python viz_jsonl_bboxs_sorie.py --input sroie_cot_train_dropped_rows.jsonl --image-root ./sroie_imgs --output-dir ./viz_sroie_bboxs --max-rows 400
'''
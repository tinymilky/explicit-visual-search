#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize OCR LINE bounding boxes on the corresponding image.

Assumptions:
- Each sample has:
  - OCR JSON: <ocr_dir>/<id>.json
  - Image:    <img_dir>/<id>.jpeg   (falls back to .jpg/.png/.webp if not found)
- BoundingBox coords are normalized in [0, 1] with fields: Left, Top, Width, Height
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def find_image_path(img_dir: Path, sample_id: str) -> Path:
    # User said id.jpeg; we still add graceful fallbacks.
    exts = [".jpeg", ".jpg", ".png", ".webp", ".tif", ".tiff"]
    for ext in exts:
        p = img_dir / f"{sample_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find image for id={sample_id} under {img_dir} with extensions {exts}"
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_blocks(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Iterate candidate block dicts from common OCR JSON structures.

    Supports:
    - AWS Textract-style: {"Blocks": [ ... ]}
    - Your example: {"LINE": [ ... ], "PAGE": [ ... ], ...}
    - Raw list: [ ... ]
    """
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                yield it
        return

    if not isinstance(obj, dict):
        return

    # Common: Textract {"Blocks": [...]}
    if isinstance(obj.get("Blocks"), list):
        for it in obj["Blocks"]:
            if isinstance(it, dict):
                yield it
        return

    # Your example: keys are block types with list values (e.g., "LINE": [...])
    for v in obj.values():
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict):
                    yield it


def extract_line_blocks(ocr_json: Any) -> List[Dict[str, Any]]:
    """
    Extract LINE blocks robustly.

    Priority:
    - If top-level has "LINE": use it directly.
    - Else scan all blocks and filter BlockType == "LINE".
    """
    # if isinstance(ocr_json, dict) and isinstance(ocr_json.get("LINE"), list):
    #     return [b for b in ocr_json["LINE"] if isinstance(b, dict)]

    # lines: List[Dict[str, Any]] = []
    # for b in iter_blocks(ocr_json):
    #     if b.get("BlockType") == "LINE":
    #         lines.append(b)
    # return lines

    if isinstance(ocr_json, dict) and isinstance(ocr_json.get("LINE"), list):
        return [b for b in ocr_json["LINE"] if isinstance(b, dict)]

    lines: List[Dict[str, Any]] = []
    for b in iter_blocks(ocr_json):
        if b.get("BlockType") == "LINE":
            lines.append(b)
    return lines

def bbox_norm_to_pixel(
    bb: Dict[str, Any],
    img_w: int,
    img_h: int,
    clamp: bool = True,
) -> Tuple[int, int, int, int]:
    left = _as_float(bb.get("Left"), 0.0)
    top = _as_float(bb.get("Top"), 0.0)
    width = _as_float(bb.get("Width"), 0.0)
    height = _as_float(bb.get("Height"), 0.0)

    x1 = left * img_w
    y1 = top * img_h
    x2 = (left + width) * img_w
    y2 = (top + height) * img_h

    if clamp:
        x1 = max(0.0, min(x1, img_w - 1.0))
        y1 = max(0.0, min(y1, img_h - 1.0))
        x2 = max(0.0, min(x2, img_w - 1.0))
        y2 = max(0.0, min(y2, img_h - 1.0))

    # Ensure proper ordering
    xa, xb = sorted([x1, x2])
    ya, yb = sorted([y1, y2])

    return int(round(xa)), int(round(ya)), int(round(xb)), int(round(yb))


def draw_line_boxes(
    image: Image.Image,
    line_blocks: List[Dict[str, Any]],
    *,
    thickness: int = 3,
    draw_text: bool = True,
    max_text_len: int = 60,
) -> Image.Image:
    # Work in RGBA so we can do semi-transparent fills if desired.
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()

    w, h = img.size
    for i, b in enumerate(line_blocks):
        geom = b.get("Geometry") or {}
        bb = (geom.get("BoundingBox") or {}) if isinstance(geom, dict) else {}
        if not isinstance(bb, dict):
            continue

        x1, y1, x2, y2 = bbox_norm_to_pixel(bb, w, h, clamp=True)
        if x2 <= x1 or y2 <= y1:
            continue

        # Rectangle outline + light fill
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(255, 0, 0, 255),
            width=thickness,
            fill=(255, 0, 0, 40),
        )

        if draw_text:
            txt = b.get("Text", "")
            if not isinstance(txt, str):
                txt = str(txt)
            txt = txt.strip().replace("\n", " ")
            if len(txt) > max_text_len:
                txt = txt[: max_text_len - 1] + "…"

            conf = b.get("Confidence", None)
            conf_s = ""
            if conf is not None:
                conf_s = f" ({_as_float(conf, 0.0):.1f})"

            label = f"{i}: {txt}{conf_s}" if txt else f"{i}{conf_s}"

            # Label background
            pad = 2
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            lx1, ly1 = x1, max(0, y1 - th - 2 * pad)
            lx2, ly2 = min(w, lx1 + tw + 2 * pad), ly1 + th + 2 * pad
            draw.rectangle([lx1, ly1, lx2, ly2], fill=(0, 0, 0, 180))
            draw.text((lx1 + pad, ly1 + pad), label, fill=(255, 255, 255, 255), font=font)

    out = Image.alpha_composite(img, overlay).convert("RGB")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="Sample id (expects <id>.json and <id>.(jpeg|jpg|png...))")
    ap.add_argument("--ocr_dir", required=True, help="Folder containing OCR json files")
    ap.add_argument("--img_dir", required=True, help="Folder containing images")
    ap.add_argument("--out", default=None, help="Output image path (default: <id>_lines.jpg next to OCR json)")
    ap.add_argument("--no_text", action="store_true", help="Do not draw LINE text labels")
    ap.add_argument("--thickness", type=int, default=3, help="Box outline thickness in pixels")
    ap.add_argument("--max_text_len", type=int, default=60, help="Max characters for the label text")
    ap.add_argument("--show", action="store_true", help="Open the output image with the default viewer")
    args = ap.parse_args()

    sample_id = args.id
    ocr_dir = Path(args.ocr_dir)
    img_dir = Path(args.img_dir)

    ocr_path = ocr_dir / f"{sample_id}.json"
    if not ocr_path.exists():
        raise FileNotFoundError(f"OCR json not found: {ocr_path}")

    img_path = find_image_path(img_dir, sample_id)

    ocr = load_json(ocr_path)
    line_blocks = extract_line_blocks(ocr)
    print(f"[INFO] id={sample_id} | image={img_path.name} | LINE blocks={len(line_blocks)}")

    img = Image.open(img_path)
    vis = draw_line_boxes(
        img,
        line_blocks,
        thickness=max(1, args.thickness),
        draw_text=not args.no_text,
        max_text_len=max(10, args.max_text_len),
    )

    out_path = Path(args.out) if args.out else (ocr_dir / f"{sample_id}_lines.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_path, quality=95)
    print(f"[OK] Saved: {out_path}")

    if args.show:
        vis.show()


if __name__ == "__main__":
    main()

'''
python vis_ocr_lines.py  --id 31078  --ocr_dir ./valid_ocr  --img_dir ./valid_images  --out ./31078_lines.jpg

'''
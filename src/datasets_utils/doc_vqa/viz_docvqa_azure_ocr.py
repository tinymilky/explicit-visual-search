#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def find_image_file(image_root: Path, stem: str) -> Optional[Path]:
    exts = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]
    for ext in exts:
        p = image_root / f"{stem}{ext}"
        if p.is_file():
            return p
    # fallback: glob by stem.*
    hits = sorted(image_root.glob(f"{stem}.*"))
    for p in hits:
        if p.suffix.lower() in exts and p.is_file():
            return p
    return None


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def poly_from_bbox8(b: List[float]) -> List[Tuple[float, float]]:
    """
    Azure OCR boundingBox: [x1,y1,x2,y2,x3,y3,x4,y4]
    """
    if not (isinstance(b, list) and len(b) == 8):
        raise ValueError(f"Expected bbox length 8, got {type(b)} len={len(b) if isinstance(b, list) else 'NA'}")
    pts = [(float(b[i]), float(b[i + 1])) for i in range(0, 8, 2)]
    return pts


def poly_bbox_xyxy(pts: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def draw_poly(draw: ImageDraw.ImageDraw, pts: List[Tuple[float, float]], width: int = 2):
    # PIL polygon outline doesn't support width consistently; use line loop
    pts2 = pts + [pts[0]]
    draw.line(pts2, width=width)


def get_font(font_size: int) -> ImageFont.FreeTypeFont:
    # Try common fonts; fallback to default
    for fp in [
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="Sample id / stem (expects <id>.json and <id>.(png/jpg/jpeg/...))")
    ap.add_argument("--ocr-root", required=True, help="Folder containing OCR json files")
    ap.add_argument("--image-root", required=True, help="Folder containing image files")
    ap.add_argument("--out", default=None, help="Output image path (default: <id>_ocrviz.png next to CWD)")
    ap.add_argument("--page", type=int, default=1, help="Page number in recognitionResults (default: 1)")
    ap.add_argument("--draw-level", choices=["line", "word", "both"], default="both",
                    help="Draw LINE boxes, WORD boxes, or both (default: both)")
    ap.add_argument("--show-text", action="store_true", help="Draw text labels near boxes (can be cluttered)")
    ap.add_argument("--max-text", type=int, default=40, help="Max characters to draw per label (default: 40)")
    ap.add_argument("--line-width", type=int, default=3, help="Polygon stroke width (default: 3)")
    ap.add_argument("--font-size", type=int, default=16, help="Font size for labels (default: 16)")
    args = ap.parse_args()

    stem = args.id
    ocr_root = Path(args.ocr_root)
    image_root = Path(args.image_root)

    ocr_path = ocr_root / f"{stem}.json"
    if not ocr_path.is_file():
        raise FileNotFoundError(f"OCR json not found: {ocr_path}")

    img_path = find_image_file(image_root, stem)
    if img_path is None:
        raise FileNotFoundError(f"Image not found under {image_root} for stem={stem}")

    out_path = Path(args.out) if args.out else Path(f"{stem}_ocrviz.png")

    ocr = load_json(ocr_path)

    # Load image
    img = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_font(args.font_size)

    # Parse recognitionResults
    recs = ocr.get("recognitionResults", [])
    if not isinstance(recs, list) or not recs:
        raise ValueError("OCR json has no recognitionResults[]")

    # Select page
    page_obj = None
    for r in recs:
        if isinstance(r, dict) and int(r.get("page", -1)) == args.page:
            page_obj = r
            break
    if page_obj is None:
        # fallback: first page
        page_obj = recs[0]

    lines = page_obj.get("lines", [])
    if not isinstance(lines, list):
        raise ValueError("recognitionResults[].lines is not a list")

    # Simple colors (RGBA)
    # line: red-ish, word: green-ish
    LINE_COLOR = (255, 0, 0, 200)
    WORD_COLOR = (0, 200, 0, 160)
    TEXT_BG = (0, 0, 0, 160)
    TEXT_FG = (255, 255, 255, 255)

    def draw_label(x: float, y: float, s: str):
        s = s.strip()
        if not s:
            return
        if len(s) > args.max_text:
            s = s[: args.max_text - 3] + "..."
        # background box
        tw, th = draw.textbbox((0, 0), s, font=font)[2:]
        pad = 2
        draw.rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], fill=TEXT_BG)
        draw.text((x + pad, y + pad), s, font=font, fill=TEXT_FG)

    # Draw
    for li, line in enumerate(lines):
        if not isinstance(line, dict):
            continue

        if args.draw_level in ("line", "both"):
            bb = line.get("boundingBox", None)
            if isinstance(bb, list) and len(bb) == 8:
                pts = poly_from_bbox8(bb)
                draw_poly(draw, pts, width=args.line_width)
                # stroke color: draw line multiple times is not needed if width works; use fill by drawing on overlay
                draw.line(pts + [pts[0]], fill=LINE_COLOR, width=args.line_width)

                if args.show_text:
                    x1, y1, _, _ = poly_bbox_xyxy(pts)
                    txt = str(line.get("text", ""))
                    draw_label(x1, max(0, y1 - args.font_size - 6), f"[L{li}] {txt}")

        if args.draw_level in ("word", "both"):
            words = line.get("words", [])
            if isinstance(words, list):
                for wi, w in enumerate(words):
                    if not isinstance(w, dict):
                        continue
                    bb = w.get("boundingBox", None)
                    if isinstance(bb, list) and len(bb) == 8:
                        pts = poly_from_bbox8(bb)
                        draw.line(pts + [pts[0]], fill=WORD_COLOR, width=max(1, args.line_width - 1))
                        if args.show_text:
                            x1, y1, _, _ = poly_bbox_xyxy(pts)
                            txt = str(w.get("text", ""))
                            draw_label(x1, max(0, y1 - args.font_size - 6), txt)

    # Composite and save
    out = Image.alpha_composite(img, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"Saved: {out_path} (image={img_path.name}, ocr={ocr_path.name})")


if __name__ == "__main__":
    main()

'''
python viz_docvqa_azure_ocr.py --id glxb0228_52 --ocr-root ./spdocvqa_ocr --image-root ./docvqa_imgs --draw-level both --show-text

'''
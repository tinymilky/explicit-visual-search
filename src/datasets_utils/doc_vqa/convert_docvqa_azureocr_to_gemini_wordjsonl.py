#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert DocVQA-style Azure Read OCR JSON (recognitionResults) into Gemini-friendly word-level JSONL.

Input OCR JSON example (Azure Read):
{
  "status": "Succeeded",
  "recognitionResults": [
    {
      "page": 1,
      "width": 1643,
      "height": 2153,
      "unit": "pixel",
      "lines": [
        {
          "boundingBox": [x1,y1,x2,y2,x3,y3,x4,y4],
          "text": "...",
          "words": [
            {"boundingBox": [...], "text": "...", "confidence": "Low" | ...}
          ]
        }
      ]
    }
  ]
}

Output JSONL (one row per WORD):
{
  "doc_id": "<stem>",
  "page": 1,
  "line_idx": 0,
  "word_idx": 0,              # global within doc_id (across pages)
  "word_in_line": 0,
  "text": "PROTECT",
  "bbox": [ymin, xmin, ymax, xmax],   # normalized to [0,1000], rounded to 2 decimals
  "unit": "norm_0_1000"
}

By default writes one <stem>.jsonl for each <stem>.json in the OCR folder.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def poly8_to_xyxy(poly8: List[Any]) -> Optional[Tuple[float, float, float, float]]:
    """
    Azure boundingBox: [x1,y1,x2,y2,x3,y3,x4,y4]
    Returns axis-aligned bbox (xmin, ymin, xmax, ymax) in the same units.
    """
    if not isinstance(poly8, list) or len(poly8) != 8:
        return None
    try:
        pts = [(float(poly8[i]), float(poly8[i + 1])) for i in range(0, 8, 2)]
    except Exception:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def xyxy_pixel_to_yxyx_norm1000(
    xyxy: Tuple[float, float, float, float],
    page_w: float,
    page_h: float,
    ndigits: int = 2,
) -> List[float]:
    xmin, ymin, xmax, ymax = xyxy
    # normalize to [0,1000]
    xmin_n = (xmin / page_w) * 1000.0
    xmax_n = (xmax / page_w) * 1000.0
    ymin_n = (ymin / page_h) * 1000.0
    ymax_n = (ymax / page_h) * 1000.0

    # clamp and reorder to [ymin,xmin,ymax,xmax]
    y1 = clamp(ymin_n, 0.0, 1000.0)
    x1 = clamp(xmin_n, 0.0, 1000.0)
    y2 = clamp(ymax_n, 0.0, 1000.0)
    x2 = clamp(xmax_n, 0.0, 1000.0)

    # ensure min<=max
    yy1, yy2 = (y1, y2) if y1 <= y2 else (y2, y1)
    xx1, xx2 = (x1, x2) if x1 <= x2 else (x2, x1)

    return [round(yy1, ndigits), round(xx1, ndigits), round(yy2, ndigits), round(xx2, ndigits)]


def iter_ocr_json_files(root: Path) -> List[Path]:
    return sorted([p for p in root.glob("*.json") if p.is_file()])


def load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-root", required=True, help="Folder containing Azure Read OCR json files (*.json)")
    ap.add_argument("--out-root", required=True, help="Output folder for converted word-level jsonl")
    ap.add_argument("--ndigits", type=int, default=2, help="Round bbox coords to N decimals (default=2)")
    ap.add_argument("--include-line-text", action="store_true",
                    help="Include 'line_text' field in each word row (slightly larger output).")
    ap.add_argument("--skip-non-succeeded", action="store_true",
                    help="Skip files whose status != 'Succeeded' (default: still attempt).")
    args = ap.parse_args()

    ocr_root = Path(args.ocr_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ocr_files = iter_ocr_json_files(ocr_root)
    if not ocr_files:
        raise FileNotFoundError(f"No *.json found in {ocr_root}")

    n_files = 0
    n_pages = 0
    n_words = 0
    n_skipped = 0

    for jp in tqdm(ocr_files, desc="Converting OCR JSON -> word JSONL", dynamic_ncols=True):
        n_files += 1
        doc_id = jp.stem

        try:
            obj = load_json(jp)
        except Exception:
            n_skipped += 1
            continue

        status = str(obj.get("status", "") or "")
        if args.skip_non_succeeded and status and status.lower() != "succeeded":
            n_skipped += 1
            continue

        recs = obj.get("recognitionResults", None)
        if not isinstance(recs, list) or not recs:
            n_skipped += 1
            continue

        out_path = out_root / f"{doc_id}.jsonl"
        word_idx_global = 0

        with open(out_path, "w", encoding="utf-8") as fout:
            for rec in recs:
                if not isinstance(rec, dict):
                    continue
                page = int(rec.get("page", 1) or 1)
                page_w = rec.get("width", None)
                page_h = rec.get("height", None)
                unit = str(rec.get("unit", "") or "")

                if page_w is None or page_h is None:
                    # cannot normalize
                    continue

                try:
                    page_w = float(page_w)
                    page_h = float(page_h)
                except Exception:
                    continue

                if page_w <= 0 or page_h <= 0:
                    continue

                n_pages += 1

                lines = rec.get("lines", [])
                if not isinstance(lines, list):
                    continue

                for li, line in enumerate(lines):
                    if not isinstance(line, dict):
                        continue
                    line_text = str(line.get("text", "") or "")
                    words = line.get("words", [])

                    # Azure usually provides words; if missing, fallback to a pseudo-word from the line itself
                    if not isinstance(words, list) or len(words) == 0:
                        bb8 = line.get("boundingBox", None)
                        xyxy = poly8_to_xyxy(bb8) if bb8 is not None else None
                        if xyxy is None:
                            continue
                        bbox = xyxy_pixel_to_yxyx_norm1000(xyxy, page_w, page_h, ndigits=args.ndigits)

                        row = {
                            "doc_id": doc_id,
                            "page": page,
                            "line_idx": li,
                            "word_idx": word_idx_global,
                            "word_in_line": 0,
                            "text": line_text,
                            "bbox": bbox,
                            "unit": "norm_0_1000",
                            "_src_unit": unit,
                            "_src_level": "LINE_FALLBACK",
                        }
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        word_idx_global += 1
                        n_words += 1
                        continue

                    for wi, w in enumerate(words):
                        if not isinstance(w, dict):
                            continue
                        text = str(w.get("text", "") or "")
                        bb8 = w.get("boundingBox", None)
                        xyxy = poly8_to_xyxy(bb8) if bb8 is not None else None
                        if xyxy is None:
                            continue

                        bbox = xyxy_pixel_to_yxyx_norm1000(xyxy, page_w, page_h, ndigits=args.ndigits)

                        row = {
                            "doc_id": doc_id,
                            "page": page,
                            "line_idx": li,
                            "word_idx": word_idx_global,
                            "word_in_line": wi,
                            "text": text,
                            "bbox": bbox,
                            "unit": "norm_0_1000",
                            "_src_unit": unit,
                            "_src_level": "WORD",
                        }

                        conf = w.get("confidence", None)
                        if conf is not None:
                            # Azure sometimes gives "Low"/"High" etc.
                            row["confidence"] = conf

                        if args.include_line_text:
                            row["line_text"] = line_text

                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        word_idx_global += 1
                        n_words += 1

    print("\nDone.")
    print(f"files_total={n_files} files_skipped={n_skipped}")
    print(f"pages_total={n_pages} words_total={n_words}")
    print(f"out_root={out_root}")


if __name__ == "__main__":
    main()

'''
python convert_docvqa_azureocr_to_gemini_wordjsonl.py --ocr-root ./spdocvqa_ocr --out-root ./docvqa_ocr_word_jsonl --ndigits 2
'''
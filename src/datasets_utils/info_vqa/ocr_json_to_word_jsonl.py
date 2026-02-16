#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert messy OCR JSON (Textract-like PAGE/LINE/WORD) into clean word-level JSONL.

Each OCR .json -> one output .jsonl:
- one row per WORD block
- bbox in Gemini format: [ymin, xmin, ymax, xmax]
- coords normalized to [0,1000], rounded to 2 decimals
- preserves original WORD list order

Optional:
- map each WORD to its parent LINE using LINE->Relationships[CHILD].Ids
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def bb01_to_1000_yxyx(bb: Dict[str, Any], ndigits: int = 2) -> Optional[List[float]]:
    """
    bb is expected like:
      {"Left": 0.1, "Top": 0.2, "Width": 0.3, "Height": 0.05}
    in [0,1] relative coords.
    Return [ymin, xmin, ymax, xmax] in [0,1000].
    """
    if not isinstance(bb, dict):
        return None
    try:
        left = float(bb.get("Left"))
        top = float(bb.get("Top"))
        w = float(bb.get("Width"))
        h = float(bb.get("Height"))
    except Exception:
        return None

    xmin = left
    ymin = top
    xmax = left + w
    ymax = top + h

    xmin = clamp(xmin, 0.0, 1.0)
    ymin = clamp(ymin, 0.0, 1.0)
    xmax = clamp(xmax, 0.0, 1.0)
    ymax = clamp(ymax, 0.0, 1.0)

    # convert to [0,1000]
    ymin_ = round(ymin * 1000.0, ndigits)
    xmin_ = round(xmin * 1000.0, ndigits)
    ymax_ = round(ymax * 1000.0, ndigits)
    xmax_ = round(xmax * 1000.0, ndigits)

    return [ymin_, xmin_, ymax_, xmax_]


def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def build_word_to_line_maps(ocr: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, str]]:
    """
    Returns:
      word_id -> line_idx
      word_id -> line_id
      line_id -> line_text
    Uses LINE blocks' Relationships CHILD Ids to map words to lines.
    """
    word_to_line_idx: Dict[str, int] = {}
    word_to_line_id: Dict[str, str] = {}
    line_id_to_text: Dict[str, str] = {}

    lines = safe_list(ocr.get("LINE"))
    for li, line in enumerate(lines):
        if not isinstance(line, dict):
            continue
        line_id = line.get("Id", None)
        if isinstance(line_id, str) and line_id:
            line_id_to_text[line_id] = str(line.get("Text", ""))

        child_ids: List[str] = []
        for rel in safe_list(line.get("Relationships")):
            if not isinstance(rel, dict):
                continue
            if rel.get("Type") == "CHILD":
                for cid in safe_list(rel.get("Ids")):
                    if isinstance(cid, str) and cid:
                        child_ids.append(cid)

        if isinstance(line_id, str) and line_id:
            for wid in child_ids:
                word_to_line_idx[wid] = li
                word_to_line_id[wid] = line_id

    return word_to_line_idx, word_to_line_id, line_id_to_text


def convert_one(
    ocr_path: Path,
    out_path: Path,
    *,
    ndigits: int = 2,
    include_line_map: bool = True,
    skip_empty_text: bool = False,
) -> Tuple[int, int]:
    """
    Convert one OCR json -> jsonl.
    Returns (n_words_written, n_words_skipped).
    """
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr = json.load(f)

    if not isinstance(ocr, dict):
        return 0, 0

    words = safe_list(ocr.get("WORD"))
    if len(words) == 0:
        return 0, 0

    word_to_line_idx: Dict[str, int] = {}
    word_to_line_id: Dict[str, str] = {}
    line_id_to_text: Dict[str, str] = {}

    if include_line_map:
        word_to_line_idx, word_to_line_id, line_id_to_text = build_word_to_line_maps(ocr)

    doc_id = ocr_path.stem
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for wi, w in enumerate(words):
            if not isinstance(w, dict):
                skipped += 1
                continue

            text = w.get("Text", "")
            if text is None:
                text = ""
            text = str(text)

            if skip_empty_text and not text.strip():
                skipped += 1
                continue

            geom = w.get("Geometry", {})
            if not isinstance(geom, dict):
                skipped += 1
                continue

            bb = geom.get("BoundingBox", None)
            bbox_1000 = bb01_to_1000_yxyx(bb, ndigits=ndigits)
            if bbox_1000 is None:
                skipped += 1
                continue

            wid = w.get("Id", None)
            wid = str(wid) if isinstance(wid, str) else None

            conf = w.get("Confidence", None)
            try:
                conf_f = float(conf) if conf is not None else None
            except Exception:
                conf_f = None

            row: Dict[str, Any] = {
                "doc_id": doc_id,
                "word_idx": wi,           # preserves original order
                "text": text,
                "confidence": conf_f,     # OCR confidence if available
                "bbox": bbox_1000,        # [ymin,xmin,ymax,xmax] in [0,1000]
            }

            if include_line_map and wid is not None:
                if wid in word_to_line_idx:
                    row["line_idx"] = word_to_line_idx[wid]
                if wid in word_to_line_id:
                    lid = word_to_line_id[wid]
                    row["line_id"] = lid
                    # line text is optional but helpful for debugging
                    if lid in line_id_to_text:
                        row["line_text"] = line_id_to_text[lid]

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    return written, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-root", required=True, help="Folder containing OCR .json files")
    ap.add_argument("--out-root", required=True, help="Folder to write cleaned word-level .jsonl files")
    ap.add_argument("--pattern", default="*.json", help="Glob pattern (default: *.json)")
    ap.add_argument("--ndigits", type=int, default=2, help="Round bbox coords to N decimals (default: 2)")
    ap.add_argument("--no-line-map", action="store_true", help="Do not map WORDs to LINEs via Relationships")
    ap.add_argument("--skip-empty-text", action="store_true", help="Skip OCR words with empty text")
    args = ap.parse_args()

    ocr_root = Path(args.ocr_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(ocr_root.glob(args.pattern))
    if not files:
        print(f"No files matched: {ocr_root} / {args.pattern}")
        return

    total_written = 0
    total_skipped = 0
    ok_files = 0
    bad_files = 0

    for p in tqdm(files, desc="Converting OCR JSON -> word JSONL", dynamic_ncols=True):
        out_path = out_root / f"{p.stem}.jsonl"
        try:
            w, s = convert_one(
                p,
                out_path,
                ndigits=args.ndigits,
                include_line_map=(not args.no_line_map),
                skip_empty_text=args.skip_empty_text,
            )
            total_written += w
            total_skipped += s
            ok_files += 1
        except Exception:
            bad_files += 1

    print("\nDone.")
    print(f"ocr_files_total={len(files)} ok_files={ok_files} bad_files={bad_files}")
    print(f"total_words_written={total_written} total_words_skipped={total_skipped}")
    print(f"out_root={out_root}")


if __name__ == "__main__":
    main()

'''
python ocr_json_to_word_jsonl.py --ocr-root ./valid_ocr --out-root ./ocr_word_jsonl --pattern *.json --ndigits 2
'''
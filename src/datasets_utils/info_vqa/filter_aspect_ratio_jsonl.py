#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter JSONL rows whose image aspect ratio is too imbalanced.

Rule:
- filter out if max(width/height, height/width) > threshold (default 4.0)

Also prints stats: total rows, kept, filtered, missing/invalid size.

Usage:
  python filter_aspect_ratio_jsonl.py \
    --input in.jsonl \
    --output out.jsonl \
    --threshold 4

Optional:
  --filtered-out filtered.jsonl   # write filtered rows for inspection
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Tuple


def get_wh(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    w = row.get("width", None)
    h = row.get("height", None)
    try:
        wf = float(w)
        hf = float(h)
        if wf <= 0 or hf <= 0:
            return None, None
        return wf, hf
    except Exception:
        return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL")
    ap.add_argument("--output", required=True, help="Output JSONL (kept rows)")
    ap.add_argument("--filtered-out", default=None, help="Optional JSONL to save filtered rows")
    ap.add_argument("--threshold", type=float, default=4.0, help="Aspect ratio threshold (default 4.0)")
    args = ap.parse_args()

    total = 0
    kept = 0
    filtered = 0
    invalid_size = 0

    fout_filt = open(args.filtered_out, "w", encoding="utf-8") if args.filtered_out else None

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                invalid_size += 1
                continue

            if not isinstance(row, dict):
                invalid_size += 1
                continue

            w, h = get_wh(row)
            if w is None or h is None:
                invalid_size += 1
                continue

            ratio = max(w / h, h / w)

            if ratio > args.threshold:
                filtered += 1
                if fout_filt is not None:
                    row_out = dict(row)
                    row_out["aspect_ratio"] = round(ratio, 6)
                    row_out["filtered_reason"] = f"ratio>{args.threshold}"
                    fout_filt.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                continue

            kept += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    if fout_filt is not None:
        fout_filt.close()

    print(
        f"Done.\n"
        f"threshold={args.threshold}\n"
        f"total={total}\n"
        f"kept={kept}\n"
        f"filtered={filtered}\n"
        f"invalid_size={invalid_size}\n"
        f"filtered_pct={(filtered / total * 100.0):.2f}%"
        if total > 0 else
        f"Done.\nthreshold={args.threshold}\ntotal=0"
    )


if __name__ == "__main__":
    main()

'''
python filter_aspect_ratio_jsonl.py --input infographicsvqa_cot_train_gemini.jsonl --output infographicsvqa_cot_train_gemini_4filtered.jsonl --threshold 4
'''
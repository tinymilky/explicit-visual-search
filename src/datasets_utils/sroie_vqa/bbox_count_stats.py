#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute statistics of how many bboxes each JSONL row has.

Assumes each row has:
- "bboxs": [[xmin,ymin,xmax,ymax], ...]   OR
- "bboxes": [{"bbox":[...]} , ...] (Gemini-style)  OR
- "bboxes": [[...], ...] (fallback)

Outputs:
- prints summary stats (mean/median/pct/quantiles)
- prints a histogram of counts (0..max_k), with optional tail bin
- optionally saves a JSON report

Usage:
  python bbox_count_stats.py --input data.jsonl
  python bbox_count_stats.py --input data.jsonl --out stats.json --max-k 30
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from tqdm import tqdm


def count_boxes(row: Dict[str, Any]) -> int:
    # Prefer bboxs (pixel xyxy)
    if isinstance(row.get("bboxs", None), list):
        return sum(1 for b in row["bboxs"] if isinstance(b, (list, tuple)) and len(b) == 4)

    # Gemini-style: bboxes can be [{"bbox":[...]}...]
    bboxes = row.get("bboxes", None)
    if isinstance(bboxes, list):
        if len(bboxes) == 0:
            return 0
        if all(isinstance(x, dict) for x in bboxes):
            return sum(1 for x in bboxes if isinstance(x.get("bbox", None), (list, tuple)) and len(x["bbox"]) == 4)
        # fallback: list of lists
        return sum(1 for x in bboxes if isinstance(x, (list, tuple)) and len(x) == 4)

    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL")
    ap.add_argument("--out", default=None, help="Optional JSON output report")
    ap.add_argument("--max-k", type=int, default=30, help="Histogram bins shown for k=0..max-k, tail grouped")
    ap.add_argument("--no-total", action="store_true", help="Do not pre-count lines for tqdm total")
    args = ap.parse_args()

    total = None
    if not args.no_total:
        try:
            with open(args.input, "rb") as f:
                total = sum(1 for _ in f)
        except Exception:
            total = None

    counts: List[int] = []
    bad_rows = 0
    rows = 0

    with open(args.input, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=total, desc="Reading JSONL", dynamic_ncols=True):
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                row = json.loads(line)
            except Exception:
                bad_rows += 1
                continue
            if not isinstance(row, dict):
                bad_rows += 1
                continue

            k = count_boxes(row)
            counts.append(int(k))

    if len(counts) == 0:
        print("No valid rows found.")
        return

    arr = np.asarray(counts, dtype=np.int64)
    ctr = Counter(arr.tolist())

    # Summary stats
    summary = {
        "rows_total": int(rows),
        "rows_valid": int(arr.size),
        "rows_bad": int(bad_rows),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "num_zero": int((arr == 0).sum()),
        "pct_zero": float((arr == 0).mean() * 100.0),
    }

    print("=== BBox count stats per row ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # Histogram 0..max_k, then tail
    max_k = max(0, int(args.max_k))
    print("\n=== Histogram (rows by bbox count) ===")
    tail = 0
    for k in range(0, max_k + 1):
        c = ctr.get(k, 0)
        print(f"{k:>3d}: {c}")
    for k, c in ctr.items():
        if k > max_k:
            tail += c
    if tail > 0:
        print(f">{max_k:>2d}: {tail}")

    # Optional JSON output
    if args.out:
        hist = {str(k): int(ctr.get(k, 0)) for k in range(0, max_k + 1)}
        if tail > 0:
            hist[f">{max_k}"] = int(tail)

        report = {
            "summary": summary,
            "histogram": hist,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Saved report: {args.out}")


if __name__ == "__main__":
    main()

'''
python bbox_count_stats.py --input sroie_cot_train.jsonl --max-k 20 --out sroie_bbox_count_stats.json
'''
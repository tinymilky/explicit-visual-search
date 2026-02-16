#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute bbox-area / image-area ratios from a JSONL file and plot a binned bar chart.

Input JSONL row example:
{"width": 592, "height": 982, "bboxs": [[57, 179, 550, 530]], ...}

Assumptions:
- bboxs are pixel coords: [xmin, ymin, xmax, ymax]
- width/height are the image size corresponding to those bbox coords

What this script can measure (choose via --mode):
- per_bbox:    one ratio per bbox (default)
- per_row_sum: one ratio per row using sum of bbox areas (clipped to 1.0)
- per_row_max: one ratio per row using the max bbox area

Histogram bins:
- quantized by step (default 0.05) from 0 to 1.0

Outputs:
- saves a barplot PNG
- prints stats and bin counts
- optionally saves bin counts JSON

Usage:
  python bbox_area_ratio_hist.py \
    --input data.jsonl \
    --plot-out ratio_hist.png \
    --step 0.05 \
    --mode per_bbox
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _fix_and_clip_box(
    b: Any, W: float, H: float
) -> Optional[Tuple[float, float, float, float]]:
    """
    b: [xmin, ymin, xmax, ymax] in pixels
    returns: (xmin, ymin, xmax, ymax) with ordering fixed and clipped to [0,W]x[0,H]
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

    xmin = _clamp(xmin, 0.0, W)
    xmax = _clamp(xmax, 0.0, W)
    ymin = _clamp(ymin, 0.0, H)
    ymax = _clamp(ymax, 0.0, H)

    return xmin, ymin, xmax, ymax


def _area_xyxy(xmin: float, ymin: float, xmax: float, ymax: float) -> float:
    w = max(0.0, xmax - xmin)
    h = max(0.0, ymax - ymin)
    return w * h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL")
    ap.add_argument("--plot-out", required=True, help="Output plot path (png/pdf/etc.)")
    ap.add_argument("--bins-out", default=None, help="Optional output JSON file with bin counts")
    ap.add_argument("--step", type=float, default=0.05, help="Bin step (default 0.05)")
    ap.add_argument(
        "--mode",
        choices=["per_bbox", "per_row_sum", "per_row_max"],
        default="per_bbox",
        help="How to aggregate ratios (default per_bbox)",
    )
    ap.add_argument("--no-total", action="store_true", help="Do not pre-count lines for tqdm total")
    args = ap.parse_args()

    if args.step <= 0 or args.step > 1:
        raise ValueError("--step must be in (0,1].")

    # bins: [0, step, 2*step, ..., 1.0]
    edges = np.arange(0.0, 1.0 + args.step + 1e-9, args.step)
    edges[-1] = 1.0  # ensure exact end
    if edges[-1] < 1.0:
        edges = np.append(edges, 1.0)

    ratios: List[float] = []

    total = None
    if not args.no_total:
        try:
            with open(args.input, "rb") as f:
                total = sum(1 for _ in f)
        except Exception:
            total = None

    rows = 0
    rows_used = 0
    rows_invalid = 0
    boxes_seen = 0
    boxes_used = 0

    with open(args.input, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=total, desc="Reading JSONL", dynamic_ncols=True):
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                row = json.loads(line)
            except Exception:
                rows_invalid += 1
                continue
            if not isinstance(row, dict):
                rows_invalid += 1
                continue

            try:
                W = float(row.get("width", None))
                H = float(row.get("height", None))
            except Exception:
                rows_invalid += 1
                continue
            if W <= 0 or H <= 0:
                rows_invalid += 1
                continue

            img_area = W * H
            raw_boxes = row.get("bboxs", []) or []
            if not isinstance(raw_boxes, list) or len(raw_boxes) == 0:
                # no boxes; skip (or you could treat as ratio=0)
                continue

            # compute areas for this row
            areas: List[float] = []
            for b in raw_boxes:
                boxes_seen += 1
                fixed = _fix_and_clip_box(b, W, H)
                if fixed is None:
                    continue
                area = _area_xyxy(*fixed)
                if area <= 0:
                    continue
                boxes_used += 1
                areas.append(area)

                if args.mode == "per_bbox":
                    r = area / img_area
                    r = float(_clamp(r, 0.0, 1.0))
                    ratios.append(r)

            if args.mode in ("per_row_sum", "per_row_max") and len(areas) > 0:
                rows_used += 1
                if args.mode == "per_row_sum":
                    r = sum(areas) / img_area
                else:
                    r = max(areas) / img_area
                r = float(_clamp(r, 0.0, 1.0))
                ratios.append(r)

    if len(ratios) == 0:
        print("No valid ratios computed. Check bboxs / width / height fields.")
        return

    # histogram: count in each bin [edges[i], edges[i+1])
    # last bin includes 1.0
    counts, _ = np.histogram(ratios, bins=edges)
    bin_left = edges[:-1]
    bin_right = edges[1:]

    # Print quick stats
    arr = np.asarray(ratios, dtype=np.float64)
    print("=== Stats ===")
    print(f"mode={args.mode} step={args.step}")
    print(f"rows_total={rows} rows_invalid={rows_invalid}")
    print(f"boxes_seen={boxes_seen} boxes_used={boxes_used}")
    print(f"ratios_count={arr.size}")
    for q in [0, 25, 50, 75, 90, 95, 99, 100]:
        v = float(np.percentile(arr, q))
        print(f"p{q:02d}={v:.6f}")

    # Barplot
    centers = (bin_left + bin_right) / 2.0
    widths = (bin_right - bin_left) * 0.95

    plt.figure()
    plt.bar(centers, counts, width=widths, align="center")
    plt.xlabel("bbox area / image area (binned)")
    plt.ylabel("count")
    plt.title(f"Area ratio histogram ({args.mode}, step={args.step})")
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=200)
    plt.close()
    print(f"[OK] Saved plot: {args.plot_out}")

    if args.bins_out:
        out = []
        for i in range(len(counts)):
            out.append(
                {
                    "bin_left": float(bin_left[i]),
                    "bin_right": float(bin_right[i]),
                    "count": int(counts[i]),
                }
            )
        with open(args.bins_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": args.mode,
                    "step": args.step,
                    "total_ratios": int(arr.size),
                    "bins": out,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[OK] Saved bin counts: {args.bins_out}")


if __name__ == "__main__":
    main()

'''
python bbox_area_ratio_hist.py --input sroie_cot_train.jsonl --plot-out sroie_bbox_area_ratio.png --step 0.05 --mode per_bbox
'''

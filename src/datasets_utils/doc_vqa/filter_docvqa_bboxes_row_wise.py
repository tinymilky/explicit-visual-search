#!/usr/bin/env python3
"""
Filter DocVQA-style jsonl samples into:
- correct.jsonl: bboxs has length 1, OR multiple boxes with high mutual overlap (mean overlap >= thresh)
- incorrect.jsonl: bboxs has length > 1 AND mean pairwise overlap < thresh

Overlap ratio here is IoU (Intersection-over-Union) between two boxes.
Rule:
- if len(bboxs) == 1 -> correct
- else compute mean IoU over all unordered pairs; if mean_iou < 0.9 -> incorrect, else correct

Usage:
  python filter_docvqa_bboxes.py \
    --input in.jsonl \
    --out-correct correct.jsonl \
    --out-incorrect incorrect.jsonl \
    --thresh 0.9
"""

import argparse
import json
import math
from itertools import combinations
from typing import List, Tuple, Optional

from tqdm import tqdm


Box = Tuple[float, float, float, float]


def to_box(b: List[float]) -> Optional[Box]:
    """Convert list-like bbox [x1,y1,x2,y2] to a valid tuple, else None."""
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    x1, y1, x2, y2 = b
    try:
        x1 = float(x1); y1 = float(y1); x2 = float(x2); y2 = float(y2)
    except Exception:
        return None
    # normalize ordering if needed
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    # degenerate boxes are allowed but will yield 0 IoU with most
    return (x1, y1, x2, y2)


def iou(a: Box, b: Box) -> float:
    """Intersection over Union of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0
    return inter / union


def mean_pairwise_iou(boxes: List[Box]) -> float:
    """Mean IoU over all unordered pairs. If <2 boxes, returns 1.0 by convention."""
    n = len(boxes)
    if n < 2:
        return 1.0
    vals = []
    for a, b in combinations(boxes, 2):
        vals.append(iou(a, b))
    return float(sum(vals) / len(vals)) if vals else 0.0


def count_lines(path: str) -> int:
    """Count lines in a file (for tqdm total)."""
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input jsonl")
    ap.add_argument("--out-correct", required=True, help="Output jsonl for correct samples")
    ap.add_argument("--out-incorrect", required=True, help="Output jsonl for incorrect samples")
    ap.add_argument("--thresh", type=float, default=0.9, help="Mean pairwise IoU threshold (default: 0.9)")
    ap.add_argument("--no-total", action="store_true", help="Do not pre-count lines for tqdm total (faster start)")
    args = ap.parse_args()

    total = None if args.no_total else count_lines(args.input)

    correct = 0
    incorrect = 0
    skipped = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.out_correct, "w", encoding="utf-8") as f_ok, \
         open(args.out_incorrect, "w", encoding="utf-8") as f_bad:

        pbar = tqdm(fin, total=total, desc="Processing", dynamic_ncols=True)
        for line in pbar:
            line = line.strip()
            if not line:
                skipped += 1
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            bboxs = obj.get("bboxs", None)
            if not isinstance(bboxs, list) or len(bboxs) == 0:
                # No boxes: treat as incorrect/problematic
                incorrect += 1
                obj["_flag_reason"] = "missing_or_empty_bboxs"
                f_bad.write(json.dumps(obj, ensure_ascii=False) + "\n")
                pbar.set_postfix_str(f"correct={correct} incorrect={incorrect} skipped={skipped}")
                continue

            if len(bboxs) == 1:
                correct += 1
                f_ok.write(json.dumps(obj, ensure_ascii=False) + "\n")
                pbar.set_postfix_str(f"correct={correct} incorrect={incorrect} skipped={skipped}")
                continue

            # Convert/clean boxes
            boxes: List[Box] = []
            for b in bboxs:
                bb = to_box(b)
                if bb is not None:
                    boxes.append(bb)

            if len(boxes) < 2:
                incorrect += 1
                obj["_flag_reason"] = "invalid_bboxs_after_parsing"
                f_bad.write(json.dumps(obj, ensure_ascii=False) + "\n")
                pbar.set_postfix_str(f"correct={correct} incorrect={incorrect} skipped={skipped}")
                continue

            m = mean_pairwise_iou(boxes)
            obj["_mean_pairwise_iou"] = m

            if m < args.thresh:
                incorrect += 1
                obj["_flag_reason"] = f"mean_pairwise_iou<{args.thresh}"
                f_bad.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                correct += 1
                f_ok.write(json.dumps(obj, ensure_ascii=False) + "\n")

            pbar.set_postfix_str(f"correct={correct} incorrect={incorrect} skipped={skipped}")

    print(f"Done.\nCorrect: {correct}\nIncorrect: {incorrect}\nSkipped: {skipped}")
    print(f"Wrote:\n  {args.out_correct}\n  {args.out_incorrect}")


if __name__ == "__main__":
    main()


'''
python filter_docvqa_bboxes_row_wise.py --input docvqa_original.jsonl --out-correct docvqa_row_wise_correct.jsonl --out-incorrect docvqa_row_wise_incorrect.jsonl --thresh 0.9
'''
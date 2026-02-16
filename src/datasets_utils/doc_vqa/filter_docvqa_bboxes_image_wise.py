#!/usr/bin/env python3
"""
Filter DocVQA-style jsonl samples into:
- correct.jsonl: image-level correct (ALL rows for that image are correct by bbox rule)
- incorrect.jsonl: image-level incorrect (ANY row for that image is incorrect by bbox rule)

Per-row bbox rule:
- if len(bboxs) == 1 -> row is correct
- else compute mean IoU over all unordered pairs; if mean_iou < thresh -> row is incorrect, else correct

Image-level rule (requested):
- if an image has at least one incorrect row -> ALL rows for that image go to incorrect.jsonl
- otherwise ALL rows for that image go to correct.jsonl

Note: requires grouping by image, so we do 2 passes over the input:
  pass1: decide which images are "bad"
  pass2: write all rows to correct/incorrect based on image
"""

import argparse
import json
from itertools import combinations
from typing import List, Tuple, Optional, Set

from tqdm import tqdm

Box = Tuple[float, float, float, float]


def to_box(b: List[float]) -> Optional[Box]:
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    x1, y1, x2, y2 = b
    try:
        x1 = float(x1); y1 = float(y1); x2 = float(x2); y2 = float(y2)
    except Exception:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def iou(a: Box, b: Box) -> float:
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
    n = len(boxes)
    if n < 2:
        return 1.0
    vals = [iou(a, b) for a, b in combinations(boxes, 2)]
    return float(sum(vals) / len(vals)) if vals else 0.0


def row_is_incorrect(obj: dict, thresh: float) -> bool:
    bboxs = obj.get("bboxs", None)
    if not isinstance(bboxs, list) or len(bboxs) == 0:
        return True

    if len(bboxs) == 1:
        return False

    boxes: List[Box] = []
    for b in bboxs:
        bb = to_box(b)
        if bb is not None:
            boxes.append(bb)

    if len(boxes) < 2:
        return True

    m = mean_pairwise_iou(boxes)
    return m < thresh


def count_lines(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input jsonl")
    ap.add_argument("--out-correct", required=True, help="Output jsonl for image-level correct samples")
    ap.add_argument("--out-incorrect", required=True, help="Output jsonl for image-level incorrect samples")
    ap.add_argument("--thresh", type=float, default=0.9, help="Mean pairwise IoU threshold (default: 0.9)")
    ap.add_argument("--no-total", action="store_true", help="Do not pre-count lines for tqdm total (faster start)")
    args = ap.parse_args()

    total = None if args.no_total else count_lines(args.input)

    # PASS 1: identify bad images (any incorrect row => image is bad)
    bad_images: Set[str] = set()
    skipped1 = 0
    with open(args.input, "r", encoding="utf-8") as fin:
        pbar1 = tqdm(fin, total=total, desc="Pass1: flag bad images", dynamic_ncols=True)
        for line in pbar1:
            line = line.strip()
            if not line:
                skipped1 += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped1 += 1
                continue

            img = obj.get("image", None)
            if not isinstance(img, str) or not img:
                # treat missing image as bad bucket keyed by None? we just skip counting as image;
                # it will be handled in pass2 as incorrect row by default.
                continue

            if img not in bad_images and row_is_incorrect(obj, args.thresh):
                bad_images.add(img)

            pbar1.set_postfix_str(f"bad_images={len(bad_images)} skipped={skipped1}")

    # PASS 2: write all rows based on image-level decision
    correct = 0
    incorrect = 0
    skipped2 = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.out_correct, "w", encoding="utf-8") as f_ok, \
         open(args.out_incorrect, "w", encoding="utf-8") as f_bad:

        pbar2 = tqdm(fin, total=total, desc="Pass2: write jsonl", dynamic_ncols=True)
        for line in pbar2:
            raw = line.strip()
            if not raw:
                skipped2 += 1
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                skipped2 += 1
                continue

            img = obj.get("image", None)

            # If image missing, fall back to row rule: incorrect rows => incorrect; else correct.
            is_bad_img = isinstance(img, str) and img in bad_images
            if is_bad_img:
                incorrect += 1
                f_bad.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                # missing/invalid image name: use row rule
                if not isinstance(img, str) or not img:
                    if row_is_incorrect(obj, args.thresh):
                        incorrect += 1
                        f_bad.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    else:
                        correct += 1
                        f_ok.write(json.dumps(obj, ensure_ascii=False) + "\n")
                else:
                    correct += 1
                    f_ok.write(json.dumps(obj, ensure_ascii=False) + "\n")

            pbar2.set_postfix_str(f"correct={correct} incorrect={incorrect} skipped={skipped2}")

    print("Done.")
    print(f"Bad images flagged: {len(bad_images)}")
    print(f"Correct rows written: {correct}")
    print(f"Incorrect rows written: {incorrect}")
    print(f"Skipped lines (pass1/pass2): {skipped1}/{skipped2}")
    print(f"Wrote:\n  {args.out_correct}\n  {args.out_incorrect}")


if __name__ == "__main__":
    main()


'''
python filter_docvqa_bboxes.py --input docvqa_cot_train.jsonl --out-correct docvqa_correct.jsonl --out-incorrect docvqa_incorrect.jsonl --thresh 0.9
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter JSONL rows by bbox area ratio, and (optionally) drop large boxes within kept rows.

Output (kept):
- bboxs contains ONLY boxes with ratio <= threshold.

Output (dropped rows file, if provided):
- keeps ORIGINAL bboxs unchanged (as requested)
- adds:
  - filtered_reason
  - dropped_bbox_indices (indices of original bboxs that were > threshold or invalid)
  - kept_bboxs_preview (optional): the bboxs that would remain after filtering (can remove if you don't want it)

Assumptions:
- bboxs are pixel coords: [xmin, ymin, xmax, ymax]
- width/height are image size in pixels matching bbox coords
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _fix_and_clip_box(b: Any, W: float, H: float) -> Optional[Tuple[float, float, float, float]]:
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
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL")
    ap.add_argument("--output", required=True, help="Output JSONL (kept rows with bboxs<=threshold)")
    ap.add_argument("--threshold", type=float, default=0.2, help="Max allowed bbox area ratio (default 0.2)")
    ap.add_argument("--keep-empty-rows", action="store_true", help="Keep rows even if all bboxs are dropped")
    ap.add_argument("--dropped-rows", default=None, help="Optional JSONL to save rows dropped due to empty bboxs")
    ap.add_argument("--log-kept-box-ratios", action="store_true", help="Add kept_bbox_ratios to output rows")
    ap.add_argument("--add-kept-preview-to-dropped", action="store_true",
                    help="In dropped-rows file, also add kept_bboxs_preview after filtering")
    args = ap.parse_args()

    if args.threshold <= 0 or args.threshold > 1.0:
        raise ValueError("--threshold must be in (0,1].")

    total = 0
    bad = 0
    kept_rows = 0
    dropped_rows = 0
    rows_missing_size = 0

    boxes_in = 0
    boxes_kept = 0
    boxes_dropped = 0

    fout_drop = open(args.dropped_rows, "w", encoding="utf-8") if args.dropped_rows else None

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                row = json.loads(line)
            except Exception:
                bad += 1
                continue
            if not isinstance(row, dict):
                bad += 1
                continue

            try:
                W = float(row.get("width", None))
                H = float(row.get("height", None))
            except Exception:
                rows_missing_size += 1
                continue
            if W <= 0 or H <= 0:
                rows_missing_size += 1
                continue

            img_area = W * H
            raw = row.get("bboxs", []) or []
            if not isinstance(raw, list):
                raw = []

            original_bboxs = raw  # keep exactly as-is for dropped file

            kept_bboxs: List[List[float]] = []
            kept_ratios: List[float] = []
            dropped_idx: List[int] = []

            for idx, b in enumerate(raw):
                boxes_in += 1
                fixed = _fix_and_clip_box(b, W, H)
                if fixed is None:
                    boxes_dropped += 1
                    dropped_idx.append(idx)
                    continue
                area = _area_xyxy(*fixed)
                ratio = 0.0 if img_area <= 0 else (area / img_area)

                if ratio > args.threshold:
                    boxes_dropped += 1
                    dropped_idx.append(idx)
                    continue

                kept_bboxs.append([float(x) for x in b])
                kept_ratios.append(float(ratio))
                boxes_kept += 1

            out_row = dict(row)
            out_row["bboxs"] = kept_bboxs
            if args.log_kept_box_ratios:
                out_row["kept_bbox_ratios"] = [round(r, 6) for r in kept_ratios]

            # drop row if no boxes left (unless keep-empty-rows)
            if len(kept_bboxs) == 0 and not args.keep_empty_rows:
                dropped_rows += 1
                if fout_drop is not None:
                    drop_row = dict(row)
                    drop_row["bboxs"] = original_bboxs  # <-- KEEP ORIGINAL bboxs
                    drop_row["filtered_reason"] = f"all bboxs ratio>{args.threshold} (or invalid)"
                    drop_row["dropped_bbox_indices"] = dropped_idx
                    if args.add_kept_preview_to_dropped:
                        drop_row["kept_bboxs_preview"] = kept_bboxs
                    fout_drop.write(json.dumps(drop_row, ensure_ascii=False) + "\n")
                continue

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            kept_rows += 1

    if fout_drop is not None:
        fout_drop.close()

    print("Done.")
    print(f"threshold={args.threshold}")
    print(f"rows_total={total}")
    print(f"rows_kept={kept_rows}")
    print(f"rows_dropped={dropped_rows}")
    print(f"rows_bad_json={bad}")
    print(f"rows_missing_or_invalid_size={rows_missing_size}")
    print(f"boxes_in={boxes_in}")
    print(f"boxes_kept={boxes_kept}")
    print(f"boxes_dropped={boxes_dropped}")
    if total > 0:
        print(f"rows_dropped_pct={(dropped_rows / total) * 100.0:.2f}%")
    print(f"output={args.output}")
    if args.dropped_rows:
        print(f"dropped_rows_file={args.dropped_rows}")


if __name__ == "__main__":
    main()


'''
python filter_bbox_ratio.py --input sroie_cot_train.jsonl --output sroie_cot_train_bboxratio_le_0p1.jsonl --threshold 0.1 --dropped-rows sroie_cot_train_dropped_rows.jsonl --add-kept-preview-to-dropped
'''

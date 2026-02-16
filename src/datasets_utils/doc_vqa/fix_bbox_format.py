#!/usr/bin/env python3
"""
Fix mixed bbox formats in Gemini-annotated JSONL and add an alternate ordering.

Input row example may contain bbox as:
- list: [ymin, xmin, ymax, xmax]
- dict: {"ymin":..., "xmin":..., "ymax":..., "xmax":...}

This script:
1) Scans each row's "bboxes" list and converts every bbox to list style [ymin, xmin, ymax, xmax].
2) Keeps the original "bboxes" key (now normalized to list style).
3) Appends a new key "bboxes_yminxminymaxxman" (requested name) whose bbox order is [xmin, ymin, xmax, ymax],
   with other fields (role/label/etc.) preserved per entry.

Usage:
  python fix_bbox_format.py \
    --input docvqa_row_wise_incorrect_annotated.jsonl \
    --output docvqa_row_wise_incorrect_annotated_fixed.jsonl
"""

import argparse
import json
from typing import Any, Dict, List, Optional


def bbox_to_list_style(b: Any) -> Optional[List[float]]:
    """Return bbox as [ymin, xmin, ymax, xmax] or None if invalid."""
    if isinstance(b, (list, tuple)) and len(b) == 4:
        try:
            return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
        except Exception:
            return None
    if isinstance(b, dict):
        keys = ["ymin", "xmin", "ymax", "xmax"]
        if all(k in b for k in keys):
            try:
                return [float(b["ymin"]), float(b["xmin"]), float(b["ymax"]), float(b["xmax"])]
            except Exception:
                return None
    return None


def reorder_to_xminyminxmaxymax(b_yxminyxmax: List[float]) -> List[float]:
    """Convert [ymin, xmin, ymax, xmax] -> [xmin, ymin, xmax, ymax]."""
    ymin, xmin, ymax, xmax = b_yxminyxmax
    return [xmin, ymin, xmax, ymax]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input annotated jsonl")
    ap.add_argument("--output", required=True, help="Output jsonl with fixed bbox formats")
    args = ap.parse_args()

    n_total = 0
    n_rows_with_bboxes = 0
    n_fixed_boxes = 0
    n_invalid_boxes = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            n_total += 1
            bboxes = row.get("bboxes", None)
            if not isinstance(bboxes, list):
                # still write through, but ensure the new key exists consistently
                row["bboxes_yminxminymaxxman"] = []
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            n_rows_with_bboxes += 1

            fixed_list: List[Dict[str, Any]] = []
            alt_list: List[Dict[str, Any]] = []

            for item in bboxes:
                if not isinstance(item, dict):
                    n_invalid_boxes += 1
                    continue

                raw_bbox = item.get("bbox", None)
                bbox_list = bbox_to_list_style(raw_bbox)
                if bbox_list is None:
                    n_invalid_boxes += 1
                    continue

                # normalize original entry to list style
                new_item = dict(item)
                new_item["bbox"] = bbox_list
                fixed_list.append(new_item)
                n_fixed_boxes += 1

                # create reordered copy
                alt_item = dict(new_item)
                alt_item["bbox"] = reorder_to_xminyminxmaxymax(bbox_list)
                alt_list.append(alt_item)

            row["bboxes"] = fixed_list
            row["bboxes_yminxminymaxxman"] = alt_list  # requested key name (kept as-is)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"Rows processed:           {n_total}")
    print(f"Rows with bboxes list:    {n_rows_with_bboxes}")
    print(f"Valid boxes converted:    {n_fixed_boxes}")
    print(f"Invalid/ignored boxes:    {n_invalid_boxes}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

'''
python fix_bbox_format.py --input docvqa_hard.jsonl --output docvqa_hard_fixed.jsonl
'''
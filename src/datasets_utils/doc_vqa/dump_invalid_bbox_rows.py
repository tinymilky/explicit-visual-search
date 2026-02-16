#!/usr/bin/env python3
"""
Split an annotated JSONL into:
- correct_rows.jsonl: rows whose bboxes are all valid
- incorrect_rows.jsonl: rows with any invalid bbox entry (or missing/non-list bboxes)

Validity rules for each bbox entry in row["bboxes"]:
- each bbox item must be a dict
- item["bbox"] must be either:
  * list/tuple of length 4 with numeric values, OR
  * dict with keys ymin/xmin/ymax/xmax with numeric values

Output rows are written as-is (no modification).
Optionally, incorrect rows get debug fields describing what was invalid.

Usage:
  python split_valid_invalid_rows.py \
    --input docvqa_row_wise_incorrect_annotated.jsonl \
    --out-correct correct_rows.jsonl \
    --out-incorrect incorrect_rows.jsonl \
    --add-debug
"""

import argparse
import json
from typing import Any, Dict, List, Tuple


def bbox_is_valid(b: Any) -> Tuple[bool, str]:
    if isinstance(b, (list, tuple)):
        if len(b) != 4:
            return False, "bbox_list_len!=4"
        try:
            _ = [float(x) for x in b]
            return True, ""
        except Exception:
            return False, "bbox_list_non_numeric"

    if isinstance(b, dict):
        keys = ["ymin", "xmin", "ymax", "xmax"]
        if not all(k in b for k in keys):
            return False, "bbox_dict_missing_keys"
        try:
            _ = float(b["ymin"]); _ = float(b["xmin"]); _ = float(b["ymax"]); _ = float(b["xmax"])
            return True, ""
        except Exception:
            return False, "bbox_dict_non_numeric"

    return False, "bbox_wrong_type"


def validate_row(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Returns (is_valid, reasons)
    is_valid=True iff row['bboxes'] exists, is a list, and all entries are valid.
    """
    bboxes = row.get("bboxes", None)
    if not isinstance(bboxes, list):
        return False, ["bboxes_not_a_list_or_missing"]

    reasons: List[str] = []
    for item in bboxes:
        if not isinstance(item, dict):
            reasons.append("bbox_item_not_dict")
            continue
        ok, reason = bbox_is_valid(item.get("bbox", None))
        if not ok:
            reasons.append(reason)

    return (len(reasons) == 0), reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input annotated jsonl")
    ap.add_argument("--out-correct", required=True, help="Output jsonl for valid rows")
    ap.add_argument("--out-incorrect", required=True, help="Output jsonl for invalid rows")
    ap.add_argument("--add-debug", action="store_true", help="Add _invalid_bbox_reasons to incorrect rows")
    args = ap.parse_args()

    total = 0
    n_ok = 0
    n_bad = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.out_correct, "w", encoding="utf-8") as f_ok, \
         open(args.out_incorrect, "w", encoding="utf-8") as f_bad:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            total += 1
            is_ok, reasons = validate_row(row)
            if is_ok:
                n_ok += 1
                f_ok.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                n_bad += 1
                if args.add_debug:
                    row = dict(row)
                    row["_invalid_bbox_reasons"] = reasons
                f_bad.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"Rows processed: {total}")
    print(f"Correct rows:   {n_ok}")
    print(f"Incorrect rows: {n_bad}")
    print(f"Wrote:\n  {args.out_correct}\n  {args.out_incorrect}")


if __name__ == "__main__":
    main()


'''
python dump_invalid_bbox_rows.py --input docvqa_row_wise_incorrect_annotated.jsonl --out-correct correct_rows.jsonl --out-incorrect incorrect_rows.jsonl --add-debug
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split an annotated JSONL into two JSONLs: success vs fail.

Success definition (matches the annotate_infovqa_gemini.py refined output):
- answer_type in {"extractive","semantic"}
- bboxes is a list
- each bbox is a dict with keys: role, label, bbox, is_semantic
- bbox is [ymin,xmin,ymax,xmax] (len=4 numeric)

Everything else is considered FAIL.

Usage:
  python split_infovqa_success_fail.py \
    --input infographicsvqa_cot_train_gemini_4filtered_part2_annotated.jsonl \
    --ok-out infographicsvqa_part2_ok.jsonl \
    --fail-out infographicsvqa_part2_fail.jsonl
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict


def is_refined_bbox(bb: Any) -> bool:
    if not isinstance(bb, dict):
        return False
    for k in ("role", "label", "bbox", "is_semantic"):
        if k not in bb:
            return False
    b = bb.get("bbox", None)
    if not (isinstance(b, list) and len(b) == 4):
        return False
    if not all(isinstance(x, (int, float)) for x in b):
        return False
    return True


def is_success_row(row: Dict[str, Any]) -> bool:
    at = row.get("answer_type", None)
    if at not in ("extractive", "semantic"):
        return False

    bboxes = row.get("bboxes", None)
    if not isinstance(bboxes, list):
        return False

    # require all bboxes to be refined dicts
    return all(is_refined_bbox(bb) for bb in bboxes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input annotated JSONL")
    ap.add_argument("--ok-out", required=True, help="Output JSONL for successful rows")
    ap.add_argument("--fail-out", required=True, help="Output JSONL for failed rows")
    ap.add_argument("--add-line-no", action="store_true", help="Add line_no to each output row")
    args = ap.parse_args()

    total = 0
    ok = 0
    fail = 0
    parse_err = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.ok_out, "w", encoding="utf-8") as fok, \
         open(args.fail_out, "w", encoding="utf-8") as fbad:

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                row = json.loads(line)
            except Exception:
                parse_err += 1
                obj = {"parse_error": True, "raw_line": line}
                if args.add_line_no:
                    obj["line_no"] = line_no
                fbad.write(json.dumps(obj, ensure_ascii=False) + "\n")
                fail += 1
                continue

            if not isinstance(row, dict):
                obj = {"parse_error": True, "raw_line": line}
                if args.add_line_no:
                    obj["line_no"] = line_no
                fbad.write(json.dumps(obj, ensure_ascii=False) + "\n")
                fail += 1
                continue

            if args.add_line_no:
                row = dict(row)
                row["line_no"] = line_no

            if is_success_row(row):
                fok.write(json.dumps(row, ensure_ascii=False) + "\n")
                ok += 1
            else:
                fbad.write(json.dumps(row, ensure_ascii=False) + "\n")
                fail += 1

    print("Done.")
    print(f"total={total} ok={ok} fail={fail} parse_err={parse_err}")
    print(f"ok_out={args.ok_out}")
    print(f"fail_out={args.fail_out}")


if __name__ == "__main__":
    main()


'''
python find_infovqa_fail_rows.py --input .\infographicsvqa_cot_train_gemini_4filtered_part2_annotated.jsonl --ok-out .\infographicsvqa_part2_ok.jsonl --fail-out .\infographicsvqa_part2_fail.jsonl --add-line-no
python find_infovqa_fail_rows.py --input .\infographicsvqa_cot_train_gemini_4filtered_part1_annotated.jsonl --ok-out .\infographicsvqa_part1_ok.jsonl --fail-out .\infographicsvqa_part1_fail.jsonl --add-line-no

'''
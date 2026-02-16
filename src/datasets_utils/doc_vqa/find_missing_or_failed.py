#!/usr/bin/env python3
"""
Find missing or failed annotations.

- Base file: docvqa_row_wise_incorrect.jsonl  (source of truth; keep its row format)
- Annotated file: docvqa_row_wise_incorrect_annotated.jsonl
- Output: rows from base that are either:
  (a) missing entirely from annotated, OR
  (b) present but failed: annotated has no usable bboxes (missing/empty)

Matching key:
  (image, question, answer)  # because one image can have multiple questions

"Failed" definition (annotated row):
  - "bboxes" key missing OR not a list OR len(bboxes)==0

Usage:
  python find_missing_or_failed.py \
    --base docvqa_row_wise_incorrect.jsonl \
    --annotated docvqa_row_wise_incorrect_annotated.jsonl \
    --output docvqa_row_wise_incorrect_todo.jsonl
"""

import argparse
import json
from typing import Dict, Tuple, Set, Any


Key = Tuple[str, str, str]  # (image, question, answer)


def norm_str(x: Any) -> str:
    # minimal normalization to reduce trivial mismatches
    if x is None:
        return ""
    s = str(x)
    return " ".join(s.split()).strip()


def make_key(obj: dict) -> Key:
    return (
        norm_str(obj.get("image", "")),
        norm_str(obj.get("question", "")),
        norm_str(obj.get("answer", "")),
    )


def is_failed_annot(obj: dict) -> bool:
    b = obj.get("bboxes", None)
    return (not isinstance(b, list)) or (len(b) == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="docvqa_row_wise_incorrect.jsonl")
    ap.add_argument("--annotated", required=True, help="docvqa_row_wise_incorrect_annotated.jsonl")
    ap.add_argument("--output", required=True, help="Output jsonl of missing/failed rows (base format)")
    args = ap.parse_args()

    # Read annotated: track which keys exist, and which are failed.
    annotated_all: Set[Key] = set()
    annotated_failed: Set[Key] = set()
    annotated_total = 0

    with open(args.annotated, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            annotated_total += 1
            k = make_key(obj)
            annotated_all.add(k)
            if is_failed_annot(obj):
                annotated_failed.add(k)

    # Scan base; write rows that are missing or failed.
    base_total = 0
    missing = 0
    failed = 0
    written = 0

    with open(args.base, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            base_total += 1

            k = make_key(obj)
            if k not in annotated_all:
                missing += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
            elif k in annotated_failed:
                failed += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1

    # Stats
    missing_or_failed = missing + failed
    pct = (missing_or_failed / base_total * 100.0) if base_total else 0.0

    print("==== Stats ====")
    print(f"Base total rows:              {base_total}")
    print(f"Annotated file rows read:     {annotated_total}")
    print(f"Missing rows (not annotated): {missing}")
    print(f"Failed rows (empty bboxes):   {failed}")
    print(f"Missing+Failed:               {missing_or_failed} ({pct:.2f}%)")
    print(f"Written to output:            {written}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

'''
python find_missing_or_failed.py \
--base docvqa_row_wise_incorrect_todo.jsonl \
--annotated docvqa_row_wise_incorrect_todo_annotated.jsonl \
--output docvqa_row_wise_incorrect_todo_annotated_todo.jsonl
'''
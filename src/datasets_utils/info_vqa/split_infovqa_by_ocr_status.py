#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split an InfoVQA annotated JSONL into:
- ok.jsonl: rows where NO bbox has ocr_status == "failed"
- failed.jsonl: rows where at least one bbox has ocr_status == "failed"

Notes:
- Looks for bbox dicts under row["bboxes"] (list of dicts).
- If a row has no "bboxes" field (or it's not a list), it is treated as OK by default.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm


def row_has_failed_bbox(row: Dict[str, Any]) -> bool:
    bxs = row.get("bboxes", None)
    if not isinstance(bxs, list):
        return False
    for b in bxs:
        if isinstance(b, dict) and str(b.get("ocr_status", "")).lower() == "failed":
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input annotated JSONL")
    ap.add_argument("--ok-out", required=True, help="Output JSONL with only OK rows")
    ap.add_argument("--fail-out", required=True, help="Output JSONL with failed rows")
    ap.add_argument("--keep-bad-json", action="store_true",
                    help="If set, write unparsable JSON lines to fail-out instead of skipping.")
    args = ap.parse_args()

    in_path = Path(args.input)
    ok_out = Path(args.ok_out)
    fail_out = Path(args.fail_out)
    ok_out.parent.mkdir(parents=True, exist_ok=True)
    fail_out.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_ok = 0
    n_fail = 0
    n_bad = 0

    # optional total for tqdm
    try:
        with open(in_path, "rb") as f:
            total = sum(1 for _ in f)
    except Exception:
        total = None

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(ok_out, "w", encoding="utf-8") as fok, \
         open(fail_out, "w", encoding="utf-8") as ffail:

        for line in tqdm(fin, total=total, desc="Splitting JSONL", dynamic_ncols=True):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            n_total += 1

            try:
                row = json.loads(line)
            except Exception:
                n_bad += 1
                if args.keep_bad_json:
                    ffail.write(line + "\n")
                    n_fail += 1
                continue

            if isinstance(row, dict) and row_has_failed_bbox(row):
                ffail.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_fail += 1
            else:
                fok.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_ok += 1

    print("Done.")
    print(f"rows_total={n_total} ok={n_ok} failed={n_fail} bad_json={n_bad}")
    print(f"ok_out={ok_out}")
    print(f"fail_out={fail_out}")


if __name__ == "__main__":
    main()

'''
python split_infovqa_by_ocr_status.py --input infovqa_gemini.jsonl --ok-out infovqa_gemini_ok.jsonl --fail-out infovqa_gemini_failed.jsonl

'''
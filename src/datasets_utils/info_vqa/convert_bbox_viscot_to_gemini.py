#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert InfoVQA-style pixel bboxes to Gemini-style normalized bboxes.

Input JSONL row fields:
- width, height: original image size (pixels)
- bboxs: list of [xmin, ymin, xmax, ymax] in pixel coords (floats ok)

Output JSONL:
- bboxes: list of dicts with "bbox" in Gemini format [ymin, xmin, ymax, xmax] scaled to [0,1000]
- coordinates rounded to at most 2 decimals
- keeps original fields by default, optionally drops bboxs

Example:
  python convert_bbox_to_gemini.py \
    --input infovqa.jsonl \
    --output infovqa_gemini.jsonl \
    --keep-bboxs
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional


def _round2(x: float) -> float:
    return round(float(x) + 1e-12, 2)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def convert_one_box(
    box_xyxy: List[Any],
    W: float,
    H: float,
) -> Optional[List[float]]:
    """
    box_xyxy: [xmin, ymin, xmax, ymax] in pixels (original image scale)
    returns:  [ymin, xmin, ymax, xmax] in [0,1000], rounded to 2 decimals
    """
    if not isinstance(box_xyxy, (list, tuple)) or len(box_xyxy) != 4:
        return None
    if W <= 0 or H <= 0:
        return None

    xmin, ymin, xmax, ymax = map(float, box_xyxy)

    # normalize ordering just in case
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin

    # clamp to image bounds
    xmin = _clamp(xmin, 0.0, W)
    xmax = _clamp(xmax, 0.0, W)
    ymin = _clamp(ymin, 0.0, H)
    ymax = _clamp(ymax, 0.0, H)

    # scale to [0,1000]
    xmin_n = (xmin / W) * 1000.0
    xmax_n = (xmax / W) * 1000.0
    ymin_n = (ymin / H) * 1000.0
    ymax_n = (ymax / H) * 1000.0

    # clamp again to [0,1000]
    xmin_n = _clamp(xmin_n, 0.0, 1000.0)
    xmax_n = _clamp(xmax_n, 0.0, 1000.0)
    ymin_n = _clamp(ymin_n, 0.0, 1000.0)
    ymax_n = _clamp(ymax_n, 0.0, 1000.0)

    return [_round2(ymin_n), _round2(xmin_n), _round2(ymax_n), _round2(xmax_n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL")
    ap.add_argument("--output", required=True, help="Output JSONL")
    ap.add_argument("--keep-bboxs", action="store_true", help="Keep original 'bboxs' in output")
    ap.add_argument("--bboxs-key", default="bboxs", help="Input bbox key (default: bboxs)")
    ap.add_argument("--out-key", default="bboxes", help="Output bbox key (default: bboxes)")
    args = ap.parse_args()

    n_in = 0
    n_out = 0
    n_box_in = 0
    n_box_out = 0
    n_bad = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                row = json.loads(line)
            except Exception:
                n_bad += 1
                continue

            W = row.get("width", None)
            H = row.get("height", None)
            try:
                Wf = float(W)
                Hf = float(H)
            except Exception:
                Wf, Hf = -1.0, -1.0

            raw = row.get(args.bboxs_key, []) or []
            if not isinstance(raw, list):
                raw = []

            out_boxes: List[Dict[str, Any]] = []
            for b in raw:
                n_box_in += 1
                gem = convert_one_box(b, Wf, Hf)
                if gem is None:
                    continue
                n_box_out += 1
                out_boxes.append({"bbox": gem})

            row[args.out_key] = out_boxes
            if not args.keep_bboxs and args.bboxs_key in row:
                row.pop(args.bboxs_key, None)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(
        f"Done.\n"
        f"rows_in={n_in} rows_out={n_out} bad_rows={n_bad}\n"
        f"boxes_in={n_box_in} boxes_out={n_box_out}"
    )


if __name__ == "__main__":
    main()

'''
python convert_bbox_to_gemini.py --input infographicsvqa_cot_train.jsonl --output infographicsvqa_cot_train_gemini.jsonl
'''
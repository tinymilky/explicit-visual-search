#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pretty-structure + pretty-print the rows in a JSONL (video-level records) into
human-readable JSON.

Revision:
- Each clip uses "st_ed_time": [start_time, end_time] instead of separate fields.

Schema for each clip:
{
  "clip_id": 4,
  "st_ed_time": [8, 10],
  "main_caption": "...",
  "additional_details": "...",
  "short_summary": "..."
}

Other behavior unchanged:
- Reads input JSONL streaming.
- Writes output streaming:
  (A) one pretty .json per row in --out_dir, and/or
  (B) a single pretty JSON array file via --out_json (streamed).
- Two tqdm bars: rows processed, clips processed.
- Optional stripping of label prefixes and optional clip sorting.

Usage:
  python pretty_structured_json.py --in_jsonl matched.jsonl --out_dir pretty_rows/ --strip_prefix --sort_clips
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Optional faster JSON
try:
    import orjson  # type: ignore

    def json_loads(line: str) -> Any:
        return orjson.loads(line.encode("utf-8"))

    def json_dumps_pretty(obj: Any) -> str:
        return orjson.dumps(
            obj,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
        ).decode("utf-8")

except Exception:
    import json

    def json_loads(line: str) -> Any:
        return json.loads(line)

    def json_dumps_pretty(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


LABEL_PATTERNS = {
    "main_caption": re.compile(r"^\s*main\s*caption\s*:\s*", re.IGNORECASE),
    "additional_details": re.compile(r"^\s*additional\s*details\s*:\s*", re.IGNORECASE),
    "short_summary": re.compile(r"^\s*short\s*summary\s*:\s*", re.IGNORECASE),
}


def norm_filename(name: str) -> str:
    base = os.path.basename(name.strip())
    while base.startswith("--"):
        base = base[2:]
    return base


def strip_label(field: str, text: Any, do_strip: bool) -> Any:
    if not do_strip or not isinstance(text, str):
        return text
    pat = LABEL_PATTERNS.get(field)
    if not pat:
        return text.strip()
    return pat.sub("", text).strip()


def to_number_if_possible(x: Any) -> Any:
    """
    Preserve ints if possible, else floats if possible, else return original.
    This helps make st_ed_time look clean: [0, 2] not ["0", "2"].
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        # Normalize floats that are actually ints (e.g., 2.0 -> 2)
        if isinstance(x, float) and x.is_integer():
            return int(x)
        return x
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return x
        # Try int
        try:
            if re.fullmatch(r"[+-]?\d+", s):
                return int(s)
        except Exception:
            pass
        # Try float
        try:
            f = float(s)
            if f.is_integer():
                return int(f)
            return f
        except Exception:
            return x
    return x


def build_structured_clip(c: Dict[str, Any], strip_prefix: bool) -> Dict[str, Any]:
    st = to_number_if_possible(c.get("start_time"))
    ed = to_number_if_possible(c.get("end_time"))

    out = {
        "clip_id": to_number_if_possible(c.get("clip_id")),
        "st_ed_time": [st, ed],
        "main_caption": strip_label("main_caption", c.get("main_caption"), strip_prefix),
        "additional_details": strip_label("additional_details", c.get("additional_details"), strip_prefix),
        "short_summary": strip_label("short_summary", c.get("short_summary"), strip_prefix),
    }
    return out


def build_structured_row(row: Dict[str, Any], strip_prefix: bool, sort_clips: bool) -> Dict[str, Any]:
    clips_raw = row.get("clips", [])
    if not isinstance(clips_raw, list):
        clips_raw = []

    clips_out: List[Dict[str, Any]] = []
    for c in clips_raw:
        if isinstance(c, dict):
            clips_out.append(build_structured_clip(c, strip_prefix))

    if sort_clips:
        def _key(x: Dict[str, Any]) -> Tuple[int, float]:
            cid = x.get("clip_id")
            try:
                cid_i = int(cid)
            except Exception:
                cid_i = 10**18
            st = None
            if isinstance(x.get("st_ed_time"), list) and x["st_ed_time"]:
                st = x["st_ed_time"][0]
            try:
                st_f = float(st)
            except Exception:
                st_f = float("inf")
            return (cid_i, st_f)

        clips_out.sort(key=_key)

    structured = {
        "idx": row.get("idx"),
        "filename": row.get("filename"),
        "path": row.get("path"),
        "num_clips": row.get("num_clips"),
        "video_stream_path": row.get("video_stream_path"),
        "clips": clips_out,
    }
    return structured


def indent_block(s: str, prefix: str = "  ") -> str:
    return "\n".join((prefix + line) if line.strip() else line for line in s.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="Input JSONL (one video record per line).")
    ap.add_argument("--out_dir", default=None, help="Output directory to write one pretty .json per row.")
    ap.add_argument("--out_json", default=None, help="Output pretty JSON array file (streamed).")
    ap.add_argument(
        "--strip_prefix",
        action="store_true",
        help='Strip leading labels like "Main caption:" from text fields.',
    )
    ap.add_argument(
        "--sort_clips",
        action="store_true",
        help="Sort clips by (clip_id, start_time) for stable reading.",
    )
    ap.add_argument(
        "--name_by",
        choices=["idx", "filename", "idx_filename"],
        default="idx_filename",
        help="How to name per-row JSON files in --out_dir.",
    )
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    if args.out_dir is None and args.out_json is None:
        raise SystemExit("You must provide at least one of --out_dir or --out_json.")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    out_json_path = Path(args.out_json) if args.out_json else None
    if out_json_path:
        out_json_path.parent.mkdir(parents=True, exist_ok=True)

    fout_array = None
    wrote_any = False
    if out_json_path:
        fout_array = out_json_path.open("w", encoding="utf-8", newline="")
        fout_array.write("[\n")

    rows_bar = tqdm(desc="rows processed", unit="row")
    clips_bar = tqdm(desc="clips processed", unit="clip", leave=False)

    try:
        with in_path.open("r", encoding="utf-8", newline="") as fin:
            for line in fin:
                if not line.strip():
                    continue

                try:
                    obj = json_loads(line)
                except Exception:
                    continue

                if not isinstance(obj, dict):
                    continue

                structured = build_structured_row(obj, args.strip_prefix, args.sort_clips)

                rows_bar.update(1)
                clips_bar.update(len(structured.get("clips", [])))

                # (A) Per-row pretty JSON file
                if out_dir:
                    idx = structured.get("idx")
                    fn = structured.get("filename") or ""
                    base = norm_filename(str(fn)) if fn else "no_filename"

                    if args.name_by == "idx":
                        stem = f"{int(idx):06d}" if isinstance(idx, int) else str(idx)
                    elif args.name_by == "filename":
                        stem = base
                    else:
                        idx_part = f"{int(idx):06d}" if isinstance(idx, int) else str(idx)
                        stem = f"{idx_part}_{base}"

                    stem = re.sub(r"[^\w\-.]+", "_", stem)
                    out_file = out_dir / f"{stem}.json"
                    out_file.write_text(json_dumps_pretty(structured) + "\n", encoding="utf-8")

                # (B) Single streamed pretty JSON array
                if fout_array is not None:
                    pretty = json_dumps_pretty(structured)
                    if wrote_any:
                        fout_array.write(",\n")
                    fout_array.write(indent_block(pretty, prefix="  "))
                    wrote_any = True

    finally:
        rows_bar.close()
        clips_bar.close()
        if fout_array is not None:
            fout_array.write("\n]\n")
            fout_array.close()

    if out_dir:
        print(f"Per-row pretty JSON written to: {out_dir}")
    if out_json_path:
        print(f"Pretty JSON array written to: {out_json_path}")


if __name__ == "__main__":
    main()


'''
python pretty_structured_json.py --in_jsonl matched.jsonl --out_dir matched_pretty/ --strip_prefix --sort_clips

python pretty_structured_json.py --in_jsonl matched.jsonl --out_json matched_pretty.json --strip_prefix --sort_clips

'''
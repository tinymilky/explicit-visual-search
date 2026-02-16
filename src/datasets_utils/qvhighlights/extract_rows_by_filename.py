#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract matching rows (by video filename) from multiple JSONL files and write them
to a new JSONL in streaming mode.

Key behavior:
- Matches on a normalized filename that strips a leading "--" (repeatedly) from BOTH
  the target list and each row's filename/path basename.
- Writes matched rows immediately (streaming write).
- Shows two tqdm progress bars: scanned rows, saved rows.
- By default, saves at most one row per target (deduplicated by normalized filename)
  and stops early once all targets are found.

Example:
  python extract_rows_by_filename.py \
    --inputs a.jsonl b.jsonl c.jsonl d.jsonl e.jsonl \
    --targets_txt targets.txt \
    --out matched.jsonl

You can also pass a directory + pattern:
  python extract_rows_by_filename.py --input_dir /path/to/jsonls --pattern "*.jsonl" --targets_txt targets.txt --out matched.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm

# Optional faster JSON
try:
    import orjson  # type: ignore

    def json_loads(b: bytes):
        return orjson.loads(b)

    def json_dumps(obj) -> str:
        # orjson.dumps returns bytes
        return orjson.dumps(obj).decode("utf-8")

except Exception:
    import json

    def json_loads(b: bytes):
        return json.loads(b.decode("utf-8"))

    def json_dumps(obj) -> str:
        return json.dumps(obj, ensure_ascii=False)


# If you don't pass --targets_txt/--targets, the script will use this embedded list.
DEFAULT_TARGETS = [
    "NUsG9BgSes0_210.0_360.0.mp4",
    "NUsG9BgSes0_60.0_210.0.mp4",
    "NUsG9BgSes0_360.0_510.0.mp4",
    "NUsG9BgSes0_660.0_810.0.mp4",
    "NUsG9BgSes0_510.0_660.0.mp4",
    "bP5KfdFJzC4_60.0_210.0.mp4",
    "bP5KfdFJzC4_210.0_360.0.mp4",
    "bP5KfdFJzC4_360.0_510.0.mp4",
    "bP5KfdFJzC4_660.0_810.0.mp4",
    "bP5KfdFJzC4_510.0_660.0.mp4",
    "nY42UppPhhg_60.0_210.0.mp4",
    "nY42UppPhhg_210.0_360.0.mp4",
    "nY42UppPhhg_360.0_510.0.mp4",
    "RoripwjYFp8_210.0_360.0.mp4",
    "RoripwjYFp8_360.0_510.0.mp4",
    "RoripwjYFp8_60.0_210.0.mp4",
    "r7A-cfBq2Xw_210.0_360.0.mp4",
    "r7A-cfBq2Xw_360.0_510.0.mp4",
    "r7A-cfBq2Xw_60.0_210.0.mp4",
    "pA6Z-qYhSNg_60.0_210.0.mp4",
    "pA6Z-qYhSNg_360.0_510.0.mp4",
    "pA6Z-qYhSNg_210.0_360.0.mp4",
    "zVwsEVwS8Kw_210.0_360.0.mp4",
    "zVwsEVwS8Kw_60.0_210.0.mp4",
    "zVwsEVwS8Kw_360.0_510.0.mp4",
    "YIUaJzjNPuo_360.0_510.0.mp4",
    "YIUaJzjNPuo_60.0_210.0.mp4",
    "YIUaJzjNPuo_210.0_360.0.mp4",
    "GAUdBAL0K5A_210.0_360.0.mp4",
    "GAUdBAL0K5A_60.0_210.0.mp4",
    "GAUdBAL0K5A_360.0_510.0.mp4",
    "jv7033VUyHE_60.0_210.0.mp4",
    "yId2wIocTys_60.0_210.0.mp4",
    "yId2wIocTys_210.0_360.0.mp4",
    "yId2wIocTys_360.0_510.0.mp4",
    "A_MFAuOwK5k_360.0_510.0.mp4",
    "A_MFAuOwK5k_210.0_360.0.mp4",
    "A_MFAuOwK5k_60.0_210.0.mp4",
    "A_MFAuOwK5k_510.0_660.0.mp4",
    "cJ8kzdeoevg_210.0_360.0.mp4",
    "cJ8kzdeoevg_360.0_510.0.mp4",
    "cJ8kzdeoevg_60.0_210.0.mp4",
    "J_6fDCo1REI_60.0_210.0.mp4",
    "Jz1Cszaqck0_360.0_510.0.mp4",
    "Jz1Cszaqck0_60.0_210.0.mp4",
    "Jz1Cszaqck0_210.0_360.0.mp4",
    "Jz1Cszaqck0_660.0_810.0.mp4",
    "Jz1Cszaqck0_510.0_660.0.mp4",
    "FL0Cos34RjU_60.0_210.0.mp4",
]


def norm_filename(name: str) -> str:
    """
    Normalize filename to match targets robustly:
    - take basename
    - strip leading "--" repeatedly (ONLY at the start)
    """
    base = os.path.basename(name.strip())
    while base.startswith("--"):
        base = base[2:]
    return base


def open_text_maybe_gzip(path: Path, mode: str = "rt"):
    # Supports .gz inputs/outputs if you ever need it.
    if path.suffix == ".gz":
        return gzip.open(path, mode=mode, encoding="utf-8", newline="")
    return path.open(mode=mode, encoding="utf-8", newline="")


def collect_inputs(inputs: List[str], input_dir: Optional[str], pattern: str) -> List[Path]:
    paths: List[Path] = []

    for p in inputs:
        pp = Path(p)
        if pp.is_dir():
            paths.extend(sorted(pp.glob(pattern)))
        else:
            paths.append(pp)

    if input_dir is not None:
        d = Path(input_dir)
        paths.extend(sorted(d.glob(pattern)))

    # De-dup while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def load_targets(targets_txt: Optional[str], targets_args: Optional[List[str]]) -> List[str]:
    if targets_args:
        raw = targets_args
    elif targets_txt:
        tpath = Path(targets_txt)
        with tpath.open("r", encoding="utf-8") as f:
            raw = [line.strip() for line in f if line.strip()]
    else:
        raw = DEFAULT_TARGETS
    return raw


def extract_row_filename(row: dict) -> Optional[str]:
    # Prefer "filename", else fall back to basename of "path".
    fn = row.get("filename")
    if isinstance(fn, str) and fn.strip():
        return fn
    p = row.get("path")
    if isinstance(p, str) and p.strip():
        return os.path.basename(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="Input JSONL file paths and/or directories (directories will be globbed with --pattern).",
    )
    ap.add_argument(
        "--input_dir",
        default=None,
        help="Optional directory containing JSONLs (will be globbed with --pattern).",
    )
    ap.add_argument("--pattern", default="*.jsonl", help='Glob pattern when scanning directories (default: "*.jsonl").')
    ap.add_argument("--targets_txt", default=None, help="Text file with one target filename per line.")
    ap.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Target filenames passed directly as CLI args (overrides --targets_txt).",
    )
    ap.add_argument("--out", required=True, help="Output JSONL path (streaming write).")
    ap.add_argument(
        "--allow_duplicates",
        action="store_true",
        help="If set, writes every matched row even if the same target is found multiple times.",
    )
    ap.add_argument(
        "--no_early_stop",
        action="store_true",
        help="If set, scans all rows even after all targets have been found.",
    )
    args = ap.parse_args()

    input_paths = collect_inputs(args.inputs, args.input_dir, args.pattern)
    if not input_paths:
        raise SystemExit("No input JSONL files found. Provide --inputs and/or --input_dir.")

    targets_raw = load_targets(args.targets_txt, args.targets)
    targets_norm: Set[str] = {norm_filename(x) for x in targets_raw}

    # For reporting: map normalized target -> original targets that normalize to it
    norm_to_originals: Dict[str, List[str]] = {}
    for t in targets_raw:
        nt = norm_filename(t)
        norm_to_originals.setdefault(nt, []).append(t)

    remaining: Set[str] = set(targets_norm)
    found_in: Dict[str, str] = {}  # normalized target -> input file where found

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scanned_bar = tqdm(desc="rows scanned", unit="row", position=0, leave=True)
    saved_bar = tqdm(
        desc="rows saved",
        unit="row",
        position=1,
        leave=True,
        total=None if args.allow_duplicates else len(targets_norm),
    )

    wrote_norm: Set[str] = set()

    try:
        with open_text_maybe_gzip(out_path, "wt") as fout:
            for jp in input_paths:
                if (not remaining) and (not args.no_early_stop) and (not args.allow_duplicates):
                    break

                with open_text_maybe_gzip(jp, "rt") as fin:
                    for line in fin:
                        if not line.strip():
                            continue

                        scanned_bar.update(1)

                        try:
                            row = json_loads(line.encode("utf-8"))
                        except Exception:
                            # If any line is corrupted, skip it and keep going.
                            continue

                        if not isinstance(row, dict):
                            continue

                        row_fn = extract_row_filename(row)
                        if not row_fn:
                            continue

                        key = norm_filename(row_fn)

                        if args.allow_duplicates:
                            if key in targets_norm:
                                fout.write(json_dumps(row) + "\n")
                                saved_bar.update(1)
                        else:
                            if key in remaining:
                                fout.write(json_dumps(row) + "\n")
                                # streaming mode: write line-by-line, no buffering assumptions
                                saved_bar.update(1)

                                remaining.remove(key)
                                found_in[key] = str(jp)

                                # Defensive: also avoid re-writing same key if it appears again
                                wrote_norm.add(key)

                                if (not remaining) and (not args.no_early_stop):
                                    break
    finally:
        scanned_bar.close()
        saved_bar.close()

    if args.allow_duplicates:
        print(f"\nDone. Output: {out_path}")
        print("Note: --allow_duplicates was set; duplicates may exist if a target appears in multiple inputs.")
        return

    missing = sorted(remaining)
    found = len(targets_norm) - len(remaining)

    print(f"\nDone. Output: {out_path}")
    print(f"Targets requested (unique after normalization): {len(targets_norm)}")
    print(f"Targets found/written: {found}")
    print(f"Targets missing: {len(missing)}")

    if missing:
        print("\nMissing targets (normalized -> originals you provided):")
        for m in missing:
            originals = norm_to_originals.get(m, [m])
            print(f"  {m}  <-  {', '.join(originals)}")

    # Optional: show where matches came from
    if found_in:
        print("\nFound targets (normalized -> input jsonl):")
        for k in sorted(found_in.keys()):
            print(f"  {k}  ->  {found_in[k]}")


if __name__ == "__main__":
    main()

'''
python extract_rows_by_filename.py --inputs ./qvhighlights/captions_part_000.jsonl ./qvhighlights/captions_part_001.jsonl ./qvhighlights/captions_part_002.jsonl ./qvhighlights/captions_part_003.jsonl ./qvhighlights/captions_part_004.jsonl --out ./matched.jsonl
'''
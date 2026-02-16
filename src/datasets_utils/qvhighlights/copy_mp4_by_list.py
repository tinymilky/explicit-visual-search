#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copy a set of .mp4 files from a source folder to a target folder.

Designed for your case:
- You already have a list of filenames (possibly without leading "--").
- In the source folder, some files may have leading "--" (or even multiple).
- This script matches by a normalized basename that strips leading "--" repeatedly.
- Two tqdm progress bars:
  1) targets processed
  2) files copied
- Copies in streaming sense (file-by-file) and can overwrite or skip existing.

Fast path:
- Builds an index of all mp4 files under --src (optionally recursive) keyed by normalized basename.
- Then copies requested files by lookup.

Usage:
  python copy_mp4_by_list.py --src /path/to/videos --dst /path/to/selected_videos --targets_txt targets.txt

Or pass targets directly:
  python copy_mp4_by_list.py --src ... --dst ... --targets A.mp4 B.mp4

Options:
  --recursive           search src directory recursively
  --flatten             copy all into dst root (default True). If off, preserves relative path from src.
  --overwrite           overwrite if file exists in dst
  --dry_run             only print what would be copied
  --extensions .mp4      (default)
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm


def norm_filename(name: str) -> str:
    """
    Normalize by basename and stripping leading '--' repeatedly.
    """
    base = os.path.basename(name.strip())
    while base.startswith("--"):
        base = base[2:]
    return base


def load_targets(targets_txt: Optional[str], targets_args: Optional[List[str]]) -> List[str]:
    if targets_args and len(targets_args) > 0:
        raw = targets_args
    elif targets_txt:
        p = Path(targets_txt)
        with p.open("r", encoding="utf-8") as f:
            raw = [line.strip() for line in f if line.strip()]
    else:
        raise SystemExit("Provide --targets_txt or --targets.")
    return raw


def build_index(src: Path, recursive: bool, exts: Tuple[str, ...]) -> Dict[str, Path]:
    """
    Build mapping: normalized_basename -> a concrete path in src.

    If duplicates exist (same normalized name appears multiple times),
    we keep the first one encountered.
    """
    index: Dict[str, Path] = {}

    it = src.rglob("*") if recursive else src.glob("*")
    for p in it:
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        key = norm_filename(p.name)
        index.setdefault(key, p)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source directory containing mp4 files.")
    ap.add_argument("--dst", required=True, help="Destination directory.")
    ap.add_argument("--targets_txt", default=None, help="Text file with one filename per line.")
    ap.add_argument("--targets", nargs="*", default=None, help="Target filenames passed directly.")
    ap.add_argument("--recursive", action="store_true", help="Search source directory recursively.")
    ap.add_argument(
        "--flatten",
        action="store_true",
        help="Copy all files into dst root (default: True).",
    )
    ap.add_argument(
        "--preserve_tree",
        action="store_true",
        help="Preserve relative path from src inside dst (overrides --flatten).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite destination if exists.")
    ap.add_argument("--dry_run", action="store_true", help="Print actions without copying.")
    ap.add_argument(
        "--extensions",
        nargs="*",
        default=[".mp4"],
        help='Extensions to consider (default: .mp4). Example: --extensions .mp4 .mkv',
    )
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists() or not src.is_dir():
        raise SystemExit(f"--src must be an existing directory: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    flatten = True
    if args.preserve_tree:
        flatten = False
    elif args.flatten:
        flatten = True
    else:
        # default behavior: flatten
        flatten = True

    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.extensions)

    targets_raw = load_targets(args.targets_txt, args.targets)
    targets_norm = [norm_filename(x) for x in targets_raw]
    targets_set: Set[str] = set(targets_norm)

    # Build index
    print(f"Indexing {src} (recursive={args.recursive}) for extensions {exts} ...")
    index = build_index(src, args.recursive, exts)
    print(f"Indexed {len(index)} unique normalized filenames.")

    targets_bar = tqdm(total=len(targets_norm), desc="targets processed", unit="file", position=0, leave=True)
    copied_bar = tqdm(total=len(targets_set), desc="files copied", unit="file", position=1, leave=True)

    missing: List[str] = []
    skipped_exists: List[str] = []
    copied: List[str] = []

    try:
        for raw_name, key in zip(targets_raw, targets_norm):
            targets_bar.update(1)

            src_path = index.get(key)
            if src_path is None:
                missing.append(raw_name)
                continue

            if flatten:
                dst_path = dst / src_path.name
            else:
                rel = src_path.relative_to(src)
                dst_path = dst / rel
                dst_path.parent.mkdir(parents=True, exist_ok=True)

            if dst_path.exists() and not args.overwrite:
                skipped_exists.append(str(dst_path))
                continue

            if args.dry_run:
                print(f"[DRY RUN] copy: {src_path} -> {dst_path}")
            else:
                # copy2 preserves metadata; for large files, still streams at OS level
                shutil.copy2(src_path, dst_path)

            copied.append(str(dst_path))
            copied_bar.update(1)

    finally:
        targets_bar.close()
        copied_bar.close()

    print("\nDone.")
    print(f"Requested targets (lines): {len(targets_raw)}")
    print(f"Unique normalized targets: {len(targets_set)}")
    print(f"Copied: {len(copied)}")
    print(f"Skipped (already exists, no --overwrite): {len(skipped_exists)}")
    print(f"Missing in source: {len(missing)}")

    if missing:
        print("\nMissing filenames (as provided):")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()

'''
# targets.txt contains your list (one filename per line)
python copy_mp4_by_list.py \
  --src /mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/datasets_to_annotate/QVHighlights/videos \
  --dst ./qvhighlights_case_study \
  --targets_txt targets.txt \
  --recursive
'''
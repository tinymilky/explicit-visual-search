#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copy images referenced by a filtered JSONL into a new folder.

Input JSONL rows must contain:
- "image": filename (e.g., "X51006555072.jpg")

This script:
- collects unique image names from the JSONL
- copies them from --src-image-dir to --dst-image-dir
- prints stats (unique images, copied, missing, duplicates)

Usage:
  python copy_images_from_jsonl.py \
    --input sroie_cot_train_bboxratio_le_0p1.jsonl \
    --src-image-dir /path/to/sroie_images \
    --dst-image-dir /path/to/sroie_images_filtered

Optional:
  --ext-fallback ".jpg,.jpeg,.png"   # try other extensions if exact name not found
  --symlink                           # create symlinks instead of copying (Linux/macOS)
  --overwrite                         # overwrite existing files
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm


def parse_ext_fallback(s: str) -> List[str]:
    exts = []
    for x in (s or "").split(","):
        x = x.strip()
        if not x:
            continue
        if not x.startswith("."):
            x = "." + x
        exts.append(x.lower())
    return exts


def find_image(src_dir: Path, name: str, ext_fallback: List[str]) -> Optional[Path]:
    p = src_dir / name
    if p.exists():
        return p

    stem = Path(name).stem
    for ext in ext_fallback:
        cand = src_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def safe_copy(src: Path, dst: Path, overwrite: bool) -> bool:
    if dst.exists():
        if not overwrite:
            return False
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def safe_symlink(src: Path, dst: Path, overwrite: bool) -> bool:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return False
        try:
            dst.unlink()
        except Exception:
            return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src.resolve(), dst)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Filtered JSONL (e.g., sroie_cot_train_bboxratio_le_0p1.jsonl)")
    ap.add_argument("--src-image-dir", required=True, help="Source image folder")
    ap.add_argument("--dst-image-dir", required=True, help="Destination folder for copied images")
    ap.add_argument("--image-key", default="image", help="JSON key for image filename (default: image)")
    ap.add_argument("--ext-fallback", default="", help='Comma-separated extensions to try if exact filename missing, e.g. ".jpg,.png"')
    ap.add_argument("--symlink", action="store_true", help="Create symlinks instead of copying (Linux/macOS)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files in destination")
    ap.add_argument("--no-total", action="store_true", help="Do not pre-count lines for tqdm total")
    args = ap.parse_args()

    src_dir = Path(args.src_image_dir)
    dst_dir = Path(args.dst_image_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    ext_fallback = parse_ext_fallback(args.ext_fallback)

    # collect unique images
    total = None
    if not args.no_total:
        try:
            with open(args.input, "rb") as f:
                total = sum(1 for _ in f)
        except Exception:
            total = None

    images: Set[str] = set()
    bad_rows = 0
    rows = 0

    with open(args.input, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=total, desc="Reading JSONL", dynamic_ncols=True):
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                row = json.loads(line)
            except Exception:
                bad_rows += 1
                continue
            if not isinstance(row, dict):
                bad_rows += 1
                continue
            name = row.get(args.image_key, None)
            if isinstance(name, str) and name:
                images.add(name)

    images_list = sorted(images)
    print(f"Rows read: {rows} (bad: {bad_rows})")
    print(f"Unique images in JSONL: {len(images_list)}")

    copied = 0
    skipped_existing = 0
    missing = 0
    errors = 0

    op = safe_symlink if args.symlink else safe_copy

    for name in tqdm(images_list, desc="Copying images", dynamic_ncols=True):
        src_path = find_image(src_dir, name, ext_fallback)
        if src_path is None:
            missing += 1
            continue

        # keep original name in destination (even if fallback extension was used)
        dst_path = dst_dir / Path(name).name

        try:
            did = op(src_path, dst_path, args.overwrite)
            if did:
                copied += 1
            else:
                skipped_existing += 1
        except Exception:
            errors += 1

    print("=== Done ===")
    print(f"Copied: {copied}")
    print(f"Skipped (already exists): {skipped_existing}")
    print(f"Missing in src: {missing}")
    print(f"Errors: {errors}")
    print(f"Destination: {dst_dir}")


if __name__ == "__main__":
    main()

'''
python copy_images_from_jsonl.py --input sroie_cot_train_bboxratio_le_0p1.jsonl --src-image-dir ./sroie_imgs --dst-image-dir ./sroie_imgs_bboxratio_le_0p1

'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect valid (image, ocr_json) assets referenced by a JSONL and copy them to new folders.

Rules:
- image file name is taken from row["image"] (e.g., "34347.jpeg")
- ocr file name is derived from image stem: Path(image).stem + ".json" (e.g., "34347.json")
- image_root and ocr_root can be different folders

Outputs:
- Copies valid images into --out-image-dir
- Copies valid OCR jsons into --out-ocr-dir
- Optionally writes --out-jsonl containing only rows whose image+ocr exist

Usage:
  python collect_valid_assets.py ^
    --input infographicsvqa_annotated.jsonl ^
    --image-root D:\vis_cot\infographicsVQA\infographicsvqa_images ^
    --ocr-root   D:\vis_cot\infographicsVQA\infographicsvqa_ocr ^
    --out-image-dir D:\vis_cot\infographicsVQA\valid_images ^
    --out-ocr-dir   D:\vis_cot\infographicsVQA\valid_ocr ^
    --out-jsonl     D:\vis_cot\infographicsVQA\valid_rows.jsonl

Optional:
  --symlink            Create symlinks instead of copying (Linux/macOS; Windows needs admin/dev mode)
  --overwrite          Overwrite if already exists
  --missing-report     Write missing images/jsons report as JSON
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from tqdm import tqdm


def safe_copy(src: Path, dst: Path, overwrite: bool) -> bool:
    if dst.exists():
        if not overwrite:
            return False
        try:
            dst.unlink()
        except Exception:
            # if dst is a dir (shouldn't), fall back
            shutil.rmtree(dst, ignore_errors=True)
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
    ap.add_argument("--input", required=True, help="Input JSONL")
    ap.add_argument("--image-root", required=True, help="Folder containing images")
    ap.add_argument("--ocr-root", required=True, help="Folder containing OCR jsons (stem.json)")
    ap.add_argument("--out-image-dir", required=True, help="Output folder for valid images")
    ap.add_argument("--out-ocr-dir", required=True, help="Output folder for valid OCR jsons")
    ap.add_argument("--out-jsonl", default=None, help="Optional: write filtered rows (valid only)")
    ap.add_argument("--image-key", default="image", help='Key for image name (default: "image")')
    ap.add_argument("--symlink", action="store_true", help="Symlink instead of copying")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite files in output dirs")
    ap.add_argument("--missing-report", default=None, help="Optional JSON report for missing assets")
    ap.add_argument("--no-total", action="store_true", help="Do not pre-count lines for tqdm")
    args = ap.parse_args()

    input_path = Path(args.input)
    image_root = Path(args.image_root)
    ocr_root = Path(args.ocr_root)
    out_img = Path(args.out_image_dir)
    out_ocr = Path(args.out_ocr_dir)
    out_img.mkdir(parents=True, exist_ok=True)
    out_ocr.mkdir(parents=True, exist_ok=True)

    op = safe_symlink if args.symlink else safe_copy

    total = None
    if not args.no_total:
        try:
            with open(input_path, "rb") as f:
                total = sum(1 for _ in f)
        except Exception:
            total = None

    # Track unique assets
    seen_images: Set[str] = set()
    valid_images: Set[str] = set()
    valid_ocr: Set[str] = set()

    missing_images: Set[str] = set()
    missing_ocr: Set[str] = set()

    rows_total = 0
    rows_valid = 0
    rows_bad_json = 0

    copied_images = 0
    copied_ocr = 0
    skipped_existing_images = 0
    skipped_existing_ocr = 0

    fout_jsonl = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=total, desc="Scanning JSONL", dynamic_ncols=True):
            line = line.strip()
            if not line:
                continue
            rows_total += 1
            try:
                row = json.loads(line)
            except Exception:
                rows_bad_json += 1
                continue
            if not isinstance(row, dict):
                rows_bad_json += 1
                continue

            img_name = row.get(args.image_key, None)
            if not isinstance(img_name, str) or not img_name:
                continue

            seen_images.add(img_name)

            img_path = image_root / img_name
            ocr_name = f"{Path(img_name).stem}.json"
            ocr_path = ocr_root / ocr_name

            img_ok = img_path.is_file()
            ocr_ok = ocr_path.is_file()

            if not img_ok:
                missing_images.add(img_name)
            if not ocr_ok:
                missing_ocr.add(ocr_name)

            if not (img_ok and ocr_ok):
                continue

            # This row is valid (both assets exist)
            rows_valid += 1
            if fout_jsonl:
                fout_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Copy each asset once per unique file
            if img_name not in valid_images:
                valid_images.add(img_name)
                dst_img = out_img / Path(img_name).name
                did = op(img_path, dst_img, args.overwrite)
                if did:
                    copied_images += 1
                else:
                    skipped_existing_images += 1

            if ocr_name not in valid_ocr:
                valid_ocr.add(ocr_name)
                dst_ocr = out_ocr / Path(ocr_name).name
                did = op(ocr_path, dst_ocr, args.overwrite)
                if did:
                    copied_ocr += 1
                else:
                    skipped_existing_ocr += 1

    if fout_jsonl:
        fout_jsonl.close()

    print("\n=== Summary ===")
    print(f"rows_total: {rows_total}")
    print(f"rows_bad_json: {rows_bad_json}")
    print(f"rows_valid (image+ocr exist): {rows_valid}")
    print(f"unique_images_referenced: {len(seen_images)}")
    print(f"unique_valid_images_copied_or_linked: {len(valid_images)}")
    print(f"unique_valid_ocr_copied_or_linked: {len(valid_ocr)}")
    print(f"copied_images: {copied_images} (skipped_existing: {skipped_existing_images})")
    print(f"copied_ocr: {copied_ocr} (skipped_existing: {skipped_existing_ocr})")
    print(f"missing_images_unique: {len(missing_images)}")
    print(f"missing_ocr_unique: {len(missing_ocr)}")
    if args.out_jsonl:
        print(f"filtered_jsonl_out: {args.out_jsonl}")
    print(f"out_image_dir: {out_img}")
    print(f"out_ocr_dir: {out_ocr}")

    if args.missing_report:
        rep = {
            "rows_total": rows_total,
            "rows_bad_json": rows_bad_json,
            "rows_valid": rows_valid,
            "unique_images_referenced": len(seen_images),
            "unique_valid_images": len(valid_images),
            "unique_valid_ocr": len(valid_ocr),
            "missing_images_unique": len(missing_images),
            "missing_ocr_unique": len(missing_ocr),
            "missing_images": sorted(missing_images),
            "missing_ocr": sorted(missing_ocr),
        }
        with open(args.missing_report, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"missing_report_out: {args.missing_report}")


if __name__ == "__main__":
    main()


'''
  python collect_valid_assets.py --input infographicsvqa_vqa_gemini_3fh_raw_gemini_bboxformat.jsonl --image-root ./infographicsvqa_images --ocr-root   ./infographicsvqa_ocr --out-image-dir ./valid_images --out-ocr-dir   ./valid_ocr --out-jsonl     ./valid_rows.jsonl
'''
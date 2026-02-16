#!/usr/bin/env python3
"""
Efficient downloader for Yuwh07/BBox_DocVQA_Train (HF Datasets repo).

Repo contents (as of Nov 2025):
- BBox_DocVQA_Train.tar.gz  (~70.2 GB): PNG pages in category folders
- BBox_DocVQA_Train.jsonl   (~14.7 MB): QA + bbox annotations referencing relative image paths

Usage examples:
  # Download both jsonl + tar.gz to /mnt/data_8/bbox_docvqa_train
  python download_bbox_docvqa_train.py --out /mnt/data_8/bbox_docvqa_train

  # Download only annotations
  python download_bbox_docvqa_train.py --metadata-only --out ./bbox_docvqa_train

  # Download only images tarball
  python download_bbox_docvqa_train.py --images-only --out ./bbox_docvqa_train

  # Download parquet-converted annotations (small)
  python download_bbox_docvqa_train.py --parquet-only --out ./bbox_docvqa_train_parquet

  # Download + extract tarball (uses system tar; uses pigz if available)
  python download_bbox_docvqa_train.py --extract --out ./bbox_docvqa_train --extract-dir ./bbox_docvqa_train/pages
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


DEFAULT_REPO = "Yuwh07/BBox_DocVQA_Train"
PARQUET_REV = "refs/convert/parquet"


def _split_patterns(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _run(cmd: List[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id, e.g. Yuwh07/BBox_DocVQA_Train")
    ap.add_argument("--revision", default="main", help='Git revision/branch (default: "main")')
    ap.add_argument("--out", required=True, help="Output directory to place downloaded files")
    ap.add_argument("--token", default=None, help="HF token (optional). If omitted, uses HF_TOKEN env / cached login.")
    ap.add_argument("--max-workers", type=int, default=16, help="Parallel download workers for snapshot_download")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--metadata-only", action="store_true", help="Download only the JSONL annotations")
    mode.add_argument("--images-only", action="store_true", help="Download only the image tarball")
    mode.add_argument("--parquet-only", action="store_true", help="Download parquet-converted annotations (small)")

    ap.add_argument("--allow", default=None,
                    help="Comma-separated allow_patterns override (e.g. '*.jsonl,*.tar.gz'). "
                         "If not set, determined by mode.")
    ap.add_argument("--ignore", default=None, help="Comma-separated ignore_patterns")

    ap.add_argument("--xet-high-perf", action="store_true",
                    help="Set HF_XET_HIGH_PERFORMANCE=1 before importing huggingface_hub (faster transfers).")

    ap.add_argument("--hf-home", default=None,
                    help="If set, export HF_HOME to control cache/token location (put on fast disk).")

    ap.add_argument("--list-files", action="store_true",
                    help="List repo files first (useful to sanity-check patterns)")

    ap.add_argument("--extract", action="store_true", help="After download, extract the tar.gz (if present).")
    ap.add_argument("--extract-dir", default=None, help="Extraction directory (default: <out>/extracted_pages)")
    ap.add_argument("--keep-tar", action="store_true", help="Keep tar.gz after extraction (default keeps it).")
    args = ap.parse_args()

    # Set env vars BEFORE importing huggingface_hub so hf-xet config is applied.
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    if args.xet_high_perf:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    # Import here (after env vars).
    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as e:
        print("ERROR: huggingface_hub is required. Install with:", file=sys.stderr)
        print('  pip install -U "huggingface_hub[hf_xet]"', file=sys.stderr)
        raise

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide revision + allow_patterns based on mode (unless user overrides with --allow).
    revision = args.revision
    allow_patterns = _split_patterns(args.allow)
    ignore_patterns = _split_patterns(args.ignore)

    if args.parquet_only:
        revision = PARQUET_REV
        if allow_patterns is None:
            allow_patterns = ["**/*.parquet"]
    else:
        if allow_patterns is None:
            if args.metadata_only:
                allow_patterns = ["*.jsonl", "**/*.jsonl"]
            elif args.images_only:
                allow_patterns = ["*.tar.gz", "**/*.tar.gz"]
            else:
                allow_patterns = ["*.jsonl", "**/*.jsonl", "*.tar.gz", "**/*.tar.gz"]

    token = args.token  # None is fine (uses cached login or HF_TOKEN env)

    if args.list_files:
        api = HfApi()
        files = api.list_repo_files(repo_id=args.repo, repo_type="dataset", revision=revision, token=token)
        print(f"Repo files @ {args.repo} ({revision}):")
        for f in files:
            print(" -", f)
        if allow_patterns:
            matched = [f for f in files if any(fnmatch.fnmatch(f, p) for p in allow_patterns)]
            print("\nFiles matching allow_patterns:")
            for f in matched:
                print(" *", f)

    print(f"\nDownloading: {args.repo} (revision={revision})")
    print(f"-> out_dir: {out_dir}")
    print(f"-> allow_patterns: {allow_patterns}")
    print(f"-> ignore_patterns: {ignore_patterns}")
    print(f"-> max_workers: {args.max_workers}")
    if args.xet_high_perf:
        print("-> HF_XET_HIGH_PERFORMANCE=1 enabled")

    # snapshot_download replicates repo structure into local_dir and uses a local .cache under it. :contentReference[oaicite:3]{index=3}
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=revision,
        local_dir=str(out_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        max_workers=args.max_workers,
        token=token,
    )

    print("\nDownload complete.")
    tar_path = out_dir / "BBox_DocVQA_Train.tar.gz"
    jsonl_path = out_dir / "BBox_DocVQA_Train.jsonl"

    if jsonl_path.exists():
        print(f"Found annotations: {jsonl_path} ({jsonl_path.stat().st_size/1e6:.1f} MB)")
    if tar_path.exists():
        print(f"Found image tarball: {tar_path} ({tar_path.stat().st_size/1e9:.2f} GB)")

    if args.extract:
        if not tar_path.exists():
            print("No tar.gz found to extract. (Maybe you used --metadata-only or --parquet-only.)")
            return
        extract_dir = Path(args.extract_dir).expanduser().resolve() if args.extract_dir else (out_dir / "extracted_pages")
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Prefer system tar with pigz if available (much faster decompression).
        pigz = shutil.which("pigz")
        tar = shutil.which("tar")
        if tar is None:
            raise RuntimeError("System 'tar' not found. Install tar or extract manually.")

        print(f"\nExtracting to: {extract_dir}")
        if pigz:
            _run([tar, "--use-compress-program", pigz, "-xf", str(tar_path), "-C", str(extract_dir)])
        else:
            _run([tar, "-xzf", str(tar_path), "-C", str(extract_dir)])

        print("Extraction complete.")
        if not args.keep_tar:
            tar_path.unlink(missing_ok=True)
            print("Removed tarball.")

    # Helpful hint about parquet branch
    if not args.parquet_only:
        print("\nTip: A small parquet-converted annotations branch exists at revision "
              f"'{PARQUET_REV}' (download with --parquet-only).")


if __name__ == "__main__":
    main()

'''
python download_bbox_docvqa_train.py --extract --out ./bbox_docvqa_train --extract-dir ./bbox_docvqa_train/pages
'''
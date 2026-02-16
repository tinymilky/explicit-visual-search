#!/usr/bin/env python3
# extract_split_tar.py
#
# Stream-extract a split TAR archive (cot_images_00, cot_images_01, ...)
# without concatenating with `cat` and without creating an intermediate tar.
#
# Usage:
#   pip install -U tqdm
#   python extract_split_tar.py --in_dir /path/to/parts --pattern "cot_images_*" --out_dir /path/to/out
#
# Notes:
# - Works for plain .tar splits like your case.
# - Uses a basic path-traversal guard.
# - Uses streaming mode (r|*), so extraction starts immediately and is memory-efficient.

import argparse
import glob
import io
import os
import tarfile
from typing import List, Optional

from tqdm import tqdm


class MultiPartReader(io.RawIOBase):
    """A file-like object that reads sequentially from multiple files as one stream."""

    def __init__(self, paths: List[str], bufsize: int = 1024 * 1024):
        if not paths:
            raise ValueError("No part files provided.")
        self.paths = paths
        self.bufsize = bufsize
        self._i = 0
        self._f: Optional[io.BufferedReader] = None
        self._open_next()

    def _open_next(self) -> None:
        if self._f is not None:
            self._f.close()
        if self._i >= len(self.paths):
            self._f = None
            return
        self._f = open(self.paths[self._i], "rb", buffering=self.bufsize)
        self._i += 1

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        if self._f is None:
            return 0
        n = self._f.readinto(b)
        if n == 0:
            self._open_next()
            return self.readinto(b)
        return n

    def close(self) -> None:
        try:
            if self._f is not None:
                self._f.close()
        finally:
            super().close()


def safe_extract_member(tf: tarfile.TarFile, member: tarfile.TarInfo, out_dir: str) -> None:
    """Prevent path traversal by ensuring member target stays within out_dir."""
    out_dir_abs = os.path.abspath(out_dir)

    # tar paths are POSIX; TarInfo.name may contain .. or absolute paths
    target_path = os.path.abspath(os.path.join(out_dir_abs, member.name))

    if not (target_path == out_dir_abs or target_path.startswith(out_dir_abs + os.sep)):
        raise RuntimeError(f"Blocked suspicious path traversal entry: {member.name!r}")

    tf.extract(member, path=out_dir_abs)


def numeric_sort_key(path: str):
    # Prefer a stable ordering for typical split names:
    # cot_images_00, cot_images_01, ...
    base = os.path.basename(path)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder containing split parts (e.g., cot_images_00...)")
    ap.add_argument("--pattern", required=True, help='Glob pattern inside --in_dir (e.g., "cot_images_*")')
    ap.add_argument("--out_dir", required=True, help="Output folder for extracted files")
    ap.add_argument("--dry_run", action="store_true", help="List entries only; do not extract")
    ap.add_argument("--strip_components", type=int, default=0,
                    help="Strip N leading path components when extracting (like tar --strip-components)")
    ap.add_argument("--progress", action="store_true", help="Show tqdm progress (counts members, not bytes)")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)

    if not os.path.isdir(in_dir):
        raise SystemExit(f"--in_dir does not exist or is not a directory: {in_dir}")

    os.makedirs(out_dir, exist_ok=True)

    # Resolve parts
    pattern = os.path.join(in_dir, args.pattern)
    parts = sorted(glob.glob(pattern), key=numeric_sort_key)

    if not parts:
        raise SystemExit(f"No files matched pattern: {pattern}")

    # Stream open as a tar archive; r|* allows autodetect compression if present
    with MultiPartReader(parts) as stream:
        with tarfile.open(fileobj=stream, mode="r|*") as tf:
            iterator = tf
            if args.progress:
                iterator = tqdm(tf, desc="Extracting", unit="member")

            for member in iterator:
                # Optionally strip leading components
                if args.strip_components > 0:
                    # Apply stripping only to normal paths; skip if insufficient components
                    comps = member.name.split("/")
                    if len(comps) <= args.strip_components:
                        continue
                    member.name = "/".join(comps[args.strip_components:])

                if args.dry_run:
                    print(member.name)
                    continue

                safe_extract_member(tf, member, out_dir)

    print(f"Done. Parts: {len(parts)}; extracted to: {out_dir}")


if __name__ == "__main__":
    main()

'''
python extract_split_tar.py --in_dir /mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/datasets_to_train/visualcot/cot_images_tar_split --pattern "cot_images_*" --out_dir ./vis_cot --progress
'''
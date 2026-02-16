#!/usr/bin/env python3
"""
Zip a folder (e.g., images) with relatively high compression.

- Uses ZIP_DEFLATED + max compresslevel=9
- Stores relative paths inside the zip (preserves folder structure)

Usage:
  python zip_folder.py --input /path/to/img_folder --output /path/to/img_folder.zip
"""

import argparse
import os
import zipfile


def zip_folder(input_dir: str, output_zip: str) -> None:
    input_dir = os.path.abspath(input_dir)
    output_zip = os.path.abspath(output_zip)

    with zipfile.ZipFile(
        output_zip,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,   # highest deflate level (Python 3.7+)
    ) as zf:
        for root, _, files in os.walk(input_dir):
            for fn in files:
                src_path = os.path.join(root, fn)
                # keep paths relative to input_dir
                arcname = os.path.relpath(src_path, start=input_dir)
                zf.write(src_path, arcname=arcname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder to zip")
    ap.add_argument("--output", required=True, help="Output .zip path")
    args = ap.parse_args()

    if not os.path.isdir(args.input):
        raise SystemExit(f"Input is not a folder: {args.input}")

    out_parent = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_parent, exist_ok=True)

    zip_folder(args.input, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

'''
python zip_folder.py --input ./infographicsvqa --output ./infographicsvqa_imgs.zip
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download + materialize SlideVQA from Hugging Face:
- Download parquet shards (fast, multi-worker).
- Stream-write a master JSONL of QA rows.
- Deduplicate slide decks: save each deck's page images once into decks/<deck_id>/page_XX.jpg
- Fully resumable.

Tested logic: robust to unknown column names via heuristics; you can override via CLI if needed.
"""

import os
import re
import json
import time
import shutil
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# HF
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import HfFolder

# datasets (for reading parquet and decoding Image features if present)
from datasets import load_dataset
from PIL import Image
from filelock import FileLock


REPO_ID = "NTT-hil-insight/SlideVQA"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_line_count(p: Path) -> int:
    if not p.exists():
        return 0
    # Fast-ish line count without loading all lines
    n = 0
    with p.open("rb") as f:
        for _ in f:
            n += 1
    return n


def _jsonl_append(path: Path, obj: Dict[str, Any]) -> None:
    # Streaming append with flush
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def _find_key(d: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    """Find a key in dict matching any candidate (case-insensitive, underscore-insensitive)."""
    norm = {re.sub(r"[^a-z0-9]+", "", k.lower()): k for k in d.keys()}
    for c in candidates:
        nc = re.sub(r"[^a-z0-9]+", "", c.lower())
        if nc in norm:
            return norm[nc]
    return None


def _is_pil_image(x: Any) -> bool:
    return isinstance(x, Image.Image)


def _decode_image_like(x: Any) -> Optional[Image.Image]:
    """
    Handle datasets Image feature outputs:
    - PIL.Image.Image
    - dict with 'bytes' and/or 'path'
    - string path
    Return PIL Image if possible, else None.
    """
    if x is None:
        return None
    if _is_pil_image(x):
        return x
    if isinstance(x, dict):
        b = x.get("bytes", None)
        p = x.get("path", None)
        if b:
            from io import BytesIO
            try:
                return Image.open(BytesIO(b)).convert("RGB")
            except Exception:
                return None
        if p and isinstance(p, str) and os.path.exists(p):
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                return None
        return None
    if isinstance(x, str) and os.path.exists(x):
        try:
            return Image.open(x).convert("RGB")
        except Exception:
            return None
    return None


def _infer_columns(example: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Heuristic detection of key fields.
    You can override via CLI if this fails.
    """
    cols = {}
    cols["qid"] = _find_key(example, ["qid", "question_id", "id", "qa_id"])
    cols["question"] = _find_key(example, ["question", "Question", "query"])
    cols["answer"] = _find_key(example, ["answer", "Answer", "gold_answer", "label"])
    cols["deck_id"] = _find_key(example, ["deck_id", "deck", "deck_name", "presentation_id", "slide_deck_id"])
    cols["evidence_pages"] = _find_key(example, ["evidence_pages", "evidence", "support_pages", "supporting_pages", "evidence_page_ids"])
    cols["num_pages"] = _find_key(example, ["num_pages", "n_pages", "page_count"])

    # A column likely holding ALL slide images for the deck (often a list/sequence)
    # Try common names:
    cols["slides"] = _find_key(example, ["slides", "slide_images", "images", "pages", "page_images", "deck_images"])

    # Some versions may store image paths separately
    cols["slide_paths"] = _find_key(example, ["slide_paths", "image_paths", "paths", "page_paths"])

    return cols


def _normalize_pages(pages: Any) -> Optional[List[int]]:
    """Normalize evidence pages to a list of ints (1-indexed) if possible."""
    if pages is None:
        return None
    if isinstance(pages, (list, tuple)):
        out = []
        for x in pages:
            try:
                out.append(int(x))
            except Exception:
                pass
        return out if out else None
    # Sometimes stored as string like "1,3,5"
    if isinstance(pages, str):
        nums = re.findall(r"\d+", pages)
        out = [int(n) for n in nums]
        return out if out else None
    # Single int
    try:
        return [int(pages)]
    except Exception:
        return None


def _guess_deck_id(example: Dict[str, Any], cols: Dict[str, Optional[str]], fallback_idx: int) -> str:
    k = cols.get("deck_id")
    if k and example.get(k) is not None:
        return str(example[k])
    # fallback: stable-ish id
    qk = cols.get("qid")
    if qk and example.get(qk) is not None:
        return f"deck_unknown_from_{example[qk]}"
    return f"deck_unknown_{fallback_idx:08d}"


def _save_deck_images_from_row(
    *,
    deck_id: str,
    example: Dict[str, Any],
    cols: Dict[str, Optional[str]],
    decks_root: Path,
    repo_id: str,
    token: Optional[str],
    page_prefix: str = "page_",
    image_ext: str = "jpg",
    per_deck_workers: int = 8,
) -> Tuple[bool, Optional[str], int]:
    """
    Save slide deck images for a deck if available in the example row.
    Returns: (saved_or_exists, error_msg, num_pages_saved)
    """
    deck_dir = decks_root / deck_id
    complete_flag = deck_dir / ".complete"
    lock_path = decks_root / f".{deck_id}.lock"

    _safe_mkdir(decks_root)
    with FileLock(str(lock_path)):
        if complete_flag.exists():
            # Already done
            try:
                n_imgs = len(list(deck_dir.glob(f"{page_prefix}*.{image_ext}")))
            except Exception:
                n_imgs = 0
            return True, None, n_imgs

        # Clean partial tmp if present
        tmp_dir = decks_root / (deck_id + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        _safe_mkdir(tmp_dir)

        # Identify source of images
        slides_key = cols.get("slides")
        paths_key = cols.get("slide_paths")

        slides_val = example.get(slides_key) if slides_key else None
        paths_val = example.get(paths_key) if paths_key else None

        # Case A: slides_val is a list of image-like objects
        pages: List[Any] = []
        if isinstance(slides_val, (list, tuple)) and len(slides_val) > 0:
            pages = list(slides_val)
        elif isinstance(paths_val, (list, tuple)) and len(paths_val) > 0:
            pages = list(paths_val)

        if not pages:
            # No images in this row; cannot materialize deck from this row
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False, f"No slide images found in row (slides_key={slides_key}, paths_key={paths_key}).", 0

        # Save pages
        num_saved = 0
        for i, p in enumerate(pages, start=1):
            out_path = tmp_dir / f"{page_prefix}{i:02d}.{image_ext}"

            # Try PIL decode
            img = _decode_image_like(p)
            if img is not None:
                try:
                    img.save(out_path, quality=95)
                    num_saved += 1
                    continue
                except Exception:
                    pass

            # If it's a string that looks like a repo-relative path, try hf_hub_download then copy
            if isinstance(p, str):
                # If already a local file
                if os.path.exists(p):
                    try:
                        shutil.copy2(p, out_path)
                        num_saved += 1
                        continue
                    except Exception:
                        pass
                # Otherwise assume repo-relative path
                try:
                    local_fp = hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=p,
                        token=token,
                    )
                    shutil.copy2(local_fp, out_path)
                    num_saved += 1
                    continue
                except Exception:
                    pass

            # If dict with path that is repo-relative
            if isinstance(p, dict):
                rp = p.get("path", None)
                if isinstance(rp, str):
                    if os.path.exists(rp):
                        try:
                            shutil.copy2(rp, out_path)
                            num_saved += 1
                            continue
                        except Exception:
                            pass
                    try:
                        local_fp = hf_hub_download(
                            repo_id=repo_id,
                            repo_type="dataset",
                            filename=rp,
                            token=token,
                        )
                        shutil.copy2(local_fp, out_path)
                        num_saved += 1
                        continue
                    except Exception:
                        pass

            # If we reach here, we failed to save this page
            # Keep going (partial decks are not acceptable, but we want a diagnostic)
            # We'll fail the whole deck later.
            continue

        # Require at least 1 page, and ideally a full deck. If dataset is 20-slide decks, enforce 20.
        # If you want to relax this, pass --min_pages_for_deck.
        if num_saved == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False, "Failed to materialize any page images for this deck.", 0

        # Finalize atomically
        if deck_dir.exists():
            shutil.rmtree(deck_dir, ignore_errors=True)
        tmp_dir.rename(deck_dir)
        complete_flag.write_text("ok\n", encoding="utf-8")

        return True, None, num_saved


def _group_parquets_by_split(files: List[str]) -> Dict[str, List[str]]:
    """
    Group parquet files by split inferred from filename prefix before first '-'.
    Example: train-00000-of-00010.parquet -> split 'train'
    """
    out: Dict[str, List[str]] = {}
    for f in files:
        if not f.endswith(".parquet"):
            continue
        base = os.path.basename(f)
        m = re.match(r"^([A-Za-z0-9_]+)-\d+-of-\d+\.parquet$", base)
        if m:
            sp = m.group(1)
        else:
            # fallback: first token until '-' or full name
            sp = base.split("-", 1)[0]
        out.setdefault(sp, []).append(f)
    # Sort shards for deterministic iteration
    for sp in out:
        out[sp] = sorted(out[sp])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", type=str, default=REPO_ID)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--splits", type=str, default="train,validation,test",
                    help="Comma-separated split names to process (must match repo shard prefixes).")
    ap.add_argument("--token", type=str, default=None,
                    help="HF token. If omitted, uses HF_TOKEN env var or cached token.")
    ap.add_argument("--hub_workers", type=int, default=16, help="Workers for snapshot_download.")
    ap.add_argument("--per_deck_workers", type=int, default=8, help="Workers used inside a deck save (I/O bound).")
    ap.add_argument("--page_prefix", type=str, default="page_")
    ap.add_argument("--image_ext", type=str, default="jpg")
    ap.add_argument("--master_jsonl", type=str, default="master.jsonl")
    ap.add_argument("--resume", action="store_true", help="Resume from existing outputs.")
    ap.add_argument("--max_rows", type=int, default=-1, help="Debug: process at most N QA rows per split.")
    # Optional overrides if heuristics fail
    ap.add_argument("--col_question", type=str, default=None)
    ap.add_argument("--col_answer", type=str, default=None)
    ap.add_argument("--col_qid", type=str, default=None)
    ap.add_argument("--col_deck_id", type=str, default=None)
    ap.add_argument("--col_evidence_pages", type=str, default=None)
    ap.add_argument("--col_slides", type=str, default=None)
    ap.add_argument("--col_slide_paths", type=str, default=None)
    ap.add_argument("--hf_token", type=str, default=None,
                help="HF token (same as --token).")
    ap.add_argument("--enable_hf_transfer", action="store_true",
                help="Enable hf_transfer acceleration (sets HF_HUB_ENABLE_HF_TRANSFER=1).")
    ap.add_argument("--force_download", action="store_true",
                help="Force re-download matching files even if cached.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    _safe_mkdir(outdir)
    decks_root = outdir / "decks"
    _safe_mkdir(decks_root)
    master_path = outdir / args.master_jsonl
    schema_path = outdir / "schema_and_example.json"

    # Prefer explicit CLI args, then env, then cached token
    token = (
        args.hf_token
        or args.token
        or os.environ.get("HF_TOKEN")
        or HfFolder.get_token()
    )

    if not token:
        raise SystemExit(
            "No HF token found. Set HF_TOKEN env var or pass --token. "
            "Also ensure you've accepted the dataset license on the HF webpage."
        )

    if args.enable_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    # 1) List repo files (requires gated access accepted)
    api = HfApi()
    try:
        repo_files = api.list_repo_files(args.repo_id, repo_type="dataset", token=token)
    except HfHubHTTPError as e:
        raise SystemExit(
            f"Failed to list repo files (likely gated / license not accepted). "
            f"Visit the dataset page, log in, accept conditions, then retry. Error: {e}"
        )

    parquet_files = [f for f in repo_files if f.endswith(".parquet")]
    if not parquet_files:
        raise SystemExit("No parquet files found in the dataset repo.")

    split_to_files = _group_parquets_by_split(parquet_files)
    available_splits = sorted(split_to_files.keys())

    # ---- split aliases (SlideVQA uses "val") ----
    split_alias = {
        "validation": "val",
        "valid": "val",
        "dev": "val",
    }

    wanted_raw = [s.strip() for s in args.splits.split(",") if s.strip()]
    wanted = [split_alias.get(s, s) for s in wanted_raw]

    # Keep only splits that exist
    wanted = [s for s in wanted if s in split_to_files]
    if not wanted:
        raise SystemExit(
            f"Requested splits not found. Available: {available_splits}. "
            f"Try --splits train,val,test"
        )

    # ---- IMPORTANT: match subfolders too ----
    # '*' matches slashes as well under fnmatch rules used by huggingface_hub.
    allow_patterns = []
    for sp in wanted:
        allow_patterns.extend(split_to_files[sp])

    print(f"[{_now()}] Will download {len(allow_patterns)} parquet files (exact paths).")
    # Optional: print a few examples for sanity
    for ex in allow_patterns[:5]:
        print(f"[{_now()}] example parquet: {ex}")

    try:
        snapshot_dir = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            token=token,
            max_workers=args.hub_workers,
            force_download=args.force_download,
        )
    except TypeError:
        snapshot_dir = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            token=token,
            force_download=args.force_download,
        )

    snapshot_dir = Path(snapshot_dir)
    print(f"[{_now()}] Snapshot dir: {snapshot_dir}")

    # 3) Prepare resume
    already_written = _read_line_count(master_path) if (args.resume and master_path.exists()) else 0
    if already_written > 0:
        print(f"[{_now()}] Resume enabled: master jsonl already has {already_written} lines. "
              f"We will skip that many QA rows (in processing order).")

    # 4) Iterate splits deterministically, stream-write JSONL, save decks once
    global_row_idx = 0  # across splits in the order we process them
    skipped = 0
    wrote = 0
    decks_saved = 0

    schema_dumped = schema_path.exists()

    for sp in wanted:
        all_local_parquets = [p for p in snapshot_dir.rglob("*.parquet")]

        def _local_parquets_for_split(sp: str) -> List[str]:
            rx = re.compile(rf"^{re.escape(sp)}-\d+-of-\d+\.parquet$")
            out = []
            for p in all_local_parquets:
                if rx.match(p.name):
                    out.append(str(p))
            return sorted(out)

        files_local = _local_parquets_for_split(sp)
        if not files_local:
            # Helpful debug: show what exists
            sample = [str(p.relative_to(snapshot_dir)) for p in all_local_parquets[:20]]
            raise SystemExit(
                f"No local parquet files found for split={sp} after snapshot_download.\n"
                f"Found {len(all_local_parquets)} parquets total. Sample:\n  " + "\n  ".join(sample)
            )


        # Load parquet split as a Dataset (memory-mapped; images decoded as needed)
        # Using 'parquet' builder to read local shards.
        ds = load_dataset("parquet", data_files=files_local, split="train")  # split name irrelevant here
        # Inspect first example once
        ex0 = ds[0]
        cols = _infer_columns(ex0)

        # Apply CLI overrides
        if args.col_qid: cols["qid"] = args.col_qid
        if args.col_question: cols["question"] = args.col_question
        if args.col_answer: cols["answer"] = args.col_answer
        if args.col_deck_id: cols["deck_id"] = args.col_deck_id
        if args.col_evidence_pages: cols["evidence_pages"] = args.col_evidence_pages
        if args.col_slides: cols["slides"] = args.col_slides
        if args.col_slide_paths: cols["slide_paths"] = args.col_slide_paths

        if not schema_dumped:
            schema_info = {
                "repo_id": args.repo_id,
                "snapshot_dir": str(snapshot_dir),
                "split": sp,
                "detected_columns": cols,
                "example_keys": list(ex0.keys()),
                "example_preview": {k: (str(ex0[k])[:300] if k in ex0 else None) for k in list(ex0.keys())[:30]},
            }
            schema_path.write_text(json.dumps(schema_info, ensure_ascii=False, indent=2), encoding="utf-8")
            schema_dumped = True
            print(f"[{_now()}] Wrote schema + example preview to: {schema_path}")

        # Process rows
        pbar = tqdm(total=len(ds), desc=f"Split={sp}", dynamic_ncols=True)
        for i, ex in enumerate(ds):
            pbar.update(1)
            global_row_idx += 1

            if args.max_rows > 0 and i >= args.max_rows:
                break

            if args.resume and already_written > 0 and skipped < already_written:
                skipped += 1
                continue

            # Extract key fields
            qid = ex.get(cols["qid"]) if cols.get("qid") else None
            question = ex.get(cols["question"]) if cols.get("question") else None
            answer = ex.get(cols["answer"]) if cols.get("answer") else None
            evidence_pages = _normalize_pages(ex.get(cols["evidence_pages"])) if cols.get("evidence_pages") else None
            deck_id = _guess_deck_id(ex, cols, fallback_idx=global_row_idx)

            # Ensure deck images are materialized once per deck.
            # NOTE: This assumes each QA row carries deck images or slide paths (common in HF packaging).
            # If not, you'll see a clear error per deck; you can then switch to a separate deck-table if provided.
            ok, err, n_pages = _save_deck_images_from_row(
                deck_id=deck_id,
                example=ex,
                cols=cols,
                decks_root=decks_root,
                repo_id=args.repo_id,
                token=token,
                page_prefix=args.page_prefix,
                image_ext=args.image_ext,
                per_deck_workers=args.per_deck_workers,
            )
            if ok and err is None:
                # Count only when we actually created it (heuristic: .complete exists now)
                if (decks_root / deck_id / ".complete").exists():
                    # Might have existed already; we can't perfectly distinguish without extra state.
                    pass
            else:
                # If deck images are not available in rows, we still write QA but flag missing_deck_images.
                n_pages = 0

            deck_rel = f"decks/{deck_id}"
            pages_rel = [f"{deck_rel}/{args.page_prefix}{j:02d}.{args.image_ext}" for j in range(1, n_pages + 1)] if n_pages else None

            row = {
                "qid": str(qid) if qid is not None else None,
                "split": sp,
                "deck_id": deck_id,
                "deck_dir": deck_rel,
                "pages": pages_rel,  # optional, convenient
                "num_pages": n_pages if n_pages else None,
                "evidence_pages": evidence_pages,
                "question": question,
                "answer": answer,
                "missing_deck_images": (not ok),
                "missing_deck_reason": err if (not ok) else None,
            }

            _jsonl_append(master_path, row)
            wrote += 1

        pbar.close()

    print(f"[{_now()}] Done.")
    print(f"[{_now()}] master_jsonl: {master_path}")
    print(f"[{_now()}] wrote_rows: {wrote}")
    print(f"[{_now()}] skipped_rows_on_resume: {skipped}")
    print(f"[{_now()}] decks_root: {decks_root}")
    print(f"[{_now()}] schema_preview: {schema_path}")
    print("\nTip: for faster hub transfers you can install hf_transfer and set:")
    print("  export HF_HUB_ENABLE_HF_TRANSFER=1")


if __name__ == "__main__":
    main()

'''
python download_slidevqa_hf.py --outdir ./slidevqa --enable_hf_transfer --splits train,validation,test --hub_workers 16 --per_deck_workers 8 --resume

'''
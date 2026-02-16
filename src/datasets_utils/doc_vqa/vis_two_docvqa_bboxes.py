#!/usr/bin/env python3
"""
Visualize DocVQA bounding boxes for TWO supported formats:

A) New format (your CoT JSONL rows):
{
  "dataset": "docvqa",
  "split": "train",
  "question_id": ...,
  "image": ["rel/path.png", "rel/path.png###[x1,y1,x2,y2]" ...],
  "conversations": [...]
}

B) Old/simple format (your last message):
{
  "question": "...",
  "answer": "...",
  "possible_answers": [...],
  "image": "mqnk0226_15.png",
  "width": 1638,
  "height": 2196,
  "bboxs": [[x1,y1,x2,y2], ...],
  "dataset": "docvqa",
  "split": "train"
}

What it does:
- Hard-codes 5 samples:
  - 2 CoT-format examples (kpfl0225_13.png)
  - 3 bboxs-format examples (mqnk0226_15.png)
- Draws one figure per sample.
- Labels each box with an ID (1..N).
- Uses a different color per sample and states it in the title.

Usage:
  python vis_docvqa_mixed_samples.py --image-root /mnt/data_8 --out-dir ./vis_out --dedup
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ----------------------------
# Hard-coded samples (5 total)
# ----------------------------

# CoT/new-format samples
SAMPLE_COT_1 = {
    "dataset": "docvqa",
    "split": "train",
    "question_id": 138460,
    "image": [
        "cot/docvqa/kpfl0225_13.png",
        "cot/docvqa/kpfl0225_13.png###[14, 645, 1776, 1617]",
    ],
    "conversations": [
        {"from": "human", "value": "<image>\nWhat is the percentage of all consumers who tried option of \"Semi Hard Strip\"? Please provide the bounding box coordinate of the region that can help you answer the question better."},
        {"from": "gpt", "value": "[0.006, 0.407, 0.807, 0.849]"},
        {"from": "human", "value": "<image>"},
        {"from": "gpt", "value": "8%"},
    ],
}

SAMPLE_COT_2 = {
    "dataset": "docvqa",
    "split": "train",
    "question_id": 138461,
    "image": [
        "cot/docvqa/kpfl0225_13.png",
        "cot/docvqa/kpfl0225_13.png###[1194, 884, 1264, 925]",
    ],
    "conversations": [
        {"from": "human", "value": "<image>\nWhat is the percentage of all consumers who tried option of \"Tobacco Film\"? Please provide the bounding box coordinate of the region that can help you answer the question better."},
        {"from": "gpt", "value": "[0.543, 0.515, 0.575, 0.534]"},
        {"from": "human", "value": "<image>"},
        {"from": "gpt", "value": "19%"},
    ],
}

# Old/simple-format samples
SAMPLE_BBX_1 = {
    "question": "when is j. lowell bond's anniversary?",
    "answer": "AUGUST",
    "possible_answers": ["AUGUST"],
    "image": "mqnk0226_15.png",
    "width": 1638,
    "height": 2196,
    "bboxs": [[1146, 368, 1444, 396], [126, 798, 563, 827]],
    "dataset": "docvqa",
    "split": "train",
}

SAMPLE_BBX_2 = {
    "question": "which month is myry averett's anniversary?",
    "answer": "august",
    "possible_answers": ["AUGUST", "august"],
    "image": "mqnk0226_15.png",
    "width": 1638,
    "height": 2196,
    # "bboxs": [[668/1000*1627, 182/1000*2190, 881/1000*1627, 196/1000*2190], [674/1000*1627, 166/1000*2190, 883/1000*1627, 182/1000*2190]],
    "bboxs": [[1075, 348, 1343, 365],
    [1024, 400, 1438, 432]],
    "dataset": "docvqa",
    "split": "train",
}

SAMPLE_BBX_3 = {
    "question": "which month is j. pryce mitchell's anniversary?",
    "answer": "JULY",
    "possible_answers": ["JULY", "july"],
    "image": "mqnk0226_15.png",
    "width": 1638,
    "height": 2196,
    "bboxs": [
        [0.5128*1267, 0.2550*1536, 0.7570*1267, 0.2778*1536],
        [0.5311*1267, 0.2914*1536, 0.7875*1267, 0.3097*1536],
    ],
    "dataset": "docvqa",
    "split": "train",
}

SAMPLE_BBX_4 = {"question": "what is the contact person name mentioned in letter?", "answer": "P. Carter", "possible_answers": ["P. Carter", "p. carter"], "image": "xnbl0037_1.png", "width": 1695, "height": 2025, "bboxs": [[429, 511, 666, 578], [429, 511, 666, 578]], "dataset": "docvqa", "split": "train"}

HARD_CODED_SAMPLES: List[Dict[str, Any]] = [
    # SAMPLE_COT_1,
    # SAMPLE_COT_2,
    # SAMPLE_BBX_1,
    # SAMPLE_BBX_2,
    # SAMPLE_BBX_3,
    SAMPLE_BBX_4,
]


# ----------------------------
# Utilities
# ----------------------------

def dedup_bboxes_xyxy(bboxes: List[List[int]]) -> List[List[int]]:
    seen = set()
    out = []
    for b in bboxes:
        t = tuple(int(x) for x in b)
        if t not in seen:
            seen.add(t)
            out.append(list(t))
    return out


def parse_bboxes_from_image_field(image_field: Any) -> Tuple[str, List[List[int]]]:
    """
    For CoT/new format:
      image[0] = rel path
      image[i] may contain "###..." with bbox payload.

    Returns:
      (rel_image_path, abs_bboxes_xyxy)
    """
    if isinstance(image_field, str):
        return image_field, []

    if not isinstance(image_field, list) or len(image_field) == 0:
        raise ValueError(f"Unexpected 'image' field format: {type(image_field)}")

    img_path = image_field[0]
    bboxes: List[List[int]] = []

    for item in image_field[1:]:
        if not isinstance(item, str) or "###" not in item:
            continue
        _, tail = item.split("###", 1)
        tail = tail.strip()

        try:
            parsed = ast.literal_eval(tail)
        except Exception:
            continue

        # single bbox [x1,y1,x2,y2]
        if isinstance(parsed, (list, tuple)) and len(parsed) == 4 and all(isinstance(x, (int, float)) for x in parsed):
            bboxes.append([int(round(parsed[0])), int(round(parsed[1])), int(round(parsed[2])), int(round(parsed[3]))])
        # multiple bboxes [[...],[...]]
        elif isinstance(parsed, (list, tuple)) and len(parsed) > 0 and isinstance(parsed[0], (list, tuple)):
            for bb in parsed:
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    bboxes.append([int(round(bb[0])), int(round(bb[1])), int(round(bb[2])), int(round(bb[3]))])

    return img_path, bboxes


def extract_question_text_cot(sample: Dict[str, Any]) -> str:
    conv = sample.get("conversations", [])
    q = ""
    for m in conv:
        if m.get("from") == "human":
            q = m.get("value", "")
            break
    q = q.replace("<image>\n", "").replace("<image>", "").strip()
    q = re.split(r"\s+Please provide the bounding box coordinate", q, maxsplit=1)[0].strip()
    return q


def extract_final_answer_cot(sample: Dict[str, Any]) -> str:
    conv = sample.get("conversations", [])
    ans = ""
    for m in reversed(conv):
        if m.get("from") != "gpt":
            continue
        v = (m.get("value") or "").strip()
        if v.startswith("[") and v.endswith("]"):
            continue
        ans = v
        break
    return ans


def extract_first_gpt_bbox_norm(sample: Dict[str, Any]) -> Optional[List[float]]:
    conv = sample.get("conversations", [])
    for m in conv:
        if m.get("from") != "gpt":
            continue
        v = (m.get("value") or "").strip()
        if not (v.startswith("[") and v.endswith("]")):
            continue
        try:
            arr = ast.literal_eval(v)
        except Exception:
            continue
        if isinstance(arr, (list, tuple)) and len(arr) == 4 and all(isinstance(x, (int, float)) for x in arr):
            return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]
    return None


def norm_xyxy_to_abs(norm_xyxy: List[float], w: int, h: int) -> List[int]:
    x1n, y1n, x2n, y2n = norm_xyxy
    return [
        int(round(x1n * w)),
        int(round(y1n * h)),
        int(round(x2n * w)),
        int(round(y2n * h)),
    ]


def get_image_rel_and_bboxes(sample: Dict[str, Any], img_size: Tuple[int, int]) -> Tuple[str, List[List[int]], str, str, str]:
    """
    Unify both formats:
      - returns rel_image_path, bboxes_abs_xyxy, title_question, title_answer, sample_id_str
    """
    w, h = img_size

    # Old/simple format
    if "bboxs" in sample and "question" in sample:
        rel = sample["image"]
        bboxes = sample.get("bboxs", [])
        q = sample.get("question", "")
        a = sample.get("answer", "")
        sid = f"bbx_{abs(hash((rel, q, a))) % (10**8)}"
        return rel, bboxes, q, a, sid

    # CoT/new format
    if "image" in sample and "conversations" in sample:
        rel, bboxes = parse_bboxes_from_image_field(sample["image"])
        q = extract_question_text_cot(sample)
        a = extract_final_answer_cot(sample)
        sid = f"qid_{sample.get('question_id','NA')}"
        # Fallback: if no abs bbox from image field, use first GPT bbox (assume normalized)
        if len(bboxes) == 0:
            norm = extract_first_gpt_bbox_norm(sample)
            if norm is not None:
                bboxes = [norm_xyxy_to_abs(norm, w, h)]
        return rel, bboxes, q, a, sid

    raise ValueError("Unknown sample format")


def draw_sample(img_path: Path, q: str, a: str, sid: str, bboxes_xyxy: List[List[int]], color: str, out_path: Path) -> None:
    img = Image.open(img_path).convert("RGB")
    fig = plt.figure(figsize=(12, 9), dpi=200)
    ax = plt.gca()
    ax.imshow(img)
    ax.axis("off")

    ax.set_title(
        f"{sid} | color={color}\nQ: {q}\nA: {a} | boxes={len(bboxes_xyxy)}",
        fontsize=10,
    )

    for i, (x1, y1, x2, y2) in enumerate(bboxes_xyxy, start=1):
        x1f, y1f, x2f, y2f = map(float, (x1, y1, x2, y2))
        rect = patches.Rectangle(
            (x1f, y1f),
            max(1.0, x2f - x1f),
            max(1.0, y2f - y1f),
            linewidth=3,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1f,
            max(0, y1f - 6),
            str(i),
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor=color, alpha=0.9),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", required=True, help="Base folder prepended to each sample's image relative path.")
    ap.add_argument("--out-dir", required=True, help="Output folder for visualization PNGs.")
    ap.add_argument("--dedup", action="store_true", help="Deduplicate identical bboxes before drawing.")
    args = ap.parse_args()

    image_root = Path(args.image_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    palette = ["red", "blue", "lime", "magenta", "cyan", "yellow"]
    color_idx = 0

    for s in HARD_CODED_SAMPLES:
        # Determine image path first to get size for normalization fallback
        # For CoT format, image rel is in s["image"][0]; for bbx format, it's s["image"].
        if isinstance(s.get("image"), list):
            rel_guess = s["image"][0]
        else:
            rel_guess = s["image"]

        img_path = (image_root / rel_guess).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path)
        w, h = img.size

        rel, bboxes, q, a, sid = get_image_rel_and_bboxes(s, (w, h))

        # Sanity: ensure rel matches the actual file we opened
        img_path = (image_root / rel).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        if args.dedup:
            bboxes = dedup_bboxes_xyxy(bboxes)

        color = palette[color_idx % len(palette)]
        color_idx += 1

        safe_sid = re.sub(r"[^a-zA-Z0-9_\-]+", "_", sid)
        out_path = out_dir / f"{safe_sid}_{color}.png"
        draw_sample(img_path, q=q, a=a, sid=sid, bboxes_xyxy=bboxes, color=color, out_path=out_path)
        print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()



'''
python vis_two_docvqa_bboxes.py --image-root ./ --out-dir ./ --dedup
'''
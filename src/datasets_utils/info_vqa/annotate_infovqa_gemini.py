#!/usr/bin/env python3
"""
Refine InfographicsVQA bbox annotations with Gemini, starting from seed bboxes in JSONL.

Input JSONL row example:
{
  "question": "...",
  "answer": "...",
  "possible_answers": [...],
  "image": "20471.jpeg",
  "width": 596,
  "height": 5107,
  "dataset": "infographicsvqa",
  "split": "train",
  "bboxes": [{"bbox":[ymin,xmin,ymax,xmax]}, ...]   # in [0,1000]
}

Output JSONL row:
- keeps all original fields
- adds "seed_bboxes" (copy of input bboxes)
- overwrites "bboxes" with Gemini-refined labeled boxes
- adds "answer_type" if provided by Gemini
- adds "token_usage"
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from google import genai
from google.genai import types


PROMPT_TEMPLATE = """You are refining grounding bounding boxes for InfographicsVQA-style questions on infographic images.

Infographics communicate information using a combination of textual, graphical, and visual elements.
A question may require:
(i) text reading, (ii) table/list lookup, (iii) charts/figures/maps, and/or (iv) visual/layout cues.

You are given a set of SEED bounding boxes that are likely related to the answer but may be noisy.
Seed bbox format:
- bbox: [ymin, xmin, ymax, xmax]
- coordinates are normalized to [0,1000]
So you can use them directly, but you must verify them visually.

Input:
- question: {question}
- possible_answers (any of these can be correct): {possible_answers}
- seed_bboxes (noisy, may include duplicates/wrong/loose): {seed_bboxes}

Your job (do this in order):
1) Examine the seed boxes first and KEEP the useful/correct ones.
2) If a kept box is not tight (too small/too large), ADJUST it to be tight.
3) If a seed box is wrong/irrelevant, REMOVE it.
4) If key context is missing (labels, legend, headers, axis, category names, regions, prerequisite numbers),
   ADD new boxes to make the reasoning unambiguous.

Labeling rules:
- If a box contains text that you will use, set label to the EXACT text inside the box.
- If evidence is visual (icons/objects) or needs enumeration (e.g., counting cars), label as:
  "car 1", "car 2", ... (or "dot 1", "person 1", etc.), using the target name from the question.
- Tight boxes only; avoid unrelated neighbors.

Semantic flag (IMPORTANT):
- For each output box, add is_semantic with the following rule:
  * is_semantic = false (extractive) if the label is explicit text that appears inside the box and you are using that text,
    EVEN IF that extracted text is later used in a computation (sorting/arithmetic/aggregation).
  * is_semantic = true (semantic) if you are NOT using extracted text from the box, but using the region as a whole, e.g.:
      - counting/identifying visual icons or elements,
      - counting text tokens as objects without reading their exact content,
      - referring to visual marks, dots, bars, pictograms, shapes, map regions, or any element where no text is extracted.
  * Visual icons/elements with no extracted text are normally semantic (is_semantic = true).

Also output an overall answer_type:
- "extractive" if the final answer can be read directly from text in the image.
- "semantic" if the final answer requires computation/aggregation (counting/sorting/arithmetic) or combining multiple evidence items.

Output JSON ONLY (no markdown). Use this exact schema:

{{
  "answer_type": "extractive|semantic",
  "bboxes": [
    {{
      "role": "context|answer",
      "label": "<exact text or enumerated name or empty>",
      "bbox": [ymin, xmin, ymax, xmax],
      "is_semantic": true|false
    }}
  ]
}}

Constraints:
- bbox coordinates must remain in [0,1000].
- Do not output boxes outside the image.
"""



# ---------- thread-local client ----------
_tls = threading.local()

def get_client(api_key: Optional[str]) -> genai.Client:
    if getattr(_tls, "client", None) is None:
        _tls.client = genai.Client(api_key=api_key) if api_key else genai.Client()
    return _tls.client


# ---------- helpers ----------
def guess_mime(image_path: str) -> str:
    mt, _ = mimetypes.guess_type(image_path)
    if mt:
        return mt
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    t = text.strip()

    # strip ```json fences if present
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()

    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and r > l:
        sub = t[l:r + 1]
        try:
            obj = json.loads(sub)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None

def usage_to_dict(usage: Any) -> Optional[Dict[str, Any]]:
    if usage is None:
        return None

    def g(name: str) -> Any:
        return getattr(usage, name, None)

    out = {
        "prompt_token_count": g("prompt_token_count"),
        "candidates_token_count": g("candidates_token_count"),
        "total_token_count": g("total_token_count"),
        "thoughts_token_count": g("thoughts_token_count"),
        "cached_content_token_count": g("cached_content_token_count"),
        "tool_use_prompt_token_count": g("tool_use_prompt_token_count"),
    }
    out = {k: v for k, v in out.items() if v is not None}
    return out or None


def _normalize_seed_bboxes(seed: Any, max_seed: int) -> List[Dict[str, Any]]:
    """
    Accepts seed as:
      - [{"bbox":[...]} ...]
      - [[...], ...]
    Returns: [{"bbox":[ymin,xmin,ymax,xmax]} ...]
    """
    out: List[Dict[str, Any]] = []
    if not seed:
        return out

    if isinstance(seed, list):
        for item in seed:
            if len(out) >= max_seed:
                break
            if isinstance(item, dict) and isinstance(item.get("bbox", None), (list, tuple)) and len(item["bbox"]) == 4:
                out.append({"bbox": list(item["bbox"])})
            elif isinstance(item, (list, tuple)) and len(item) == 4:
                out.append({"bbox": list(item)})
    return out


def call_gemini_refine(
    *,
    api_key: Optional[str],
    model: str,
    thinking_level: str,
    image_path: str,
    question: str,
    possible_answers: List[str],
    seed_bboxes: List[Dict[str, Any]],
    max_retries: int,
    base_sleep: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        possible_answers=json.dumps(possible_answers, ensure_ascii=False),
        seed_bboxes=json.dumps(seed_bboxes, ensure_ascii=False),
    )

    mime_type = guess_mime(image_path)
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    client = get_client(api_key)
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
    )

    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
                config=config,
            )
            parsed = extract_json_from_text(resp.text)
            usage = usage_to_dict(getattr(resp, "usage_metadata", None))
            return parsed, usage
        except Exception:
            if attempt >= max_retries:
                break
            sleep_s = base_sleep * (2 ** attempt) * (0.7 + 0.6 * random.random())
            time.sleep(sleep_s)

    return None, None


def _valid_bbox_list(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and len(x) == 4 and all(isinstance(v, (int, float)) for v in x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL with seed bboxes")
    ap.add_argument("--output", required=True, help="Output JSONL (streaming append)")
    ap.add_argument("--image-root", required=True, help="Folder containing images referenced by 'image'")
    ap.add_argument("--model", default="gemini-3-flash-preview", help="Model id")
    ap.add_argument("--thinking-level", default="medium", help="minimal|low|medium|high")
    ap.add_argument("--threads", type=int, default=8, help="Worker threads")
    ap.add_argument("--max-inflight", type=int, default=64, help="Max in-flight requests")
    ap.add_argument("--max-retries", type=int, default=5, help="Retries per request")
    ap.add_argument("--base-sleep", type=float, default=0.8, help="Base backoff seconds")
    ap.add_argument("--api-key", default=None, help="Optional API key (else use env GEMINI_API_KEY/GOOGLE_API_KEY)")
    ap.add_argument("--max-seed-boxes", type=int, default=50, help="Max seed boxes to include in prompt")
    ap.add_argument("--no-total", action="store_true", help="Do not pre-count lines for tqdm total")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    write_lock = threading.Lock()
    fout = open(args.output, "a", encoding="utf-8", buffering=1)  # line-buffered

    def write_row(payload: Dict[str, Any]) -> None:
        with write_lock:
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")

    total = None
    if not args.no_total:
        try:
            with open(args.input, "rb") as f:
                total = sum(1 for _ in f)
        except Exception:
            total = None

    processed = 0
    ok = 0
    fail = 0
    skipped = 0

    inflight: Dict[Any, Dict[str, Any]] = {}

    def submit_one(executor: ThreadPoolExecutor, row: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[Any]]:
        image_name = row.get("image", "")
        if not isinstance(image_name, str) or not image_name:
            return "NO_IMAGE", row, None

        image_path = os.path.join(args.image_root, image_name)
        if not os.path.isfile(image_path):
            return "NO_IMAGE", row, None

        question = row.get("question", "") or ""
        possible_answers = row.get("possible_answers", []) or []
        if not isinstance(possible_answers, list):
            possible_answers = [str(possible_answers)]

        seed = _normalize_seed_bboxes(row.get("bboxes", []) or [], args.max_seed_boxes)

        # keep a copy for debugging
        row_out = dict(row)
        row_out["seed_bboxes"] = seed

        fut = executor.submit(
            call_gemini_refine,
            api_key=args.api_key,
            model=args.model,
            thinking_level=args.thinking_level,
            image_path=image_path,
            question=question,
            possible_answers=possible_answers,
            seed_bboxes=seed,
            max_retries=args.max_retries,
            base_sleep=args.base_sleep,
        )
        return "OK", row_out, fut

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        with open(args.input, "r", encoding="utf-8") as fin:
            pbar = tqdm(fin, total=total, desc="Refining bboxes (JSONL)", dynamic_ncols=True)
            for line in pbar:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    skipped += 1
                    continue
                if not isinstance(row, dict):
                    skipped += 1
                    continue

                status, payload, fut = submit_one(ex, row)
                if status == "NO_IMAGE" or fut is None:
                    payload["token_usage"] = None
                    # keep original bboxes if we cannot process
                    write_row(payload)
                    processed += 1
                    fail += 1
                    pbar.set_postfix_str(f"ok={ok} fail={fail} skipped={skipped} done={processed}")
                    continue

                inflight[fut] = payload

                while len(inflight) >= args.max_inflight:
                    done, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
                    for h in done:
                        payload2 = inflight.pop(h)
                        try:
                            parsed, usage = h.result()
                        except Exception:
                            parsed, usage = None, None

                        payload2["token_usage"] = usage

                        if parsed and isinstance(parsed.get("bboxes", None), list):
                            # validate and keep only valid boxes
                            refined: List[Dict[str, Any]] = []
                            for bb in parsed["bboxes"]:
                                if not isinstance(bb, dict):
                                    continue
                                bbox = bb.get("bbox", None)
                                if not _valid_bbox_list(bbox):
                                    continue
                                refined.append({
                                    "role": bb.get("role", "context"),
                                    "label": bb.get("label", ""),
                                    "bbox": list(bbox),
                                    "is_semantic": bool(bb.get("is_semantic", False)),
                                })
                            payload2["bboxes"] = refined
                            if "answer_type" in parsed:
                                payload2["answer_type"] = parsed["answer_type"]
                            ok += 1
                        else:
                            fail += 1

                        write_row(payload2)
                        processed += 1
                        pbar.set_postfix_str(f"ok={ok} fail={fail} skipped={skipped} done={processed}")

            # drain
            while inflight:
                done, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
                for h in done:
                    payload2 = inflight.pop(h)
                    try:
                        parsed, usage = h.result()
                    except Exception:
                        parsed, usage = None, None

                    payload2["token_usage"] = usage

                    if parsed and isinstance(parsed.get("bboxes", None), list):
                        refined: List[Dict[str, Any]] = []
                        for bb in parsed["bboxes"]:
                            if not isinstance(bb, dict):
                                continue
                            bbox = bb.get("bbox", None)
                            if not _valid_bbox_list(bbox):
                                continue
                            refined.append({
                                "role": bb.get("role", "context"),
                                "label": bb.get("label", ""),
                                "bbox": list(bbox),
                                "is_semantic": bool(bb.get("is_semantic", False)),
                            })
                        payload2["bboxes"] = refined
                        if "answer_type" in parsed:
                            payload2["answer_type"] = parsed["answer_type"]
                        ok += 1
                    else:
                        fail += 1

                    write_row(payload2)
                    processed += 1
                    pbar.set_postfix_str(f"ok={ok} fail={fail} skipped={skipped} done={processed}")

    fout.close()
    print(f"Done. ok={ok} fail={fail} skipped={skipped} total_written={processed}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()


'''
python annotate_infovqa_gemini.py \
  --input ./tmp.json \
  --output ./tmp_annotated.jsonl \
  --image-root ./infographicsvqa_images \
  --model gemini-3-flash-preview \
  --thinking-level medium \
  --threads 32 \
  --max-inflight 128

python annotate_infovqa_gemini.py --input ./infographicsvqa_cot_train_gemini_4filtered_part1.jsonl --output ./infographicsvqa_cot_train_gemini_4filtered_part1_annotated.jsonl --image-root ./infographicsvqa_images --model gemini-3-flash-preview --thinking-level high --threads 32 --max-inflight 128 --api-key <YOUR_API_KEY>


python annotate_infovqa_gemini.py --input ./infographicsvqa_cot_train_gemini_4filtered_part2.jsonl --output ./infographicsvqa_cot_train_gemini_4filtered_part2_annotated.jsonl --image-root ./infographicsvqa_images --model gemini-3-flash-preview --thinking-level high --threads 32 --max-inflight 128 --api-key <YOUR_API_KEY>

'''
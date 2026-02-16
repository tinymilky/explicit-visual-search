#!/usr/bin/env python3
"""
Annotate DocVQA (question, answer, image) triplets with Gemini bounding boxes.

- Model: gemini-3-flash-preview
- Thinking level: medium
- Parallelism: multi-threaded (ThreadPoolExecutor)
- Output: streaming JSONL writes as soon as each API call returns

Input JSONL rows may contain extra keys; we only read:
  question, answer, image, width, height

Output JSONL rows contain:
  question, answer, image, width, height, bboxes, token_usage

Setup:
  pip install -U google-genai tqdm

Auth:
  export GEMINI_API_KEY="..."   # or GOOGLE_API_KEY
"""

import argparse
import json
import mimetypes
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from google import genai
from google.genai import types


PROMPT_TEMPLATE = """You are annotating answer-grounding bounding boxes in a document image.

Input (triplet):
- question: {question}
- answer: {answer}

Task:
Return the minimal set of bounding boxes that justify the answer.

Rules:
- If the answer is a direct text span, output ONE tight box covering the answer text.
- If the answer requires context (e.g., “which month/section/category” or table lookup), output the minimal set of boxes:
  1) header/label/context that determines the answer,
  2) row/column label(s) needed to locate the answer (for tables),
  3) the specific entry/value supporting the answer.
- Use bbox format: [ymin, xmin, ymax, xmax] with coordinates rescaled to [0,1000].
- The "label" must be the EXACT text contained in that box (no placeholders).
- Tight boxes only; avoid including unrelated neighboring lines.
- No OCR/tools.

Output JSON only:
{{"bboxes":[{{"role":"context|answer","label":"<exact text>","bbox":[ymin,xmin,ymax,xmax]}}]}}
"""


# ---------- thread-local client ----------
_tls = threading.local()

def get_client(api_key: Optional[str]) -> genai.Client:
    if getattr(_tls, "client", None) is None:
        _tls.client = genai.Client(api_key=api_key) if api_key else genai.Client()
    return _tls.client


# ---------- helpers ----------
def count_lines(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)

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


def call_gemini_annotate(
    *,
    api_key: Optional[str],
    model: str,
    thinking_level: str,
    image_path: str,
    question: str,
    answer: str,
    max_retries: int,
    base_sleep: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    prompt = PROMPT_TEMPLATE.format(question=question, answer=answer)
    mime_type = guess_mime(image_path)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    client = get_client(api_key)
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        # keep other params default
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input jsonl")
    ap.add_argument("--output", required=True, help="Output jsonl (streaming append)")
    ap.add_argument("--image-root", required=True, help="Folder containing images referenced by jsonl 'image'")
    ap.add_argument("--model", default="gemini-3-flash-preview", help="Model id")
    ap.add_argument("--thinking-level", default="medium", help="minimal|low|medium|high")
    ap.add_argument("--threads", type=int, default=8, help="Worker threads")
    ap.add_argument("--max-inflight", type=int, default=64, help="Max in-flight requests")
    ap.add_argument("--max-retries", type=int, default=5, help="Retries per request")
    ap.add_argument("--base-sleep", type=float, default=0.8, help="Base backoff seconds")
    ap.add_argument("--api-key", default=None, help="Optional API key (else use env GEMINI_API_KEY/GOOGLE_API_KEY)")
    ap.add_argument("--no-total", action="store_true", help="Do not pre-count lines for tqdm total")
    args = ap.parse_args()

    total = None if args.no_total else count_lines(args.input)

    write_lock = threading.Lock()
    processed = 0
    ok = 0
    fail = 0
    skipped = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fout = open(args.output, "a", encoding="utf-8", buffering=1)  # line-buffered

    # future -> payload
    inflight: Dict[Any, Dict[str, Any]] = {}

    def make_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question": obj.get("question", ""),
            "answer": obj.get("answer", ""),
            "image": obj.get("image", ""),
            "width": obj.get("width", None),
            "height": obj.get("height", None),
        }

    def submit_one(executor: ThreadPoolExecutor, obj: Dict[str, Any]) -> Optional[Any]:
        payload = make_payload(obj)
        image = payload["image"]
        if not isinstance(image, str) or not image:
            return ("NO_IMAGE", payload)

        image_path = os.path.join(args.image_root, image)
        if not os.path.isfile(image_path):
            return ("NO_IMAGE", payload)

        fut = executor.submit(
            call_gemini_annotate,
            api_key=args.api_key,
            model=args.model,
            thinking_level=args.thinking_level,
            image_path=image_path,
            question=payload["question"],
            answer=payload["answer"],
            max_retries=args.max_retries,
            base_sleep=args.base_sleep,
        )
        return (fut, payload)

    def write_row(payload: Dict[str, Any]) -> None:
        with write_lock:
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=args.threads) as ex, open(args.input, "r", encoding="utf-8") as fin:
        pbar = tqdm(fin, total=total, desc="Annotating", dynamic_ncols=True)

        for line in pbar:
            line = line.strip()
            if not line:
                skipped += 1
                continue
            try:
                obj = json.loads(line)
            except Exception:
                skipped += 1
                continue

            handle = submit_one(ex, obj)
            if handle is None:
                skipped += 1
                continue

            fut_or_flag, payload = handle

            if fut_or_flag == "NO_IMAGE":
                payload["bboxes"] = []
                payload["token_usage"] = None
                write_row(payload)
                processed += 1
                fail += 1
                pbar.set_postfix_str(f"ok={ok} fail={fail} skipped={skipped} done={processed}")
                continue

            inflight[fut_or_flag] = payload

            # bound inflight
            while len(inflight) >= args.max_inflight:
                done, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
                for h in done:
                    payload2 = inflight.pop(h)
                    try:
                        parsed, usage = h.result()
                    except Exception:
                        parsed, usage = None, None

                    if parsed and isinstance(parsed.get("bboxes", None), list):
                        payload2["bboxes"] = parsed["bboxes"]
                        payload2["token_usage"] = usage
                        ok += 1
                    else:
                        payload2["bboxes"] = []
                        payload2["token_usage"] = usage
                        fail += 1

                    write_row(payload2)
                    processed += 1
                    pbar.set_postfix_str(f"ok={ok} fail={fail} skipped={skipped} done={processed}")

        # flush remaining
        while inflight:
            done, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
            for h in done:
                payload2 = inflight.pop(h)
                try:
                    parsed, usage = h.result()
                except Exception:
                    parsed, usage = None, None

                if parsed and isinstance(parsed.get("bboxes", None), list):
                    payload2["bboxes"] = parsed["bboxes"]
                    payload2["token_usage"] = usage
                    ok += 1
                else:
                    payload2["bboxes"] = []
                    payload2["token_usage"] = usage
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
python annotate_docvqa_gemini.py \
  --input ./tmp.jsonl \
  --output ./tmp_annotated.jsonl \
  --image-root ./docvqa_imgs \
  --model gemini-3-pro-preview \
  --thinking-level high \
  --threads 32 \
  --api-key <your_api_key_here> \
  --max-inflight 128

python annotate_docvqa_gemini.py --input ./tmp.jsonl --output ./tmp_annotated.jsonl --image-root ./docvqa_imgs --model gemini-3-flash-preview --thinking-level high --threads 32 --api-key <YOUR_API_KEY> --max-inflight 128
'''
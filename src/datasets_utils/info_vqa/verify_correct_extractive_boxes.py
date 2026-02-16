#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3 (tweaked): Verify/correct Gemini extractive bboxes (is_semantic=false) using OCR word-level JSONL.

New logic additions:
1) If OCR anchor hits exist, relax (not remove) acceptance thresholds (cov/cr/dist) to rescue near-misses.
2) If reason == "no_anchor_hits" for an extractive box -> definite fail (no rescue).
3) After matching:
   - If a failed box is NOT an answer box (role != "answer"), DROP it from output.
4) If a failed box IS an answer box:
   - If question starts with "how many":
       * parse integer from row["answer"] or row["possible_answers"]
       * if (# answer boxes in row) == parsed answer -> KEEP Gemini answer boxes and do NOT mark row as failure
       * else -> row is a failure
   - Else (question not "how many") -> row is a failure

Outputs:
- --output : all rows, with corrected boxes; failed non-answer boxes removed
- --fail-out : only rows considered failure under rule (4)
- --report-out : per-box diagnostics

Requires OCR word-level JSONL per image:
  ocr_word_root / f"{Path(image).stem}.jsonl"
Each line: {"word_idx":..., "text":..., "bbox":[ymin,xmin,ymax,xmax], "line_idx": optional, ...}
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


# ---------------------------
# Normalization / token logic
# ---------------------------

STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "for", "with", "at", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those",
}

NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "a": 1, "an": 1
}

_re_num = re.compile(r"[0-9]")
_re_non_alnum = re.compile(r"[^a-z0-9]+")
_re_keep_numdot = re.compile(r"[^0-9.]+")
_re_decimal_comma = re.compile(r"(?<=\d),(?=\d)")  # comma between digits


def norm_basic(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return s.lower().strip()


def tokenize(s: str) -> List[str]:
    s = norm_basic(s)
    return [t for t in s.split() if t]


def alnum_skeleton(s: str) -> str:
    s = norm_basic(s)
    s = _re_non_alnum.sub("", s)  # keep a-z0-9 only
    return s


def numeric_skeleton(s: str) -> str:
    s = norm_basic(s)
    s = _re_decimal_comma.sub(".", s)
    s = _re_keep_numdot.sub("", s)  # keep digits and dots
    if s.count(".") <= 1:
        return s
    first = s.find(".")
    return s[:first + 1] + s[first + 1:].replace(".", "")


def is_numericish(s: str) -> bool:
    return bool(_re_num.search(norm_basic(s)))


def token_equivalent(a: str, b: str) -> bool:
    if is_numericish(a) and is_numericish(b):
        na, nb = numeric_skeleton(a), numeric_skeleton(b)
        if na and nb and na == nb:
            return True
    return alnum_skeleton(a) == alnum_skeleton(b)


def char_ratio_skeleton(a: str, b: str) -> float:
    aa = alnum_skeleton(a)
    bb = alnum_skeleton(b)
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb).ratio()


def choose_anchor_tokens(label_tokens: List[str], max_anchors: int = 5) -> List[str]:
    scored: List[Tuple[float, str]] = []
    for t in label_tokens:
        if not t or t in STOPWORDS:
            continue
        sk = alnum_skeleton(t)
        if not sk:
            continue
        score = 0.0
        if is_numericish(t):
            score += 10.0
        score += min(len(sk), 12) / 12.0 * 2.0
        scored.append((score, t))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [t for _, t in scored[:max_anchors]]


def parse_answer_int_from_row(row: Dict[str, Any]) -> Optional[int]:
    # Try row["answer"] then row["possible_answers"].
    candidates: List[str] = []
    ans = row.get("answer", None)
    if isinstance(ans, str) and ans.strip():
        candidates.append(ans)

    pa = row.get("possible_answers", None)
    if isinstance(pa, list):
        for x in pa:
            if isinstance(x, str) and x.strip():
                candidates.append(x)

    for s in candidates:
        ss = norm_basic(s)
        m = re.search(r"\d+", ss)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                pass
        for t in tokenize(ss):
            if t in NUM_WORDS:
                return NUM_WORDS[t]
    return None


def is_how_many_question(q: str) -> bool:
    return isinstance(q, str) and q.strip().lower().startswith("how many")


# ---------------------------
# Geometry / bbox
# ---------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def bbox_union_yxyx(boxes: List[List[float]]) -> Optional[List[float]]:
    if not boxes:
        return None
    ys = [b[0] for b in boxes]
    xs = [b[1] for b in boxes]
    ye = [b[2] for b in boxes]
    xe = [b[3] for b in boxes]
    return [min(ys), min(xs), max(ye), max(xe)]


def bbox_iou_yxyx(a: List[float], b: List[float]) -> float:
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    inter_y1 = max(ay1, by1)
    inter_x1 = max(ax1, bx1)
    inter_y2 = min(ay2, by2)
    inter_x2 = min(ax2, bx2)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter = inter_h * inter_w
    area_a = max(0.0, ay2 - ay1) * max(0.0, ax2 - ax1)
    area_b = max(0.0, by2 - by1) * max(0.0, bx2 - bx1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def bbox_center_yx(b: List[float]) -> Tuple[float, float]:
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)


def bbox_center_dist_norm(a: List[float], b: List[float]) -> float:
    # normalize by diagonal of [0,1000]^2
    ay, ax = bbox_center_yx(a)
    by, bx = bbox_center_yx(b)
    return math.hypot(ay - by, ax - bx) / 1414.21356237


def round_bbox(b: List[float], ndigits: int = 2) -> List[float]:
    return [round(float(x), ndigits) for x in b]


# ---------------------------
# OCR document structures
# ---------------------------

@dataclass
class OCRToken:
    tok: str
    word_idx: int
    pos: int


@dataclass
class OCRWord:
    text: str
    bbox: List[float]  # yxyx in [0,1000]
    line_idx: Optional[int] = None
    conf: Optional[float] = None


@dataclass
class OCRDoc:
    doc_id: str
    words: List[OCRWord]
    tokens: List[OCRToken]
    inv: Dict[str, List[int]]                 # token skeleton -> token positions
    word_tokens_range: List[Tuple[int, int]]  # per word_idx: token [start,end)


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _extract_word_rows(ocr_word_jsonl: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(ocr_word_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


@lru_cache(maxsize=256)
def load_ocr_doc(ocr_word_jsonl: str) -> Optional[OCRDoc]:
    p = Path(ocr_word_jsonl)
    if not p.is_file():
        return None

    rows = _extract_word_rows(p)
    if not rows:
        return None

    words: List[OCRWord] = []
    tokens: List[OCRToken] = []
    inv: Dict[str, List[int]] = {}
    word_tokens_range: List[Tuple[int, int]] = []

    pos = 0
    for r in rows:
        text = str(r.get("text", "") or "")
        bbox = r.get("bbox", None)
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        bbox_f = [clamp(float(x), 0.0, 1000.0) for x in bbox]

        li = r.get("line_idx", None)
        li_i = _safe_int(li) if li is not None else None
        conf = _safe_float(r.get("confidence", None))

        words.append(OCRWord(text=text, bbox=bbox_f, line_idx=li_i, conf=conf))

        start_pos = pos
        # A WORD row may contain multiple tokens like "OF USERS"
        for t in tokenize(text):
            sk = alnum_skeleton(t)
            if sk:
                inv.setdefault(sk, []).append(pos)
            tokens.append(OCRToken(tok=t, word_idx=len(words) - 1, pos=pos))
            pos += 1
        end_pos = pos
        word_tokens_range.append((start_pos, end_pos))

    return OCRDoc(doc_id=p.stem, words=words, tokens=tokens, inv=inv, word_tokens_range=word_tokens_range)


# ---------------------------
# Matching / correction
# ---------------------------

@dataclass
class MatchResult:
    ok: bool
    score: float
    coverage: float
    char_ratio: float
    iou: float
    dist: float
    matched_word_indices: List[int]
    corrected_bbox: Optional[List[float]]
    reason: str


def _label_tokens_for_scoring(label: str) -> List[str]:
    toks = tokenize(label)
    content = [t for t in toks if t not in STOPWORDS and alnum_skeleton(t)]
    return content if content else toks


def _find_anchor_positions(doc: OCRDoc, anchors: List[str]) -> List[int]:
    pos_set = set()
    for a in anchors:
        k = alnum_skeleton(a)
        if not k:
            continue
        for p in doc.inv.get(k, []):
            pos_set.add(p)
    return sorted(pos_set)


def _window_token_range(doc: OCRDoc, center_pos: int, *, window_words: int = 120, window_lines: int = 2) -> Tuple[int, int]:
    if center_pos < 0 or center_pos >= len(doc.tokens):
        return (0, len(doc.tokens))
    center_wi = doc.tokens[center_pos].word_idx
    line_idx = doc.words[center_wi].line_idx

    if line_idx is not None:
        lo_line = line_idx - window_lines
        hi_line = line_idx + window_lines
        word_ids = [i for i, w in enumerate(doc.words) if w.line_idx is not None and lo_line <= w.line_idx <= hi_line]
        if not word_ids:
            lo_w = max(0, center_wi - window_words)
            hi_w = min(len(doc.words) - 1, center_wi + window_words)
        else:
            lo_w = min(word_ids)
            hi_w = max(word_ids)
    else:
        lo_w = max(0, center_wi - window_words)
        hi_w = min(len(doc.words) - 1, center_wi + window_words)

    lo_tok = doc.word_tokens_range[lo_w][0]
    hi_tok = doc.word_tokens_range[hi_w][1]
    lo_tok = int(clamp(lo_tok, 0, len(doc.tokens)))
    hi_tok = int(clamp(hi_tok, 0, len(doc.tokens)))
    return lo_tok, hi_tok


def _subseq_match_from(
    doc: OCRDoc,
    label_toks: List[str],
    start_pos: int,
    end_pos: int,
    max_scan: int,
) -> Tuple[float, List[int], List[int]]:
    i = 0
    j = start_pos
    scanned = 0
    matched_word = set()
    matched_token_positions: List[int] = []

    while j < end_pos and i < len(label_toks) and scanned < max_scan:
        if token_equivalent(label_toks[i], doc.tokens[j].tok):
            matched_word.add(doc.tokens[j].word_idx)
            matched_token_positions.append(j)
            i += 1
        j += 1
        scanned += 1

    matched_count = i
    coverage = 0.0 if len(label_toks) == 0 else matched_count / float(len(label_toks))
    return coverage, sorted(matched_word), matched_token_positions


def _cluster_words(doc: OCRDoc, word_indices: List[int], pred_bbox: Optional[List[float]]) -> Optional[List[float]]:
    if not word_indices:
        return None

    if all(doc.words[i].line_idx is None for i in word_indices):
        return bbox_union_yxyx([doc.words[i].bbox for i in word_indices])

    pairs = sorted(
        [(i, doc.words[i].line_idx) for i in word_indices if doc.words[i].line_idx is not None],
        key=lambda x: x[1]
    )

    clusters: List[List[int]] = []
    cur: List[int] = []
    prev_li: Optional[int] = None
    for wi, li in pairs:
        if prev_li is None or li == prev_li or li == prev_li + 1:
            cur.append(wi)
        else:
            clusters.append(cur)
            cur = [wi]
        prev_li = li
    if cur:
        clusters.append(cur)

    best_bbox = None
    best_score = -1e9
    for cl in clusters:
        cl_box = bbox_union_yxyx([doc.words[i].bbox for i in cl])
        if cl_box is None:
            continue
        if pred_bbox is not None:
            iou = bbox_iou_yxyx(pred_bbox, cl_box)
            dist = bbox_center_dist_norm(pred_bbox, cl_box)
            score = iou - 0.1 * dist + 0.005 * len(cl)
        else:
            score = float(len(cl))
        if score > best_score:
            best_score = score
            best_bbox = cl_box
    return best_bbox


def match_label_to_ocr(
    doc: OCRDoc,
    label: str,
    pred_bbox: Optional[List[float]],
    *,
    window_words: int = 120,
    window_lines: int = 2,
    lambda_iou: float = 0.15,
    lambda_dist: float = 0.10,
    max_candidates: int = 120,
    max_scan_mult: int = 6,
) -> MatchResult:
    label_toks = _label_tokens_for_scoring(label)
    if not label_toks:
        return MatchResult(False, 0.0, 0.0, 0.0, 0.0, 1.0, [], None, "empty_label_tokens")

    anchors = choose_anchor_tokens(label_toks, max_anchors=5)
    anchor_positions = _find_anchor_positions(doc, anchors)

    if not anchor_positions:
        # definite fail case per your rule
        return MatchResult(False, 0.0, 0.0, 0.0, 0.0, 1.0, [], None, "no_anchor_hits")

    if len(anchor_positions) > max_candidates:
        step = max(1, len(anchor_positions) // max_candidates)
        anchor_positions = anchor_positions[::step][:max_candidates]

    best = MatchResult(False, -1e9, 0.0, 0.0, 0.0, 1.0, [], None, "init")

    require_numeric = any(is_numericish(t) for t in label_toks)

    for ap in anchor_positions:
        lo, hi = _window_token_range(doc, ap, window_words=window_words, window_lines=window_lines)

        for back in (0, 5, 10, 20):
            start = max(lo, ap - back)
            max_scan = min(hi - start, max_scan_mult * len(label_toks) + 50)

            cov, widxs, matched_token_pos = _subseq_match_from(doc, label_toks, start, hi, max_scan)
            if not widxs:
                continue

            if require_numeric:
                ok_num = any(is_numericish(doc.tokens[p].tok) for p in matched_token_pos)
                if not ok_num:
                    cov *= 0.6

            cand_text = " ".join(doc.words[i].text for i in widxs)
            cr = char_ratio_skeleton(label, cand_text)

            cand_bbox = _cluster_words(doc, widxs, pred_bbox)
            if cand_bbox is None:
                continue
            cand_bbox = [clamp(x, 0.0, 1000.0) for x in cand_bbox]

            if pred_bbox is not None:
                iou = bbox_iou_yxyx(pred_bbox, cand_bbox)
                dist = bbox_center_dist_norm(pred_bbox, cand_bbox)
            else:
                iou = 0.0
                dist = 1.0

            text_score = 0.7 * cov + 0.3 * cr
            final_score = text_score + lambda_iou * iou - lambda_dist * dist

            if final_score > best.score:
                best = MatchResult(True, final_score, cov, cr, iou, dist, widxs, cand_bbox, "matched")

    if not best.ok or best.corrected_bbox is None:
        return MatchResult(False, 0.0, 0.0, 0.0, 0.0, 1.0, [], None, "no_valid_candidate")

    # ---------------------------
    # acceptance thresholds
    # ---------------------------
    n = len(label_toks)

    if n <= 4:
        min_cov = 0.75
        min_cr = 0.55
    elif n <= 10:
        min_cov = 0.65
        min_cr = 0.50
    else:
        min_cov = 0.55
        min_cr = 0.45

    # (A) distance-based relaxation (already)
    dist_relax = 0.06
    cov_relax = 0.15
    cr_relax = 0.10

    min_cov_eff = min_cov
    min_cr_eff = min_cr

    if pred_bbox is not None and best.dist <= dist_relax:
        min_cov_eff = max(0.30, min_cov - cov_relax)
        min_cr_eff = max(0.30, min_cr - cr_relax)

    # Strict gate
    if best.coverage < min_cov_eff and best.char_ratio < min_cr_eff:
        # (B) additional relaxed gate if we had anchor hits (we do here)
        # relax cov/cr AND allow larger distance, but still require some signal
        dist_relax2 = 0.12
        min_cov_rel = max(0.25, min_cov_eff - 0.20)
        min_cr_rel = max(0.25, min_cr_eff - 0.15)

        text_score = 0.7 * best.coverage + 0.3 * best.char_ratio

        dist_ok = True
        if pred_bbox is not None:
            dist_ok = (best.dist <= dist_relax2) or (best.iou >= 0.02)

        if dist_ok and (
            (best.coverage >= min_cov_rel) or
            (best.char_ratio >= min_cr_rel) or
            (text_score >= 0.42 and (best.coverage >= 0.35 or best.char_ratio >= 0.35))
        ):
            best.reason = "ok_relaxed"
            return best

        best.ok = False
        best.reason = f"below_threshold cov={best.coverage:.3f} cr={best.char_ratio:.3f} dist={best.dist:.3f}"
        return best

    best.reason = "ok"
    return best


# ---------------------------
# Main I/O
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Annotated JSONL input")
    ap.add_argument("--output", required=True, help="Corrected JSONL output")
    ap.add_argument("--ocr-word-root", required=True, help="Root folder containing cleaned OCR word JSONLs")
    ap.add_argument("--fail-out", default=None, help="Optional: write rows considered failure under rule (4)")
    ap.add_argument("--report-out", default=None, help="Optional: per-box report JSONL")
    ap.add_argument("--window-words", type=int, default=120)
    ap.add_argument("--window-lines", type=int, default=2)
    ap.add_argument("--lambda-iou", type=float, default=0.15)
    ap.add_argument("--lambda-dist", type=float, default=0.10)
    ap.add_argument("--max-candidates", type=int, default=120)
    ap.add_argument("--max-scan-mult", type=int, default=6)
    ap.add_argument("--add-metadata", action="store_true")
    ap.add_argument("--ndigits", type=int, default=2)
    ap.add_argument("--no-total", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    ocr_root = Path(args.ocr_word_root)
    fail_out = Path(args.fail_out) if args.fail_out else None
    rep_out = Path(args.report_out) if args.report_out else None

    total = None
    if not args.no_total:
        try:
            with open(in_path, "rb") as f:
                total = sum(1 for _ in f)
        except Exception:
            total = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fail_out:
        fail_out.parent.mkdir(parents=True, exist_ok=True)
    if rep_out:
        rep_out.parent.mkdir(parents=True, exist_ok=True)

    fout_fail = open(fail_out, "w", encoding="utf-8") if fail_out else None
    fout_rep = open(rep_out, "w", encoding="utf-8") if rep_out else None

    rows_total = 0
    rows_ok = 0
    rows_fail = 0
    bad_json = 0
    missing_ocr = 0

    boxes_verified = 0
    boxes_corrected = 0
    boxes_failed = 0
    boxes_removed = 0

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(tqdm(fin, total=total, desc="OCR verify/correct (tweaked)", dynamic_ncols=True), start=1):
            line = line.strip()
            if not line:
                continue
            rows_total += 1

            try:
                row = json.loads(line)
            except Exception:
                bad_json += 1
                continue
            if not isinstance(row, dict):
                bad_json += 1
                continue

            img = row.get("image", None)
            bboxes = row.get("bboxes", None)

            # write unchanged if missing essential fields
            if not isinstance(img, str) or not img or not isinstance(bboxes, list):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            doc_id = Path(img).stem
            ocr_path = ocr_root / f"{doc_id}.jsonl"
            doc = load_ocr_doc(str(ocr_path))
            if doc is None:
                missing_ocr += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                if fout_fail:
                    tmp = dict(row)
                    tmp["_ocr_fail_reason"] = "missing_ocr_word_jsonl"
                    fout_fail.write(json.dumps(tmp, ensure_ascii=False) + "\n")
                continue

            question = row.get("question", "")
            how_many = is_how_many_question(question)

            new_bboxes: List[Dict[str, Any]] = []
            failed_answer_boxes = 0

            # pass 1: correct/verify; mark fails; drop failed non-answer boxes
            for bi, bb in enumerate(bboxes):
                if not isinstance(bb, dict):
                    # keep non-dict as-is
                    new_bboxes.append(bb)
                    continue

                role = bb.get("role", None)
                label = bb.get("label", "")
                is_sem = bb.get("is_semantic", None)
                pred_bbox = bb.get("bbox", None)

                # Only process extractive boxes with non-empty label
                if is_sem is not False or not isinstance(label, str) or not label.strip():
                    new_bboxes.append(bb)
                    continue

                pred_bbox_f: Optional[List[float]] = None
                if isinstance(pred_bbox, list) and len(pred_bbox) == 4:
                    try:
                        pred_bbox_f = [clamp(float(x), 0.0, 1000.0) for x in pred_bbox]
                    except Exception:
                        pred_bbox_f = None

                mr = match_label_to_ocr(
                    doc,
                    label,
                    pred_bbox_f,
                    window_words=args.window_words,
                    window_lines=args.window_lines,
                    lambda_iou=args.lambda_iou,
                    lambda_dist=args.lambda_dist,
                    max_candidates=args.max_candidates,
                    max_scan_mult=args.max_scan_mult,
                )

                # definite fail if no_anchor_hits
                if mr.reason == "no_anchor_hits":
                    mr.ok = False

                if mr.ok and mr.corrected_bbox is not None:
                    out_bb = dict(bb)
                    corrected = False
                    if pred_bbox_f is None:
                        corrected = True
                    else:
                        corrected = (bbox_iou_yxyx(pred_bbox_f, mr.corrected_bbox) < 0.6)

                    out_bb["bbox"] = round_bbox(mr.corrected_bbox, ndigits=args.ndigits)
                    if corrected:
                        boxes_corrected += 1
                    else:
                        boxes_verified += 1

                    if args.add_metadata:
                        out_bb["bbox_source"] = "ocr"
                        out_bb["ocr_status"] = "corrected" if corrected else "verified"
                        out_bb["ocr_reason"] = mr.reason
                        out_bb["ocr_score"] = round(mr.score, 4)
                        out_bb["ocr_coverage"] = round(mr.coverage, 4)
                        out_bb["ocr_char_ratio"] = round(mr.char_ratio, 4)
                        out_bb["ocr_iou_with_pred"] = round(mr.iou, 4)
                        out_bb["ocr_dist_to_pred"] = round(mr.dist, 4)
                        out_bb["ocr_matched_word_indices"] = mr.matched_word_indices[:200]

                    new_bboxes.append(out_bb)

                else:
                    boxes_failed += 1

                    # Rule (3): failed non-answer boxes are removed
                    if role != "answer":
                        boxes_removed += 1
                        # optionally report removal
                        if fout_rep:
                            rep = {
                                "line_no": row.get("line_no", line_no),
                                "image": img,
                                "bbox_idx": bi,
                                "role": role,
                                "label": label,
                                "pred_bbox": pred_bbox_f,
                                "ok": False,
                                "reason": mr.reason,
                                "action": "removed_non_answer",
                            }
                            fout_rep.write(json.dumps(rep, ensure_ascii=False) + "\n")
                        continue

                    # failed answer box: keep gemini bbox, but may trigger row failure later
                    failed_answer_boxes += 1
                    out_bb = dict(bb)
                    if args.add_metadata:
                        out_bb["bbox_source"] = "gemini"
                        out_bb["ocr_status"] = "failed"
                        out_bb["ocr_fail_reason"] = mr.reason
                    new_bboxes.append(out_bb)

                # report per box
                if fout_rep:
                    rep = {
                        "line_no": row.get("line_no", line_no),
                        "image": img,
                        "bbox_idx": bi,
                        "role": role,
                        "label": label,
                        "pred_bbox": pred_bbox_f,
                        "ok": mr.ok,
                        "reason": mr.reason,
                        "score": round(mr.score, 6),
                        "coverage": round(mr.coverage, 6),
                        "char_ratio": round(mr.char_ratio, 6),
                        "iou": round(mr.iou, 6),
                        "dist": round(mr.dist, 6),
                        "corrected_bbox": round_bbox(mr.corrected_bbox, ndigits=args.ndigits) if mr.corrected_bbox else None,
                        "matched_word_indices": mr.matched_word_indices[:200],
                    }
                    fout_rep.write(json.dumps(rep, ensure_ascii=False) + "\n")

            # Rule (4): decide whether this row is failure based on failed ANSWER boxes
            row_is_fail = False
            if failed_answer_boxes > 0:
                if how_many:
                    target_n = parse_answer_int_from_row(row)
                    num_answer_boxes = sum(1 for bb in new_bboxes if isinstance(bb, dict) and bb.get("role") == "answer")
                    if target_n is None or num_answer_boxes != target_n:
                        row_is_fail = True
                else:
                    row_is_fail = True

            out_row = dict(row)
            out_row["bboxes"] = new_bboxes

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            if row_is_fail:
                rows_fail += 1
                if fout_fail:
                    fout_fail.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            else:
                rows_ok += 1

    if fout_fail:
        fout_fail.close()
    if fout_rep:
        fout_rep.close()

    print("\nDone.")
    print(f"rows_total={rows_total} rows_ok={rows_ok} rows_fail={rows_fail} bad_json={bad_json}")
    print(f"missing_ocr_files={missing_ocr}")
    print(f"boxes_verified={boxes_verified} boxes_corrected={boxes_corrected} boxes_failed={boxes_failed} boxes_removed={boxes_removed}")
    print(f"output={out_path}")
    if fail_out:
        print(f"fail_out={fail_out}")
    if rep_out:
        print(f"report_out={rep_out}")


if __name__ == "__main__":
    main()


'''
python verify_correct_extractive_boxes.py --input  ./infographicsvqa_vqa_gemini_3fh_raw_gemini_bboxformat.jsonl --output ./ocr_corrected.jsonl --ocr-word-root ./ocr_word_jsonl --fail-out ./ocr_failed.jsonl --report-out ./ocr_report.jsonl --add-metadata

'''
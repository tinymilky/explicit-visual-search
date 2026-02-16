#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DocVQA OCR correction (all-extractive).

CHANGE vs prior version:
- Rows with ANY failed box (or missing OCR / missing image field) are NOT written to --output.
- They are written ONLY to --fail-out (required if you don't want to lose them).

Everything else (thresholds, matching logic) stays the same.
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
# Text normalization
# ---------------------------

STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "for", "with", "at", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those",
}

_re_num = re.compile(r"[0-9]")
_re_non_alnum = re.compile(r"[^a-z0-9]+")
_re_keep_numdot = re.compile(r"[^0-9.]+")
_re_decimal_comma = re.compile(r"(?<=\d),(?=\d)")  # comma between digits


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def norm_basic(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return s.lower().strip()


def tokenize(s: str) -> List[str]:
    s = norm_basic(s)
    return [t for t in s.split() if t]


def alnum_skeleton(s: str) -> str:
    s = norm_basic(s)
    s = _re_non_alnum.sub("", s)
    return s


def numeric_skeleton(s: str) -> str:
    s = norm_basic(s)
    s = _re_decimal_comma.sub(".", s)
    s = _re_keep_numdot.sub("", s)
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


def unique_labels_from_row(row: Dict[str, Any], fallback: str) -> List[str]:
    labels: List[str] = []
    if isinstance(fallback, str) and fallback.strip():
        labels.append(fallback.strip())
    pa = row.get("possible_answers", None)
    if isinstance(pa, list):
        for x in pa:
            if isinstance(x, str) and x.strip():
                labels.append(x.strip())
    seen = set()
    out = []
    for s in labels:
        k = norm_basic(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out[:8]


# ---------------------------
# Geometry helpers
# ---------------------------

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
    ay, ax = bbox_center_yx(a)
    by, bx = bbox_center_yx(b)
    return math.hypot(ay - by, ax - bx) / 1414.21356237


def round_bbox(b: List[float], ndigits: int = 2) -> List[float]:
    return [round(float(x), ndigits) for x in b]


def pixel_xyxy_to_norm_yxyx(b: List[float], w: float, h: float) -> List[float]:
    xmin, ymin, xmax, ymax = map(float, b)
    y1 = clamp((ymin / h) * 1000.0, 0.0, 1000.0)
    x1 = clamp((xmin / w) * 1000.0, 0.0, 1000.0)
    y2 = clamp((ymax / h) * 1000.0, 0.0, 1000.0)
    x2 = clamp((xmax / w) * 1000.0, 0.0, 1000.0)
    if y1 > y2:
        y1, y2 = y2, y1
    if x1 > x2:
        x1, x2 = x2, x1
    return [y1, x1, y2, x2]


def norm_yxyx_to_pixel_xyxy(b: List[float], w: float, h: float, as_int: bool = True) -> List[Any]:
    y1, x1, y2, x2 = map(float, b)
    xmin = clamp((x1 / 1000.0) * w, 0.0, w)
    xmax = clamp((x2 / 1000.0) * w, 0.0, w)
    ymin = clamp((y1 / 1000.0) * h, 0.0, h)
    ymax = clamp((y2 / 1000.0) * h, 0.0, h)
    if as_int:
        return [int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))]
    return [xmin, ymin, xmax, ymax]


def detect_input_bbox_format(b: List[float], w: Optional[float], h: Optional[float]) -> str:
    mx = max(float(x) for x in b)
    if w is not None and h is not None:
        if (w > 1200 or h > 1200) and mx <= 1000.0:
            return "norm1000_yxyx"
        return "pixel_xyxy"
    return "pixel_xyxy"


# ---------------------------
# OCR doc representation
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
    conf: Optional[Any] = None


@dataclass
class OCRDoc:
    words: List[OCRWord]
    tokens: List[OCRToken]
    inv: Dict[str, List[int]]
    word_tokens_range: List[Tuple[int, int]]


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
        li_i = int(li) if isinstance(li, int) or (isinstance(li, str) and li.isdigit()) else None

        words.append(OCRWord(text=text, bbox=bbox_f, line_idx=li_i, conf=r.get("confidence", None)))

        start_pos = pos
        for t in tokenize(text):
            sk = alnum_skeleton(t)
            if sk:
                inv.setdefault(sk, []).append(pos)
            tokens.append(OCRToken(tok=t, word_idx=len(words) - 1, pos=pos))
            pos += 1
        end_pos = pos
        word_tokens_range.append((start_pos, end_pos))

    return OCRDoc(words=words, tokens=tokens, inv=inv, word_tokens_range=word_tokens_range)


# ---------------------------
# Matching
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
    used_label: str


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


def _subseq_match_from(doc: OCRDoc, label_toks: List[str], start_pos: int, end_pos: int, max_scan: int) -> Tuple[float, List[int], List[int]]:
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


def _bbox_union_yxyx(doc: OCRDoc, word_indices: List[int]) -> Optional[List[float]]:
    if not word_indices:
        return None
    ys = [doc.words[i].bbox[0] for i in word_indices]
    xs = [doc.words[i].bbox[1] for i in word_indices]
    ye = [doc.words[i].bbox[2] for i in word_indices]
    xe = [doc.words[i].bbox[3] for i in word_indices]
    return [min(ys), min(xs), max(ye), max(xe)]


def match_label_to_ocr_single(
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
        return MatchResult(False, 0.0, 0.0, 0.0, 0.0, 1.0, [], None, "empty_label_tokens", label)

    anchors = choose_anchor_tokens(label_toks, max_anchors=5)
    anchor_positions = _find_anchor_positions(doc, anchors)

    if not anchor_positions:
        return MatchResult(False, 0.0, 0.0, 0.0, 0.0, 1.0, [], None, "no_anchor_hits", label)

    if len(anchor_positions) > max_candidates:
        step = max(1, len(anchor_positions) // max_candidates)
        anchor_positions = anchor_positions[::step][:max_candidates]

    require_numeric = any(is_numericish(t) for t in label_toks)

    best_score = -1e9
    best_cov = 0.0
    best_cr = 0.0
    best_iou = 0.0
    best_dist = 1.0
    best_widxs: List[int] = []
    best_bbox: Optional[List[float]] = None

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

            cand_bbox = _bbox_union_yxyx(doc, widxs)
            if cand_bbox is None:
                continue
            cand_bbox = [clamp(x, 0.0, 1000.0) for x in cand_bbox]

            if pred_bbox is not None:
                iou = bbox_iou_yxyx(pred_bbox, cand_bbox)
                dist = bbox_center_dist_norm(pred_bbox, cand_bbox)
            else:
                iou, dist = 0.0, 1.0

            text_score = 0.7 * cov + 0.3 * cr
            score = text_score + lambda_iou * iou - lambda_dist * dist

            if score > best_score:
                best_score = score
                best_cov = cov
                best_cr = cr
                best_iou = iou
                best_dist = dist
                best_widxs = widxs
                best_bbox = cand_bbox

    if best_bbox is None:
        return MatchResult(False, 0.0, 0.0, 0.0, 0.0, 1.0, [], None, "no_valid_candidate", label)

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

    dist_relax = 0.06
    cov_relax = 0.15
    cr_relax = 0.10

    min_cov_eff = min_cov
    min_cr_eff = min_cr

    if pred_bbox is not None and best_dist <= dist_relax:
        min_cov_eff = max(0.30, min_cov - cov_relax)
        min_cr_eff = max(0.30, min_cr - cr_relax)

    if best_cov < min_cov_eff and best_cr < min_cr_eff:
        dist_relax2 = 0.12
        min_cov_rel = max(0.25, min_cov_eff - 0.20)
        min_cr_rel = max(0.25, min_cr_eff - 0.15)

        text_score = 0.7 * best_cov + 0.3 * best_cr
        dist_ok = True
        if pred_bbox is not None:
            dist_ok = (best_dist <= dist_relax2) or (best_iou >= 0.02)

        if dist_ok and (
            (best_cov >= min_cov_rel) or
            (best_cr >= min_cr_rel) or
            (text_score >= 0.42 and (best_cov >= 0.35 or best_cr >= 0.35))
        ):
            return MatchResult(True, best_score, best_cov, best_cr, best_iou, best_dist, best_widxs, best_bbox, "ok_relaxed", label)

        return MatchResult(False, best_score, best_cov, best_cr, best_iou, best_dist, best_widxs, best_bbox,
                           f"below_threshold cov={best_cov:.3f} cr={best_cr:.3f} dist={best_dist:.3f}", label)

    return MatchResult(True, best_score, best_cov, best_cr, best_iou, best_dist, best_widxs, best_bbox, "ok", label)


def match_with_candidates(doc: OCRDoc, labels: List[str], pred_bbox: Optional[List[float]], **kwargs) -> MatchResult:
    best_any: Optional[MatchResult] = None
    best_ok: Optional[MatchResult] = None
    for lab in labels:
        mr = match_label_to_ocr_single(doc, lab, pred_bbox, **kwargs)
        if best_any is None or mr.score > best_any.score:
            best_any = mr
        if mr.ok and (best_ok is None or mr.score > best_ok.score):
            best_ok = mr
    return best_ok if best_ok is not None else (best_any if best_any is not None else MatchResult(False, 0, 0, 0, 0, 1, [], None, "no_candidates", ""))


# ---------------------------
# Row processing
# ---------------------------

def process_row_boxes(
    row: Dict[str, Any],
    doc: OCRDoc,
    *,
    bbox_field: str,
    input_bbox_format: str,
    out_bbox_format: str,
    ndigits: int,
    as_int_pixel: bool,
    add_metadata: bool,
    match_kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool, List[Dict[str, Any]]]:
    image = row.get("image", "")
    w = row.get("width", None)
    h = row.get("height", None)
    w_f = float(w) if isinstance(w, (int, float)) else None
    h_f = float(h) if isinstance(h, (int, float)) else None

    answer = row.get("answer", "")
    labels = unique_labels_from_row(row, fallback=str(answer) if answer is not None else "")

    per_box_reports: List[Dict[str, Any]] = []
    row_fail = False

    def to_norm_pred(b: List[float]) -> Optional[List[float]]:
        fmt = input_bbox_format
        if fmt == "auto":
            fmt = detect_input_bbox_format(b, w_f, h_f)
        if fmt == "norm1000_yxyx":
            y1, x1, y2, x2 = map(float, b)
            return [clamp(y1, 0, 1000), clamp(x1, 0, 1000), clamp(y2, 0, 1000), clamp(x2, 0, 1000)]
        if w_f is None or h_f is None or w_f <= 0 or h_f <= 0:
            return None
        return pixel_xyxy_to_norm_yxyx(b, w_f, h_f)

    def from_norm_out(b_norm: List[float], original: Any) -> Any:
        fmt = out_bbox_format
        if fmt == "same":
            fmt = input_bbox_format
        if fmt == "auto":
            fmt = input_bbox_format
        if fmt == "norm1000_yxyx":
            return round_bbox(b_norm, ndigits=ndigits)
        if w_f is None or h_f is None or w_f <= 0 or h_f <= 0:
            return original
        return norm_yxyx_to_pixel_xyxy(b_norm, w_f, h_f, as_int=as_int_pixel)

    if bbox_field == "auto":
        bbox_field = "bboxes" if isinstance(row.get("bboxes", None), list) else "bboxs"

    if bbox_field == "bboxes":
        bxs = row.get("bboxes", [])
        if not isinstance(bxs, list):
            return row, False, per_box_reports

        new_bxs: List[Any] = []
        for i, bb in enumerate(bxs):
            if not isinstance(bb, dict):
                new_bxs.append(bb)
                continue
            pred = bb.get("bbox", None)
            label = bb.get("label", None)
            lab_cands = labels
            if isinstance(label, str) and label.strip():
                lab_cands = [label.strip()] + [x for x in labels if norm_basic(x) != norm_basic(label.strip())]

            if not (isinstance(pred, list) and len(pred) == 4 and all(isinstance(x, (int, float)) for x in pred)):
                new_bxs.append(bb)
                continue

            pred_norm = to_norm_pred([float(x) for x in pred])
            mr = match_with_candidates(doc, lab_cands, pred_norm, **match_kwargs)

            if mr.reason == "no_anchor_hits":
                mr.ok = False

            out_bb = dict(bb)
            if mr.ok and mr.corrected_bbox is not None:
                out_bb["bbox"] = from_norm_out(mr.corrected_bbox, pred)
                if add_metadata:
                    out_bb["ocr_status"] = mr.reason
                    out_bb["ocr_used_label"] = mr.used_label
                    out_bb["ocr_score"] = round(mr.score, 4)
                    out_bb["ocr_cov"] = round(mr.coverage, 4)
                    out_bb["ocr_cr"] = round(mr.char_ratio, 4)
                    out_bb["ocr_iou"] = round(mr.iou, 4)
                    out_bb["ocr_dist"] = round(mr.dist, 4)
                new_bxs.append(out_bb)
            else:
                row_fail = True
                if add_metadata:
                    out_bb["ocr_status"] = "failed"
                    out_bb["ocr_fail_reason"] = mr.reason
                    out_bb["ocr_used_label"] = mr.used_label
                new_bxs.append(out_bb)

            per_box_reports.append({
                "image": image,
                "box_idx": i,
                "input_field": "bboxes",
                "pred_bbox": pred,
                "pred_bbox_norm": round_bbox(pred_norm, ndigits=2) if pred_norm else None,
                "ok": bool(mr.ok),
                "reason": mr.reason,
                "used_label": mr.used_label,
                "score": round(mr.score, 6),
                "coverage": round(mr.coverage, 6),
                "char_ratio": round(mr.char_ratio, 6),
                "iou": round(mr.iou, 6),
                "dist": round(mr.dist, 6),
                "corrected_bbox_norm": round_bbox(mr.corrected_bbox, ndigits=2) if mr.corrected_bbox else None,
            })

        new_row = dict(row)
        new_row["bboxes"] = new_bxs
        return new_row, row_fail, per_box_reports

    bxs = row.get("bboxs", [])
    if not isinstance(bxs, list):
        return row, False, per_box_reports

    new_bboxs: List[Any] = []
    for i, pred in enumerate(bxs):
        if not (isinstance(pred, list) and len(pred) == 4 and all(isinstance(x, (int, float)) for x in pred)):
            new_bboxs.append(pred)
            continue

        pred_norm = to_norm_pred([float(x) for x in pred])
        mr = match_with_candidates(doc, labels, pred_norm, **match_kwargs)

        if mr.reason == "no_anchor_hits":
            mr.ok = False

        if mr.ok and mr.corrected_bbox is not None:
            out_bbox = from_norm_out(mr.corrected_bbox, pred)
            new_bboxs.append(out_bbox)
        else:
            row_fail = True
            new_bboxs.append(pred)

        per_box_reports.append({
            "image": image,
            "box_idx": i,
            "input_field": "bboxs",
            "pred_bbox": pred,
            "pred_bbox_norm": round_bbox(pred_norm, ndigits=2) if pred_norm else None,
            "ok": bool(mr.ok),
            "reason": mr.reason,
            "used_label": mr.used_label,
            "score": round(mr.score, 6),
            "coverage": round(mr.coverage, 6),
            "char_ratio": round(mr.char_ratio, 6),
            "iou": round(mr.iou, 6),
            "dist": round(mr.dist, 6),
            "corrected_bbox_norm": round_bbox(mr.corrected_bbox, ndigits=2) if mr.corrected_bbox else None,
        })

    new_row = dict(row)
    new_row["bboxs"] = new_bboxs
    return new_row, row_fail, per_box_reports


# ---------------------------
# Main (ONLY OK rows written to --output)
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL")
    ap.add_argument("--output", required=True, help="Output JSONL (ONLY rows with zero failed boxes)")
    ap.add_argument("--ocr-word-root", required=True, help="Folder containing OCR word-level JSONL: <stem>.jsonl")
    ap.add_argument("--fail-out", required=True, help="JSONL of rows that failed (any failed box OR missing OCR/image)")
    ap.add_argument("--report-out", default=None, help="Optional per-box report JSONL")

    ap.add_argument("--bbox-field", choices=["auto", "bboxs", "bboxes"], default="auto")
    ap.add_argument("--input-bbox-format", choices=["auto", "pixel_xyxy", "norm1000_yxyx"], default="auto")
    ap.add_argument("--out-bbox-format", choices=["same", "pixel_xyxy", "norm1000_yxyx"], default="same")
    ap.add_argument("--ndigits", type=int, default=2)
    ap.add_argument("--pixel-int", action="store_true", help="If output is pixel, round to int")
    ap.add_argument("--add-metadata", action="store_true", help="Add OCR debug fields into dict-boxes (bboxes)")

    ap.add_argument("--window-words", type=int, default=120)
    ap.add_argument("--window-lines", type=int, default=2)
    ap.add_argument("--lambda-iou", type=float, default=0.15)
    ap.add_argument("--lambda-dist", type=float, default=0.10)
    ap.add_argument("--max-candidates", type=int, default=120)
    ap.add_argument("--max-scan-mult", type=int, default=6)

    ap.add_argument("--no-total", action="store_true", help="Disable total line count pre-pass")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    ocr_root = Path(args.ocr_word_root)
    fail_out = Path(args.fail_out)
    rep_out = Path(args.report_out) if args.report_out else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fail_out.parent.mkdir(parents=True, exist_ok=True)
    if rep_out:
        rep_out.parent.mkdir(parents=True, exist_ok=True)

    total = None
    if not args.no_total:
        try:
            with open(in_path, "rb") as f:
                total = sum(1 for _ in f)
        except Exception:
            total = None

    rows_total = 0
    rows_ok_written = 0
    rows_failed_written = 0
    bad_json = 0
    missing_ocr_files = 0
    missing_image_field = 0

    boxes_ok = 0
    boxes_failed = 0

    fout_rep = open(rep_out, "w", encoding="utf-8") if rep_out else None

    match_kwargs = dict(
        window_words=args.window_words,
        window_lines=args.window_lines,
        lambda_iou=args.lambda_iou,
        lambda_dist=args.lambda_dist,
        max_candidates=args.max_candidates,
        max_scan_mult=args.max_scan_mult,
    )

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout_ok, \
         open(fail_out, "w", encoding="utf-8") as fout_fail:

        for line_no, line in enumerate(tqdm(fin, total=total, desc="DocVQA OCR-correct", dynamic_ncols=True), start=1):
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
            if not isinstance(img, str) or not img:
                missing_image_field += 1
                tmp = dict(row)
                tmp["_ocr_fail_reason"] = "missing_image_field"
                tmp["line_no"] = row.get("line_no", line_no)
                fout_fail.write(json.dumps(tmp, ensure_ascii=False) + "\n")
                rows_failed_written += 1
                continue

            doc_id = Path(img).stem
            ocr_path = ocr_root / f"{doc_id}.jsonl"
            doc = load_ocr_doc(str(ocr_path))
            if doc is None:
                missing_ocr_files += 1
                tmp = dict(row)
                tmp["_ocr_fail_reason"] = "missing_ocr_word_jsonl"
                tmp["line_no"] = row.get("line_no", line_no)
                fout_fail.write(json.dumps(tmp, ensure_ascii=False) + "\n")
                rows_failed_written += 1
                continue

            new_row, row_has_fail, reports = process_row_boxes(
                row,
                doc,
                bbox_field=args.bbox_field,
                input_bbox_format=args.input_bbox_format,
                out_bbox_format=args.out_bbox_format,
                ndigits=args.ndigits,
                as_int_pixel=(True if args.pixel_int else False),
                add_metadata=args.add_metadata,
                match_kwargs=match_kwargs,
            )

            for r in reports:
                if r.get("ok", False):
                    boxes_ok += 1
                else:
                    boxes_failed += 1
                if fout_rep:
                    r2 = dict(r)
                    r2["line_no"] = row.get("line_no", line_no)
                    fout_rep.write(json.dumps(r2, ensure_ascii=False) + "\n")

            new_row["line_no"] = new_row.get("line_no", line_no)

            if row_has_fail:
                fout_fail.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                rows_failed_written += 1
            else:
                fout_ok.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                rows_ok_written += 1

    if fout_rep:
        fout_rep.close()

    print("\nDone.")
    print(f"rows_total={rows_total} ok_written={rows_ok_written} fail_written={rows_failed_written} bad_json={bad_json}")
    print(f"missing_image_field={missing_image_field} missing_ocr_files={missing_ocr_files}")
    print(f"boxes_ok={boxes_ok} boxes_failed={boxes_failed}")
    print(f"ok_output={out_path}")
    print(f"fail_out={fail_out}")
    if rep_out:
        print(f"report_out={rep_out}")


if __name__ == "__main__":
    main()


'''
python docvqa_ocr_correct.py --input docvqa_hard_raw.jsonl --output docvqa_hard_raw_ocr_corrected.jsonl --ocr-word-root ./docvqa_ocr_word_jsonl --bbox-field bboxes --input-bbox-format norm1000_yxyx --out-bbox-format same  --fail-out docvqa_hard_raw_ocr_failed.jsonl --report-out docvqa_hard_raw_ocr_report.jsonl


python docvqa_ocr_correct.py --input docvqa_medium_raw.jsonl --output docvqa_medium_raw_ocr_corrected.jsonl --ocr-word-root ./docvqa_ocr_word_jsonl --bbox-field bboxes --input-bbox-format norm1000_yxyx --out-bbox-format same  --fail-out docvqa_medium_raw_ocr_failed.jsonl --report-out docvqa_medium_raw_ocr_report.jsonl

'''
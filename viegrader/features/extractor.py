"""Trích đặc trưng ngôn ngữ có thể giải thích được.

Bốn nhóm, ánh xạ thẳng vào 4 tiêu chí của rubric:

    surface_*   -> độ dài, cấu trúc bài            (hình thức)
    lexical_*   -> phong phú từ vựng, thuật ngữ    (nội dung, diễn đạt)
    error_*     -> chính tả, dấu câu, văn nói      (diễn đạt)
    discourse_* -> liên kết, lập luận, bố cục      (lập luận)
    semantic_*  -> bám đề, bao phủ ý               (nội dung)  [xem semantic.py]

Ưu điểm của đặc trưng tường minh trong bối cảnh NCKH: mỗi con số đều giải thích
được cho giảng viên và sinh viên, khác với vector ẩn của mô hình sâu.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..cleaning.normalize import (
    INFORMAL_MARKERS,
    diacritic_ratio,
    segment_paragraphs,
    split_sentences,
)
from .vietnamese import (
    AI_STYLE_MARKERS,
    ALL_CONNECTIVES,
    ARGUMENT_MARKERS,
    CITATION_PATTERNS,
    CLICHE_PHRASES,
    CONFUSION_MAP,
    CONNECTIVES,
    STOPWORDS,
    is_valid_syllable,
)

WORD_RE = re.compile(r"[a-zA-ZÀ-ỹđĐ]+", re.UNICODE)
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


# --------------------------------------------------------------------------- #
def surface_features(text: str) -> Dict[str, float]:
    words = WORD_RE.findall(text)
    sents = split_sentences(text)
    paras = segment_paragraphs(text)
    n_w, n_s, n_p = len(words), len(sents), len(paras)
    sent_lens = [len(WORD_RE.findall(s)) for s in sents] or [0]
    word_lens = [len(w) for w in words] or [0]

    return {
        "n_chars": float(len(text)),
        "n_words": float(n_w),
        "n_sentences": float(n_s),
        "n_paragraphs": float(n_p),
        "mean_sentence_len": float(np.mean(sent_lens)),
        "std_sentence_len": float(np.std(sent_lens)),
        "max_sentence_len": float(np.max(sent_lens)),
        "pct_long_sentences": _safe_div(sum(1 for x in sent_lens if x > 40), n_s),
        "pct_short_sentences": _safe_div(sum(1 for x in sent_lens if x < 5), n_s),
        "mean_word_len": float(np.mean(word_lens)),
        "words_per_paragraph": _safe_div(n_w, n_p),
        "sentences_per_paragraph": _safe_div(n_s, n_p),
        "n_numbers": float(len(NUM_RE.findall(text))),
        "diacritic_ratio": diacritic_ratio(text),
    }


# --------------------------------------------------------------------------- #
def _mtld(tokens: Sequence[str], threshold: float = 0.72) -> float:
    """MTLD - đo phong phú từ vựng, ít phụ thuộc độ dài hơn TTR."""
    def _run(seq: Sequence[str]) -> float:
        factors, start, types = 0.0, 0, set()
        for i, t in enumerate(seq):
            types.add(t)
            ttr = len(types) / (i - start + 1)
            if ttr <= threshold:
                factors += 1
                start, types = i + 1, set()
        if start < len(seq):
            ttr = len(types) / (len(seq) - start)
            factors += (1 - ttr) / (1 - threshold) if threshold < 1 else 0
        return _safe_div(len(seq), factors) if factors else float(len(seq))

    if len(tokens) < 20:
        return float(len(set(tokens)))
    return (_run(tokens) + _run(list(reversed(tokens)))) / 2


def lexical_features(text: str, domain_terms: Optional[Sequence[str]] = None) -> Dict[str, float]:
    words = [w.lower() for w in WORD_RE.findall(text)]
    n = len(words)
    types = set(words)
    content = [w for w in words if w not in STOPWORDS]
    freq = Counter(words)

    # Entropy phân bố từ: bài lặp ý nhiều -> entropy thấp
    probs = np.array(list(freq.values()), dtype=float)
    probs = probs / probs.sum() if probs.sum() else probs
    entropy = float(-(probs * np.log2(probs + 1e-12)).sum()) if n else 0.0

    dt = 0
    if domain_terms:
        low = text.lower()
        dt = sum(low.count(t.lower()) for t in domain_terms)

    return {
        "ttr": _safe_div(len(types), n),
        "root_ttr": _safe_div(len(types), math.sqrt(n)) if n else 0.0,
        "mtld": _mtld(words),
        "content_word_ratio": _safe_div(len(content), n),
        "unique_content_ratio": _safe_div(len(set(content)), max(1, len(content))),
        "lexical_entropy": entropy,
        "hapax_ratio": _safe_div(sum(1 for v in freq.values() if v == 1), max(1, len(types))),
        "top1_word_freq": _safe_div(freq.most_common(1)[0][1], n) if freq else 0.0,
        "long_word_ratio": _safe_div(sum(1 for w in words if len(w) >= 7), n),
        "term_density": _safe_div(dt, n),
    }


# --------------------------------------------------------------------------- #
def error_features(text: str) -> Dict[str, float]:
    words = WORD_RE.findall(text)
    n = len(words)
    invalid = [w for w in words if not is_valid_syllable(w)]
    # Bỏ qua từ viết hoa giữa câu (tên riêng, viết tắt) để giảm báo sai
    invalid = [w for w in invalid if not (w[0].isupper() or w.isupper())]

    low = text.lower()
    confusion = sum(low.count(k) for k in CONFUSION_MAP)
    informal = sum(low.count(m) for m in INFORMAL_MARKERS)

    sents = split_sentences(text)
    no_cap = sum(1 for s in sents if s and s[0].islower())
    no_end = sum(1 for s in sents if s and s[-1] not in ".!?…\"')")
    # Lỗi khoảng trắng quanh dấu câu còn sót
    punct_space = len(re.findall(r"\s[,.;:!?]", text)) + len(re.findall(r"[,.;:][^\s\d]", text))

    return {
        "spell_error_rate": _safe_div(len(invalid), n),
        "n_spell_errors": float(len(invalid)),
        "confusion_rate": _safe_div(confusion, max(1, n / 100)),
        "informal_rate": _safe_div(informal, n),
        "no_capital_sentence_rate": _safe_div(no_cap, len(sents)),
        "no_end_punct_rate": _safe_div(no_end, len(sents)),
        "punct_error_rate": _safe_div(punct_space, max(1, len(text) / 100)),
    }


def list_spell_errors(text: str, limit: int = 30) -> List[str]:
    """Danh sách từ nghi sai chính tả - dùng làm bằng chứng trong phản hồi."""
    words = WORD_RE.findall(text)
    out, seen = [], set()
    for w in words:
        if w[0].isupper() or w.isupper():
            continue
        if not is_valid_syllable(w) and w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
        if len(out) >= limit:
            break
    return out


def list_confusions(text: str) -> List[str]:
    low = text.lower()
    return [f"'{k}' → nên là '{v}'" for k, v in CONFUSION_MAP.items() if k in low]


# --------------------------------------------------------------------------- #
def discourse_features(text: str) -> Dict[str, float]:
    low = text.lower()
    words = WORD_RE.findall(text)
    n = max(1, len(words))
    sents = split_sentences(text)
    paras = segment_paragraphs(text)

    conn_counts = {
        f"conn_{k}": float(sum(low.count(c) for c in v)) for k, v in CONNECTIVES.items()
    }
    total_conn = sum(conn_counts.values())
    arg = sum(low.count(m) for m in ARGUMENT_MARKERS)
    cliche = sum(low.count(m) for m in CLICHE_PHRASES)
    ai_style = sum(low.count(m) for m in AI_STYLE_MARKERS)
    has_cite = any(p.search(text) for p in CITATION_PATTERNS)

    # Mạch lạc từ vựng: độ chồng lấn từ nội dung giữa các câu liền kề
    def content_set(s: str) -> set:
        return {w.lower() for w in WORD_RE.findall(s) if w.lower() not in STOPWORDS}

    overlaps = []
    for i in range(len(sents) - 1):
        a, b = content_set(sents[i]), content_set(sents[i + 1])
        if a and b:
            overlaps.append(len(a & b) / len(a | b))
    # Liên kết giữa các đoạn
    p_overlaps = []
    for i in range(len(paras) - 1):
        a, b = content_set(paras[i]), content_set(paras[i + 1])
        if a and b:
            p_overlaps.append(len(a & b) / len(a | b))

    # Bố cục: có mở bài và kết bài không
    first, last = (paras[0].lower() if paras else ""), (paras[-1].lower() if paras else "")
    has_intro = any(c in first for c in CONNECTIVES["trinh_tu"][:4]) or len(paras) >= 3
    has_conclusion = any(c in last for c in CONNECTIVES["ket_luan"])

    out = {
        "connective_density": _safe_div(total_conn, n / 100),
        "connective_variety": float(sum(1 for v in conn_counts.values() if v > 0)),
        "argument_marker_density": _safe_div(arg, n / 100),
        "cliche_density": _safe_div(cliche, n / 100),
        "ai_style_score": _safe_div(ai_style, n / 100),
        "has_citation": 1.0 if has_cite else 0.0,
        "sentence_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        "sentence_overlap_min": float(np.min(overlaps)) if overlaps else 0.0,
        "paragraph_overlap_mean": float(np.mean(p_overlaps)) if p_overlaps else 0.0,
        "has_intro": 1.0 if has_intro else 0.0,
        "has_conclusion": 1.0 if has_conclusion else 0.0,
        "structure_score": (
            (1.0 if has_intro else 0.0) + (1.0 if has_conclusion else 0.0)
            + min(1.0, len(paras) / 4.0)
        ) / 3.0,
    }
    out.update(conn_counts)
    return out


# --------------------------------------------------------------------------- #
def keyword_coverage(text: str, keywords: Sequence[str]) -> Dict[str, float]:
    """Tỉ lệ ý cốt lõi (do giảng viên khai báo trong đáp án) xuất hiện trong bài.

    ``keywords`` có thể là từ khoá đơn hoặc nhóm đồng nghĩa dạng "a|b|c".
    """
    if not keywords:
        return {"keyword_coverage": 0.0, "keyword_hits": 0.0, "keyword_total": 0.0}
    low = text.lower()
    hits = 0
    for kw in keywords:
        variants = [v.strip().lower() for v in str(kw).split("|") if v.strip()]
        if any(v in low for v in variants):
            hits += 1
    return {
        "keyword_coverage": _safe_div(hits, len(keywords)),
        "keyword_hits": float(hits),
        "keyword_total": float(len(keywords)),
    }


def missing_keywords(text: str, keywords: Sequence[str]) -> List[str]:
    low = text.lower()
    out = []
    for kw in keywords or []:
        variants = [v.strip().lower() for v in str(kw).split("|") if v.strip()]
        if variants and not any(v in low for v in variants):
            out.append(variants[0])
    return out


# --------------------------------------------------------------------------- #
def extract_all(
    text: str,
    prompt_text: str = "",
    keywords: Optional[Sequence[str]] = None,
    domain_terms: Optional[Sequence[str]] = None,
    extra: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Gộp toàn bộ đặc trưng tường minh cho một bài làm."""
    f: Dict[str, float] = {}
    f.update(surface_features(text))
    f.update(lexical_features(text, domain_terms))
    f.update(error_features(text))
    f.update(discourse_features(text))
    f.update(keyword_coverage(text, keywords or []))
    if extra:
        f.update({k: float(v) for k, v in extra.items()})
    return f


FEATURE_GROUPS: Dict[str, List[str]] = {
    "noi_dung": ["keyword_coverage", "keyword_hits", "term_density", "n_words",
                 "content_word_ratio", "lexical_entropy", "sim_prompt", "sim_reference"],
    "lap_luan": ["n_paragraphs", "connective_density", "connective_variety",
                 "argument_marker_density", "structure_score", "has_intro",
                 "has_conclusion", "sentence_overlap_mean", "paragraph_overlap_mean"],
    "dien_dat": ["spell_error_rate", "confusion_rate", "informal_rate", "ttr", "mtld",
                 "mean_sentence_len", "std_sentence_len", "pct_long_sentences",
                 "punct_error_rate", "no_capital_sentence_rate", "diacritic_ratio"],
    "hinh_thuc": ["n_words", "has_citation", "dup_ratio", "prompt_copy_ratio",
                  "ai_style_score", "cliche_density", "n_paragraphs"],
}

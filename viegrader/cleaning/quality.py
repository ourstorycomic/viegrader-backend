"""Kiểm định chất lượng bản ghi: quyết định GIỮ / GẮN CỜ / LOẠI.

Nguyên tắc: KHÔNG âm thầm xoá dữ liệu. Mọi bản ghi bị loại đều được ghi ra
``rejected.csv`` kèm lí do, để báo cáo nghiên cứu thống kê được tỉ lệ và
nguyên nhân loại - đây là phần bắt buộc khi mô tả bộ dữ liệu.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .normalize import diacritic_ratio, split_sentences

KEEP, FLAG, REJECT = "keep", "flag", "reject"


@dataclass
class QualityVerdict:
    decision: str                       # keep | flag | reject
    reasons: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class QualityConfig:
    min_words: int = 20                 # dưới ngưỡng này coi như không có bài
    max_words: int = 5000
    min_diacritic_ratio: float = 0.05   # bài viết không dấu
    warn_diacritic_ratio: float = 0.12
    max_non_vn_ratio: float = 0.30      # tỉ lệ ký tự ngoài bảng chữ cái tiếng Việt
    min_unique_word_ratio: float = 0.15 # chống spam lặp một từ
    max_upper_ratio: float = 0.50
    min_sentences: int = 2
    max_repeat_line_ratio: float = 0.50


_WORD_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
_LATIN_VN = re.compile(r"[a-zA-ZÀ-ỹ]")


def assess(text: str, cfg: Optional[QualityConfig] = None) -> QualityVerdict:
    cfg = cfg or QualityConfig()
    reasons: List[str] = []
    t = text.strip()
    words = _WORD_RE.findall(t)
    n_words = len(words)
    sents = split_sentences(t)
    letters = [c for c in t if c.isalpha()]
    dia = diacritic_ratio(t)
    uniq = len(set(w.lower() for w in words)) / n_words if n_words else 0.0
    upper = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
    non_vn = 1.0 - (sum(1 for c in letters if _LATIN_VN.match(c)) / len(letters)) if letters else 1.0
    lines = [ln.strip().lower() for ln in t.split("\n") if ln.strip()]
    rep_line = 1.0 - (len(set(lines)) / len(lines)) if lines else 0.0

    stats = {
        "n_words": float(n_words),
        "n_sentences": float(len(sents)),
        "diacritic_ratio": round(dia, 4),
        "unique_word_ratio": round(uniq, 4),
        "upper_ratio": round(upper, 4),
        "non_vn_char_ratio": round(non_vn, 4),
        "repeat_line_ratio": round(rep_line, 4),
    }

    decision = KEEP
    if n_words < cfg.min_words:
        decision = REJECT
        reasons.append(f"Bài quá ngắn ({n_words} từ < {cfg.min_words})")
    if n_words > cfg.max_words:
        decision = REJECT
        reasons.append(f"Bài quá dài bất thường ({n_words} từ)")
    if dia < cfg.min_diacritic_ratio and n_words >= cfg.min_words:
        decision = REJECT
        reasons.append(f"Bài viết không dấu (diacritic_ratio={dia:.3f})")
    elif dia < cfg.warn_diacritic_ratio:
        decision = FLAG if decision == KEEP else decision
        reasons.append(f"Thiếu dấu nhiều (diacritic_ratio={dia:.3f})")
    if non_vn > cfg.max_non_vn_ratio:
        decision = REJECT
        reasons.append(f"Phần lớn không phải tiếng Việt (non_vn={non_vn:.2f})")
    if uniq < cfg.min_unique_word_ratio and n_words >= cfg.min_words:
        decision = REJECT
        reasons.append(f"Nghi spam lặp từ (unique_word_ratio={uniq:.2f})")
    if rep_line > cfg.max_repeat_line_ratio and len(lines) > 4:
        decision = FLAG if decision == KEEP else decision
        reasons.append(f"Nhiều dòng lặp lại ({rep_line:.2f})")
    if upper > cfg.max_upper_ratio and n_words > 30:
        decision = FLAG if decision == KEEP else decision
        reasons.append("Viết hoa toàn bộ")
    if len(sents) < cfg.min_sentences and n_words >= cfg.min_words:
        decision = FLAG if decision == KEEP else decision
        reasons.append("Bài không tách được thành câu (thiếu dấu chấm câu)")

    return QualityVerdict(decision=decision, reasons=reasons, stats=stats)

"""Chuẩn hoá văn bản tiếng Việt.

Các lớp chuẩn hoá được tách rời để có thể bật/tắt độc lập và để báo cáo được
"đã sửa bao nhiêu chỗ, loại nào" - phục vụ phần mô tả tiền xử lí trong báo cáo
nghiên cứu.

Thứ tự khuyến nghị:
    1. Gỡ mã điều khiển / HTML / ký tự vô hình
    2. Chuẩn hoá Unicode về NFC
    3. Chuẩn hoá kiểu đặt dấu thanh (hoà/hòa)
    4. Chuẩn hoá khoảng trắng và dấu câu
    5. Giãn từ viết tắt / teencode
    6. Gỡ lặp ký tự, emoji
    7. Chuẩn hoá xuống dòng thành đoạn văn
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Bảng chữ cái tiếng Việt
# --------------------------------------------------------------------------- #
VN_LOWER = "aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệghiìíỉĩịklmnoòóỏõọôồốổỗộơờớởỡợpqrstuùúủũụưừứửữựvxyỳýỷỹỵ"
VN_CHARS = set(VN_LOWER) | set(VN_LOWER.upper()) | set("fjwzFJWZ")
VN_DIACRITIC_CHARS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
VN_DIACRITIC_CHARS |= {c.upper() for c in VN_DIACRITIC_CHARS}

# --------------------------------------------------------------------------- #
# 1. Ký tự vô hình / điều khiển
# --------------------------------------------------------------------------- #
INVISIBLE = dict.fromkeys(
    map(ord, "​‌‍⁠﻿­᠎"), None
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

HTML_ENTITIES = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&apos;": "'", "&hellip;": "…",
}
HTML_TAG_RE = re.compile(r"<[^>]{1,80}>")

# --------------------------------------------------------------------------- #
# 3. Chuẩn hoá kiểu đặt dấu thanh (old-style -> new-style)
#    "hòa" (dấu trên o) vs "hoà" (dấu trên a). Chọn 1 kiểu duy nhất để mọi
#    thống kê từ vựng nhất quán. Mặc định: kiểu MỚI (hoà, thuỷ, quý).
# --------------------------------------------------------------------------- #
_TONE_PAIRS_OLD_TO_NEW = {
    # oa
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    # oe
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    # uy
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
}
_TONE_NEW_TO_OLD = {v: k for k, v in _TONE_PAIRS_OLD_TO_NEW.items()}

# --------------------------------------------------------------------------- #
# 5. Từ điển teencode / viết tắt thường gặp trong bài làm của sinh viên
# --------------------------------------------------------------------------- #
TEENCODE: Dict[str, str] = {
    "ko": "không", "k": "không", "kg": "không", "khg": "không", "hok": "không",
    "kh": "không", "hong": "không", "hem": "không", "k0": "không",
    "dc": "được", "đc": "được", "duoc": "được", "j": "gì", "z": "vậy",
    "vs": "với", "v": "vậy", "ntn": "như thế nào", "nhu the nao": "như thế nào",
    "bth": "bình thường", "bt": "bình thường", "nx": "nữa", "nc": "nước",
    "ng": "người", "ngta": "người ta", "mn": "mọi người", "ae": "anh em",
    "vn": "Việt Nam", "hs": "học sinh", "sv": "sinh viên", "gv": "giảng viên",
    "tp": "thành phố", "tphcm": "Thành phố Hồ Chí Minh", "hn": "Hà Nội",
    "cty": "công ty", "ctr": "chương trình", "kt": "kinh tế", "xh": "xã hội",
    "vd": "ví dụ", "vdu": "ví dụ", "tv": "thành viên", "nvl": "nguyên vật liệu",
    "sx": "sản xuất", "kd": "kinh doanh", "qly": "quản lý", "ql": "quản lý",
    "pt": "phát triển", "nn": "nhà nước", "cn": "công nghệ", "cnh": "công nghiệp hoá",
    "hđh": "hiện đại hoá", "tt": "thị trường", "dn": "doanh nghiệp",
    "đt": "đầu tư", "lđ": "lao động", "sp": "sản phẩm", "kh": "khách hàng",
    "tks": "cảm ơn", "ok": "được", "oke": "được", "okay": "được",
}
# Từ ngữ văn nói cần đánh dấu (không thay thế, chỉ đếm để chấm tiêu chí diễn đạt)
INFORMAL_MARKERS = {
    "thì là", "kiểu như", "nói chung là", "cái này", "cái mà", "á", "ạ", "nha",
    "nhé", "nhá", "ừ", "ờ", "vãi", "cực kỳ luôn", "siêu", "chill", "ok",
    "mình nghĩ là", "theo mình", "kiểu", "đấy", "ấy", "thôi",
}

# --------------------------------------------------------------------------- #
# 6. Emoji / emoticon / lặp ký tự
# --------------------------------------------------------------------------- #
EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF"
    "\U0001F000-\U0001F0FF" "\U00002190-\U000021FF" "\U0000FE00-\U0000FE0F" "]",
    flags=re.UNICODE,
)
EMOTICON_RE = re.compile(r"(?<!\w)[:;=8][\-o\*']?[\)\]\(\[dDpP/\\:\}\{@\|]+(?!\w)")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}", flags=re.UNICODE)
REPEAT_PUNCT_RE = re.compile(r"([!?.,;:])\1{1,}")

# --------------------------------------------------------------------------- #
# 4. Khoảng trắng và dấu câu
# --------------------------------------------------------------------------- #
QUOTES = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "′": "'", "″": '"', "`": "'", "´": "'",
}
DASHES = {"–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-"}

SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?%\)\]\}…])")
NO_SPACE_AFTER_PUNCT_RE = re.compile(r"([,.;:!?])(?=[^\s\d\)\]\}…])")
SPACE_AFTER_OPEN_RE = re.compile(r"([\(\[\{])\s+")
MULTI_SPACE_RE = re.compile(r"[ \t ]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

URL_RE = re.compile(r"https?://\S+|www\.\S+")


# =========================================================================== #
@dataclass
class NormalizeReport:
    """Thống kê số lần can thiệp - dùng để mô tả tiền xử lí trong báo cáo."""

    counts: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def bump(self, key: str, n: int = 1) -> None:
        if n:
            self.counts[key] = self.counts.get(key, 0) + n

    def as_dict(self) -> Dict[str, int]:
        return dict(sorted(self.counts.items()))


# =========================================================================== #
def strip_invisible(text: str, rep: NormalizeReport | None = None) -> str:
    """Gỡ ký tự điều khiển, ký tự zero-width, thẻ HTML."""
    before = len(text)
    for ent, ch in HTML_ENTITIES.items():
        text = text.replace(ent, ch)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.translate(INVISIBLE)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if rep:
        rep.bump("invisible_removed", max(0, before - len(text)))
    return text


def to_nfc(text: str, rep: NormalizeReport | None = None) -> str:
    """Chuẩn hoá Unicode về dạng tổ hợp sẵn (NFC).

    Rất quan trọng với tiếng Việt: cùng chữ 'ế' có thể được lưu bằng 1 code point
    (NFC) hoặc 2-3 code point (NFD). Không chuẩn hoá thì thống kê ký tự, so khớp
    từ điển và tokenizer đều sai.
    """
    out = unicodedata.normalize("NFC", text)
    if rep and out != text:
        rep.bump("unicode_nfc", 1)
    return out


def normalize_tone_marks(text: str, style: str = "new",
                         rep: NormalizeReport | None = None) -> str:
    """Thống nhất kiểu đặt dấu thanh cho các vần oa/oe/uy.

    style='new'  -> hoà, thuỷ, quý  (kiểu hiện hành trong SGK)
    style='old'  -> hòa, thủy, quý
    """
    table = _TONE_PAIRS_OLD_TO_NEW if style == "new" else _TONE_NEW_TO_OLD
    n = 0
    for src, dst in table.items():
        if src in text:
            n += text.count(src)
            text = text.replace(src, dst)
        su, du = src.upper(), dst.upper()
        if su in text:
            n += text.count(su)
            text = text.replace(su, du)
    if rep:
        rep.bump("tone_style_fixed", n)
    return text


def normalize_punct(text: str, rep: NormalizeReport | None = None) -> str:
    """Chuẩn hoá dấu nháy, gạch ngang, dấu câu lặp và khoảng trắng quanh dấu."""
    n = 0
    for src, dst in {**QUOTES, **DASHES}.items():
        if src in text:
            n += text.count(src)
            text = text.replace(src, dst)
    text, k = REPEAT_PUNCT_RE.subn(r"\1", text)
    n += k
    text, k = SPACE_BEFORE_PUNCT_RE.subn(r"\1", text)
    n += k
    text, k = NO_SPACE_AFTER_PUNCT_RE.subn(r"\1 ", text)
    n += k
    text, k = SPACE_AFTER_OPEN_RE.subn(r"\1", text)
    n += k
    if rep:
        rep.bump("punct_fixed", n)
    return text


def normalize_whitespace(text: str, rep: NormalizeReport | None = None) -> str:
    """Gộp khoảng trắng, giữ ranh giới đoạn văn (dòng trống)."""
    text = MULTI_SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def expand_teencode(text: str, extra: Dict[str, str] | None = None,
                    rep: NormalizeReport | None = None) -> str:
    """Giãn teencode / viết tắt về dạng chuẩn.

    CẢNH BÁO PHƯƠNG PHÁP: bước này làm mất tín hiệu 'dùng văn nói' vốn là căn cứ
    chấm tiêu chí diễn đạt. Vì vậy hàm ``clean_text`` mặc định ĐẾM teencode trước
    rồi mới giãn, và số đếm được giữ lại làm đặc trưng ``teencode_rate``.
    """
    table = dict(TEENCODE)
    if extra:
        table.update(extra)
    n = 0

    def _sub(m: re.Match) -> str:
        nonlocal n
        w = m.group(0)
        low = w.lower()
        if low in table:
            n += 1
            out = table[low]
            return out.capitalize() if w[0].isupper() else out
        return w

    text = re.sub(r"\b[\wÀ-ỹ]+\b", _sub, text, flags=re.UNICODE)
    if rep:
        rep.bump("teencode_expanded", n)
    return text


def count_teencode(text: str) -> int:
    toks = re.findall(r"\b[\wÀ-ỹ]+\b", text.lower(), flags=re.UNICODE)
    return sum(1 for t in toks if t in TEENCODE)


def strip_emoji(text: str, rep: NormalizeReport | None = None) -> str:
    text, a = EMOJI_RE.subn("", text)
    text, b = EMOTICON_RE.subn("", text)
    if rep:
        rep.bump("emoji_removed", a + b)
    return text


def collapse_repeats(text: str, rep: NormalizeReport | None = None) -> str:
    """'hayyyy' -> 'hay'. Giữ lại 1 ký tự vì tiếng Việt không có phụ âm đôi lặp."""
    text, n = REPEAT_CHAR_RE.subn(r"\1", text)
    if rep:
        rep.bump("repeat_chars_collapsed", n)
    return text


def mask_urls(text: str, rep: NormalizeReport | None = None) -> str:
    text, n = URL_RE.subn("<URL>", text)
    if rep:
        rep.bump("urls_masked", n)
    return text


def diacritic_ratio(text: str) -> float:
    """Tỉ lệ ký tự mang dấu trên tổng ký tự chữ cái.

    Bài tiếng Việt viết đúng chính tả thường có tỉ lệ 0.18–0.35.
    Dưới ~0.05 nghĩa là bài viết không dấu -> cần gắn cờ (mô hình PhoBERT sẽ
    hoạt động rất kém trên văn bản không dấu).
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c in VN_DIACRITIC_CHARS) / len(letters)


def segment_paragraphs(text: str, min_len: int = 15) -> List[str]:
    """Tách đoạn văn: ưu tiên dòng trống, nếu bài viết liền mạch thì tách theo
    dòng đơn, cuối cùng mới coi toàn bài là 1 đoạn."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in text.split("\n") if len(p.strip()) >= min_len]
    return paras or ([text.strip()] if text.strip() else [])


SENT_END_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-Ỹ0-9\"'(])")


def split_sentences(text: str) -> List[str]:
    """Tách câu theo dấu kết thúc. Đủ dùng cho đặc trưng thống kê; nếu cài
    ``underthesea`` thì ``features.surface`` sẽ dùng bộ tách câu tốt hơn."""
    text = re.sub(r"\b([A-ZÀ-Ỹ])\.\s*", r"\1.", text)  # tránh cắt ở viết tắt
    parts = [s.strip() for s in SENT_END_RE.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


# =========================================================================== #
def clean_text(
    text: str,
    *,
    tone_style: str = "new",
    do_teencode: bool = True,
    do_emoji: bool = True,
    do_repeat: bool = True,
    do_url: bool = True,
    teencode_extra: Dict[str, str] | None = None,
) -> Tuple[str, NormalizeReport]:
    """Chạy toàn bộ chuỗi chuẩn hoá, trả về (văn bản sạch, báo cáo can thiệp)."""
    rep = NormalizeReport()
    rep.bump("teencode_found", count_teencode(text))
    rep.notes.append(f"diacritic_ratio_raw={diacritic_ratio(text):.3f}")

    t = strip_invisible(text, rep)
    t = to_nfc(t, rep)
    if do_url:
        t = mask_urls(t, rep)
    if do_emoji:
        t = strip_emoji(t, rep)
    if do_repeat:
        t = collapse_repeats(t, rep)
    t = normalize_tone_marks(t, tone_style, rep)
    if do_teencode:
        t = expand_teencode(t, teencode_extra, rep)
    t = normalize_punct(t, rep)
    t = normalize_whitespace(t, rep)
    t = to_nfc(t)
    return t, rep

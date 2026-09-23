"""Tài nguyên ngôn ngữ tiếng Việt dùng cho trích đặc trưng.

Điểm cốt lõi: **bộ kiểm tra âm tiết hợp lệ**. Tiếng Việt là ngôn ngữ đơn lập,
mỗi âm tiết có cấu trúc chặt: (phụ âm đầu) + vần + thanh điệu. Tập âm tiết hợp lệ
là hữu hạn (~6.200). Vì vậy có thể phát hiện lỗi chính tả **không cần từ điển từ**
bằng cách sinh toàn bộ tổ hợp phụ âm đầu × vần × thanh rồi đối chiếu.

Cách này tránh được nhược điểm của từ điển: không phụ thuộc dữ liệu ngoài, chạy
offline, và không báo sai với thuật ngữ chuyên ngành ghép từ các âm tiết hợp lệ.
Hạn chế: không bắt được lỗi "đúng âm tiết nhưng sai từ" (vd. *"chia sẽ"* thay vì
*"chia sẻ"*) - phần này xử lí bằng danh sách cặp nhầm lẫn phổ biến bên dưới.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Set, Tuple

# --------------------------------------------------------------------------- #
# 1. Phụ âm đầu
# --------------------------------------------------------------------------- #
ONSETS: List[str] = [
    "", "b", "c", "ch", "d", "đ", "g", "gh", "gi", "h", "k", "kh", "l", "m",
    "n", "ng", "ngh", "nh", "p", "ph", "qu", "r", "s", "t", "th", "tr", "v", "x",
]

# --------------------------------------------------------------------------- #
# 2. Vần (không mang thanh)
# --------------------------------------------------------------------------- #
RIMES: List[str] = [
    # nguyên âm đơn
    "a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y",
    # nguyên âm đôi
    "ia", "iê", "yê", "ua", "uô", "ưa", "ươ",
    # âm đệm + nguyên âm
    "oa", "oe", "oo", "uy", "uê", "uơ", "uâ", "uă", "ue", "ui", "uo",
    # kết thúc bằng bán nguyên âm
    "ai", "ao", "au", "ay", "âu", "ây", "eo", "êu", "iu", "oi", "ôi", "ơi",
    "ua", "ui", "ưi", "ưu", "iêu", "yêu", "uôi", "ươi", "ươu", "oai", "oay",
    "uay", "uôi", "uai", "uôm", "iu", "ao", "eo", "êu", "ia",
    # kết thúc bằng phụ âm cuối
    "am", "ăm", "âm", "em", "êm", "im", "om", "ôm", "ơm", "um", "ươm", "iêm", "yêm",
    "an", "ăn", "ân", "en", "ên", "in", "on", "ôn", "ơn", "un", "ưn", "ươn", "iên", "yên", "uôn",
    "ang", "ăng", "âng", "eng", "êng", "ing", "ong", "ông", "ung", "ưng", "ương", "iêng", "uông",
    "anh", "ênh", "inh", "oanh", "uynh", "uênh",
    "ach", "êch", "ich", "oach", "uych",
    "ap", "ăp", "âp", "ep", "êp", "ip", "op", "ôp", "ơp", "up", "ươp", "iêp", "yêp",
    "at", "ăt", "ât", "et", "êt", "it", "ot", "ôt", "ơt", "ut", "ưt", "ươt", "iêt", "yêt", "uôt",
    "ac", "ăc", "âc", "ec", "oc", "ôc", "uc", "ưc", "ươc", "iêc", "uôc",
    "oam", "oan", "oang", "oat", "oac", "oăn", "oăng", "oăt", "oăc", "oen", "oet",
    "uân", "uâng", "uât", "uôm", "uyên", "uyêt", "uyn", "uyt", "uym",
    "iêm", "ươm", "ươp",
    "oam", "uâc",
]
RIMES = sorted(set(RIMES), key=len, reverse=True)

# --------------------------------------------------------------------------- #
# 3. Thanh điệu - áp lên nguyên âm chính
# --------------------------------------------------------------------------- #
TONE_MAP: Dict[str, str] = {
    "a": "àáảãạ", "ă": "ằắẳẵặ", "â": "ầấẩẫậ",
    "e": "èéẻẽẹ", "ê": "ềếểễệ",
    "i": "ìíỉĩị",
    "o": "òóỏõọ", "ô": "ồốổỗộ", "ơ": "ờớởỡợ",
    "u": "ùúủũụ", "ư": "ừứửữự",
    "y": "ỳýỷỹỵ",
}
_TONED_TO_BASE: Dict[str, str] = {}
for base, toned in TONE_MAP.items():
    for ch in toned:
        _TONED_TO_BASE[ch] = base


def strip_tone(s: str) -> str:
    """Bỏ dấu thanh, GIỮ dấu phụ (ă, â, ê, ô, ơ, ư, đ)."""
    return "".join(_TONED_TO_BASE.get(c, c) for c in s)


def strip_all_diacritics(s: str) -> str:
    """Bỏ toàn bộ dấu, kể cả dấu phụ: 'nghiệp' -> 'nghiep'."""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


@lru_cache(maxsize=1)
def valid_syllables() -> Set[str]:
    """Sinh tập âm tiết hợp lệ (không thanh) từ phụ âm đầu × vần, có lọc ràng buộc
    chính tả tiếng Việt (k/c/q, g/gh, ng/ngh...)."""
    out: Set[str] = set()
    front_vowels = ("i", "e", "ê", "y")
    for on in ONSETS:
        for ri in RIMES:
            if not ri:
                continue
            first = ri[0]
            # Quy tắc chính tả
            if on in ("k",) and first not in front_vowels:
                continue
            if on in ("c",) and first in front_vowels:
                continue
            if on == "gh" and first not in ("i", "e", "ê"):
                continue
            if on == "g" and first in ("i", "e", "ê") and ri not in ("i",):
                continue
            if on == "ngh" and first not in ("i", "e", "ê"):
                continue
            if on == "ng" and first in ("i", "e", "ê"):
                continue
            if on == "qu" and first not in ("a", "ă", "â", "e", "ê", "i", "y", "o", "ô", "ơ", "u"):
                continue
            if on == "p" and ri not in ("in", "ip", "a", "e", "o", "in"):
                continue
            out.add(on + ri)
    # Âm tiết mượn / tên riêng thường gặp
    out |= {"pa", "pi", "pe", "po", "pu", "ga", "go", "gu", "ka", "ki", "ko",
            "we", "fa", "vi", "ve", "va", "za"}
    return out


@lru_cache(maxsize=200_000)
def is_valid_syllable(word: str) -> bool:
    w = word.lower().strip()
    if not w or not re.fullmatch(r"[a-zà-ỹđ]+", w):
        return False
    return strip_tone(w) in valid_syllables()


# --------------------------------------------------------------------------- #
# 4. Cặp nhầm lẫn chính tả phổ biến (đúng âm tiết nhưng sai từ)
#    Dùng để cảnh báo, KHÔNG tự sửa (tránh làm hỏng nhãn diễn đạt).
# --------------------------------------------------------------------------- #
CONFUSION_PAIRS: List[Tuple[str, str]] = [
    ("chia sẽ", "chia sẻ"), ("sữa chữa", "sửa chữa"), ("bổ xung", "bổ sung"),
    ("chuẩn đoán", "chẩn đoán"), ("cọ sát", "cọ xát"), ("dành giật", "giành giật"),
    ("giành thời gian", "dành thời gian"), ("sáng lạng", "xán lạn"),
    ("tựu chung", "tựu trung"), ("trau chuốt", "chau chuốt"),
    ("xúc tích", "súc tích"), ("thăm quan", "tham quan"),
    ("vô hình chung", "vô hình trung"), ("nhậm chức", "nhận chức"),
    ("khắc phục hậu quả", "khắc phục hậu quả"), ("đường xá", "đường sá"),
    ("xuất xắc", "xuất sắc"), ("sáng lạn", "xán lạn"), ("nhất chí", "nhất trí"),
    ("chân thành", "chân thành"), ("giả thuyết", "giả thiết"),
    ("chuyện môn", "chuyên môn"), ("phản ánh", "phản ánh"),
    ("bàng quang", "bàng quan"), ("cải thiện", "cải thiện"),
    ("kết cục", "kết cuộc"), ("năng nỗ", "năng nổ"), ("nổ lực", "nỗ lực"),
    ("cảm ơn", "cảm ơn"), ("sơ xuất", "sơ suất"), ("xán lạng", "xán lạn"),
    ("trong xuốt", "trong suốt"), ("suất sắc", "xuất sắc"),
    ("dữ liệu", "dữ liệu"), ("sử lý", "xử lý"), ("sử dụng", "sử dụng"),
    ("tham gia", "tham gia"), ("chú trọng", "chú trọng"),
]
CONFUSION_MAP = {a: b for a, b in CONFUSION_PAIRS if a != b}

# --------------------------------------------------------------------------- #
# 5. Từ chức năng / hư từ (stopwords) - dùng cho TTR nội dung
# --------------------------------------------------------------------------- #
STOPWORDS: Set[str] = set("""
và là của có được cho những các một trong với đã sẽ đang không này đó khi thì
mà nên như để về từ theo tại bởi vì nếu nhưng hay hoặc cũng rất nhiều ít vẫn
đều còn chỉ nữa lại ra vào lên xuống ở trên dưới sau trước giữa cùng bằng
người ta tôi bạn chúng anh chị em họ nó ai gì nào đâu sao bao nhiêu
việc điều cái con chiếc sự cách thứ phần mỗi mọi toàn tất cả
""".split())

# --------------------------------------------------------------------------- #
# 6. Từ nối / phương tiện liên kết - đặc trưng cho tiêu chí lập luận & mạch lạc
# --------------------------------------------------------------------------- #
CONNECTIVES: Dict[str, List[str]] = {
    "bo_sung": ["ngoài ra", "bên cạnh đó", "hơn nữa", "thêm vào đó", "đồng thời",
                "không những thế", "mặt khác", "cũng như"],
    "nguyen_nhan": ["vì", "bởi vì", "do", "do đó", "vì vậy", "cho nên", "nên",
                    "dẫn đến", "kéo theo", "nguyên nhân là", "sở dĩ"],
    "ket_qua": ["kết quả là", "vì thế", "từ đó", "nhờ vậy", "hệ quả là", "suy ra"],
    "tuong_phan": ["tuy nhiên", "nhưng", "trái lại", "ngược lại", "mặc dù",
                   "dù vậy", "song", "thế nhưng", "trong khi đó"],
    "vi_du": ["ví dụ", "chẳng hạn", "cụ thể là", "điển hình là", "minh chứng",
              "đơn cử", "thực tế cho thấy"],
    "trinh_tu": ["trước hết", "đầu tiên", "thứ nhất", "thứ hai", "thứ ba",
                 "tiếp theo", "sau đó", "cuối cùng", "một là", "hai là"],
    "ket_luan": ["tóm lại", "nhìn chung", "nói tóm lại", "kết luận", "như vậy",
                 "có thể thấy", "tổng kết lại", "vì những lẽ trên"],
    "nhan_manh": ["đặc biệt", "quan trọng hơn", "cần nhấn mạnh", "đáng chú ý",
                  "rõ ràng", "chắc chắn"],
}
ALL_CONNECTIVES = sorted({c for v in CONNECTIVES.values() for c in v}, key=len, reverse=True)

# Dấu hiệu lập luận: nêu luận điểm, dẫn chứng, phản biện
ARGUMENT_MARKERS: List[str] = [
    "theo tôi", "quan điểm", "luận điểm", "có thể khẳng định", "cần nhận thấy",
    "điều này cho thấy", "chứng tỏ", "phân tích", "lí giải", "lý giải",
    "một mặt", "mặt khác", "phản biện", "ngược lại có ý kiến", "giả sử",
    "số liệu", "thống kê", "nghiên cứu chỉ ra", "theo tác giả", "dẫn chứng",
]

# Dấu hiệu trích dẫn nguồn
CITATION_PATTERNS = [
    re.compile(r"\[\d+\]"),
    re.compile(r"\([A-ZÀ-Ỹ][\w\.]+,?\s*\d{4}\)"),
    re.compile(r"(?i)\b(theo|dẫn theo|trích|nguồn)\s*[:\-]?\s*[A-ZÀ-Ỹ]"),
    re.compile(r"(?i)\b(tài liệu tham khảo|danh mục tài liệu)\b"),
]

# Cụm sáo rỗng / văn mẫu - đặc trưng phát hiện bài học thuộc lòng
CLICHE_PHRASES: List[str] = [
    "từ xưa đến nay", "trong cuộc sống hiện đại ngày nay", "có thể nói rằng",
    "như chúng ta đã biết", "không thể phủ nhận rằng", "đóng vai trò vô cùng quan trọng",
    "là một vấn đề nhức nhối", "gây ra nhiều hậu quả nghiêm trọng",
    "mỗi chúng ta cần phải", "hãy chung tay", "thiết nghĩ",
]

# Cụm thường gặp ở văn bản do máy sinh (dùng làm tín hiệu tham khảo, KHÔNG kết luận)
AI_STYLE_MARKERS: List[str] = [
    "điều quan trọng cần lưu ý là", "trong bối cảnh đó", "đóng vai trò then chốt",
    "một cách toàn diện", "đa chiều", "tối ưu hoá", "tận dụng tối đa",
    "nói tóm lại, có thể thấy rằng", "dưới đây là", "thứ nhất, thứ hai, thứ ba",
    "không chỉ ... mà còn", "trong thời đại số",
]

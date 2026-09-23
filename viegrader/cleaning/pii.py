"""Ẩn danh hoá dữ liệu bài làm.

Bắt buộc trong nghiên cứu giáo dục: bộ dữ liệu bài làm chứa dữ liệu cá nhân của
người học. Quy trình ở đây thực hiện *giả danh hoá* (pseudonymisation):
    - Thay định danh trực tiếp bằng thẻ giữ chỗ (<TEN>, <MSSV>, <EMAIL>...)
    - Sinh ``student_hash`` = HMAC-SHA256(mã SV, khoá bí mật) để vẫn ghép được
      dữ liệu giữa các lần chấm mà không lộ danh tính.
Khoá bí mật KHÔNG được lưu cùng bộ dữ liệu công bố.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"\b(?:\+?84|0)(?:\d[\s.-]?){8,10}\b")
# Mã sinh viên phổ biến ở VN: 8-12 ký tự chữ+số, ví dụ 21A4010123, B21DCCN001
MSSV_RE = re.compile(r"\b(?=[A-Z0-9]{7,12}\b)(?=.*\d)[A-Z][A-Z0-9]{6,11}\b")
MSSV_NUM_RE = re.compile(r"\b\d{8,12}\b")
CCCD_RE = re.compile(r"\b\d{12}\b")
DOB_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}\b")
URL_RE = re.compile(r"https?://\S+")

# Nhãn thường đứng trước thông tin cá nhân trong bài làm
LABELLED_NAME_RE = re.compile(
    r"(?i)\b(họ\s*(?:và\s*)?tên|sinh\s*viên|hoc\s*vien|học\s*viên|thí\s*sinh|người\s*viết)\s*"
    r"[:\-]\s*([^\n,;]{2,60})"
)
LABELLED_ID_RE = re.compile(
    r"(?i)\b(mssv|mã\s*sv|mã\s*sinh\s*viên|msv|sbd|số\s*báo\s*danh|mã\s*số)\s*[:\-]\s*([\w\d.]{3,20})"
)
LABELLED_CLASS_RE = re.compile(r"(?i)\b(lớp|khoá|khóa|nhóm)\s*[:\-]\s*([^\n,;]{1,30})")


@dataclass
class PIIReport:
    counts: Dict[str, int]
    student_id_found: Optional[str] = None
    name_found: Optional[str] = None


def hash_id(value: str, secret: str) -> str:
    """HMAC-SHA256 rút gọn 16 hex - đủ tránh va chạm ở quy mô vài chục nghìn bài."""
    return hmac.new(secret.encode("utf-8"), value.strip().lower().encode("utf-8"),
                    hashlib.sha256).hexdigest()[:16]


def anonymize(text: str, secret: str = "viegrader-default-key") -> Tuple[str, PIIReport]:
    """Thay thế thông tin định danh trong bài làm bằng thẻ giữ chỗ."""
    counts: Dict[str, int] = {}
    sid: Optional[str] = None
    name: Optional[str] = None

    def bump(k: str, n: int) -> None:
        if n:
            counts[k] = counts.get(k, 0) + n

    m = LABELLED_ID_RE.search(text)
    if m:
        sid = m.group(2)
    m = LABELLED_NAME_RE.search(text)
    if m:
        name = m.group(2).strip()

    text, n = LABELLED_NAME_RE.subn(lambda m: f"{m.group(1)}: <TEN>", text)
    bump("ten", n)
    text, n = LABELLED_ID_RE.subn(lambda m: f"{m.group(1)}: <MSSV>", text)
    bump("mssv_label", n)
    text, n = LABELLED_CLASS_RE.subn(lambda m: f"{m.group(1)}: <LOP>", text)
    bump("lop", n)
    text, n = EMAIL_RE.subn("<EMAIL>", text)
    bump("email", n)
    text, n = PHONE_RE.subn("<SDT>", text)
    bump("sdt", n)
    text, n = CCCD_RE.subn("<CCCD>", text)
    bump("cccd", n)
    text, n = DOB_RE.subn("<NGAYSINH>", text)
    bump("ngay_sinh", n)
    text, n = MSSV_RE.subn("<MSSV>", text)
    bump("mssv", n)
    text, n = MSSV_NUM_RE.subn("<MSSV>", text)
    bump("mssv_so", n)

    return text, PIIReport(counts=counts, student_id_found=sid, name_found=name)


def strip_header(text: str, max_lines: int = 6) -> str:
    """Cắt bỏ phần đầu bài chứa họ tên/lớp/MSSV.

    Chỉ cắt những dòng ngắn (<80 ký tự) có chứa nhãn định danh, dừng ngay khi
    gặp dòng văn xuôi thật để không mất nội dung.
    """
    lines = text.split("\n")
    keep_from = 0
    label = re.compile(
        r"(?i)(họ\s*(và\s*)?tên|mssv|mã\s*sv|lớp|khoá|khóa|sbd|số\s*báo\s*danh|"
        r"môn|học\s*phần|giảng\s*viên|ngày\s*(thi|làm)|<TEN>|<MSSV>|<LOP>)"
    )
    for i, ln in enumerate(lines[:max_lines]):
        s = ln.strip()
        if not s:
            keep_from = i + 1
            continue
        if len(s) < 80 and label.search(s):
            keep_from = i + 1
        else:
            break
    return "\n".join(lines[keep_from:]).strip()

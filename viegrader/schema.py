"""Các cấu trúc dữ liệu lõi của viegrader.

Toàn bộ pipeline (làm sạch -> gán nhãn -> trích đặc trưng -> chấm) đều trao đổi
qua các dataclass trong file này để tránh phụ thuộc vào định dạng file cụ thể.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# 1. Đơn vị dữ liệu bài làm
# --------------------------------------------------------------------------- #
@dataclass
class Essay:
    """Một bài làm tự luận của sinh viên."""

    essay_id: str
    text: str                                   # nội dung bài làm (đã hoặc chưa làm sạch)
    prompt_id: Optional[str] = None             # mã câu hỏi / đề bài
    prompt_text: Optional[str] = None           # nội dung đề bài
    course_id: Optional[str] = None             # mã học phần
    student_hash: Optional[str] = None          # định danh SV đã băm (không lưu tên thật)
    raw_text: Optional[str] = None              # bản gốc trước khi làm sạch
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.raw_text is None:
            self.raw_text = self.text

    @property
    def fingerprint(self) -> str:
        """Vân tay nội dung, dùng để phát hiện trùng lặp tuyệt đối."""
        norm = " ".join(self.text.lower().split())
        return hashlib.sha1(norm.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 2. Rubric
# --------------------------------------------------------------------------- #
@dataclass
class Level:
    """Một mức chất lượng trong một tiêu chí rubric."""

    name: str                 # ví dụ: "Tốt", "Khá", "Đạt", "Chưa đạt"
    score: float              # điểm quy đổi của mức này (theo thang của tiêu chí)
    descriptor: str           # mô tả biểu hiện quan sát được
    indicators: List[str] = field(default_factory=list)  # dấu hiệu định lượng hỗ trợ gán nhãn


@dataclass
class Criterion:
    """Một tiêu chí chấm."""

    key: str                  # định danh máy, ví dụ "noi_dung"
    name: str                 # tên hiển thị, ví dụ "Nội dung kiến thức"
    weight: float             # trọng số (tổng các trọng số = 1.0)
    max_score: float          # điểm tối đa của tiêu chí
    description: str = ""
    levels: List[Level] = field(default_factory=list)
    # Đặc trưng gợi ý dùng cho tiêu chí này (để mô hình per-trait tập trung đúng tín hiệu)
    feature_hints: List[str] = field(default_factory=list)

    def level_scores(self) -> List[float]:
        return sorted({lv.score for lv in self.levels})


@dataclass
class Rubric:
    """Bộ tiêu chí chấm đầy đủ cho một loại bài kiểm tra."""

    rubric_id: str
    name: str
    scale_max: float = 10.0        # thang điểm tổng
    scale_step: float = 0.25       # bước làm tròn điểm tổng
    criteria: List[Criterion] = field(default_factory=list)
    rules: List[Dict[str, Any]] = field(default_factory=list)   # quy định chấm (rule engine)
    meta: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Criterion:
        for c in self.criteria:
            if c.key == key:
                return c
        raise KeyError(f"Không tìm thấy tiêu chí '{key}' trong rubric '{self.rubric_id}'")

    @property
    def keys(self) -> List[str]:
        return [c.key for c in self.criteria]

    def validate(self) -> List[str]:
        """Kiểm tra tính nhất quán của rubric. Trả về danh sách cảnh báo."""
        warns: List[str] = []
        if not self.criteria:
            warns.append("Rubric không có tiêu chí nào.")
        total_w = sum(c.weight for c in self.criteria)
        if abs(total_w - 1.0) > 1e-6:
            warns.append(f"Tổng trọng số = {total_w:.4f}, khác 1.0.")
        seen = set()
        for c in self.criteria:
            if c.key in seen:
                warns.append(f"Trùng key tiêu chí: {c.key}")
            seen.add(c.key)
            if not c.levels:
                warns.append(f"Tiêu chí '{c.key}' chưa khai báo mức chất lượng.")
                continue
            top = max(lv.score for lv in c.levels)
            if abs(top - c.max_score) > 1e-6:
                warns.append(
                    f"Tiêu chí '{c.key}': mức cao nhất {top} khác max_score {c.max_score}."
                )
        return warns


# --------------------------------------------------------------------------- #
# 3. Kết quả chấm
# --------------------------------------------------------------------------- #
@dataclass
class CriterionScore:
    key: str
    name: str
    raw_score: float                 # điểm mô hình dự đoán (liên tục)
    score: float                     # điểm sau khi neo về mức rubric
    level: Optional[str] = None      # tên mức đạt được
    weight: float = 0.0
    confidence: float = 0.0          # 0..1
    evidence: List[str] = field(default_factory=list)   # bằng chứng trích từ bài làm
    comment: str = ""


@dataclass
class ScoreResult:
    essay_id: str
    rubric_id: str
    total: float                                    # điểm tổng cuối cùng
    total_before_rules: float = 0.0                 # điểm trước khi áp quy định chấm
    criteria: List[CriterionScore] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)  # cờ cảnh báo: đạo văn, lạc đề, quá ngắn...
    needs_human_review: bool = False
    review_reason: str = ""
    confidence: float = 0.0
    feedback: str = ""
    applied_rules: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, **kw)


# --------------------------------------------------------------------------- #
# 4. Nhãn của giám khảo (dùng cho huấn luyện & đo đồng thuận)
# --------------------------------------------------------------------------- #
@dataclass
class Annotation:
    essay_id: str
    rater_id: str
    scores: Dict[str, float]              # {criterion_key: score}
    total: Optional[float] = None
    round: int = 1                        # vòng gán nhãn (1, 2, phân xử = 3)
    duration_sec: Optional[float] = None
    note: str = ""

    def compute_total(self, rubric: Rubric) -> float:
        if self.total is not None:
            return self.total
        t = 0.0
        for c in rubric.criteria:
            s = self.scores.get(c.key)
            if s is None:
                continue
            t += (s / c.max_score) * c.weight * rubric.scale_max
        self.total = round(t, 4)
        return self.total

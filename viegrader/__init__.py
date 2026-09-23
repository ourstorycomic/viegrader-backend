"""viegrader — Hệ thống chấm điểm tự động bài tự luận tiếng Việt theo rubric.

Kiến trúc 5 tầng:
    1. io_utils   — nạp bài làm từ thư mục/CSV/Excel/Word/PDF
    2. cleaning   — chuẩn hoá, ẩn danh, kiểm định chất lượng, chống trùng lặp
    3. labeling   — rubric, lấy mẫu, sổ tay gán nhãn, đo đồng thuận, phân xử
    4. features + models — đặc trưng tường minh + PhoBERT + LLM, tổng hợp có trọng số
    5. scoring + evaluation — thi hành quy định chấm, sinh phản hồi, đo hiệu năng
"""

__version__ = "0.5.1"

from .config import load_rubric, default_rubric_path, thresholds
from .schema import (
    Essay,
    Rubric,
    Criterion,
    Level,
    ScoreResult,
    CriterionScore,
    Annotation,
)
from .grader import Grader, GraderConfig

__all__ = [
    "__version__",
    "load_rubric", "default_rubric_path", "thresholds",
    "Essay", "Rubric", "Criterion", "Level", "ScoreResult", "CriterionScore",
    "Annotation", "Grader", "GraderConfig",
]

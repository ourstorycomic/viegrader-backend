"""Sinh nhận xét tự động cho sinh viên.

Nguyên tắc sư phạm: phản hồi phải (1) gắn với tiêu chí rubric, (2) dẫn được bằng
chứng cụ thể trong bài, (3) nêu hành động sửa được, (4) không phán xét người viết.

Nếu có LLM, nhận xét của LLM được dùng làm phần diễn giải; phần bằng chứng định
lượng luôn do hệ thống sinh để bảo đảm chính xác.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..features.extractor import list_confusions, list_spell_errors, missing_keywords
from ..schema import Criterion, Rubric, ScoreResult

# Gợi ý hành động theo tiêu chí và mức độ
ADVICE: Dict[str, Dict[str, List[str]]] = {
    "noi_dung": {
        "low": [
            "Bổ sung các ý cốt lõi mà đề bài yêu cầu; hiện bài mới chạm tới một phần.",
            "Với mỗi khái niệm, hãy nêu định nghĩa rồi mới phân tích, tránh nói chung chung.",
        ],
        "mid": [
            "Đi sâu hơn vào 1–2 ý trọng tâm thay vì liệt kê nhiều ý ở mức bề mặt.",
            "Gắn lí thuyết với ví dụ trong học phần để chứng minh đã hiểu bản chất.",
        ],
        "high": ["Nội dung vững; có thể mở rộng bằng phản ví dụ hoặc giới hạn của lí thuyết."],
    },
    "lap_luan": {
        "low": [
            "Mỗi đoạn nên có một câu chủ đề đứng đầu nêu rõ luận điểm.",
            "Bổ sung từ nối (vì vậy, tuy nhiên, cụ thể là) để người đọc theo được mạch ý.",
        ],
        "mid": [
            "Sau mỗi dẫn chứng cần một câu phân tích chỉ ra dẫn chứng đó chứng minh điều gì.",
            "Cân nhắc nêu và phản biện một ý kiến trái chiều để lập luận chắc hơn.",
        ],
        "high": ["Lập luận mạch lạc; có thể sắp xếp luận điểm theo thứ tự tăng dần sức thuyết phục."],
    },
    "dien_dat": {
        "low": [
            "Rà lại chính tả trước khi nộp; nhiều lỗi làm giảm độ tin cậy của bài.",
            "Tách các câu dài trên 40 từ thành câu ngắn hơn.",
            "Thay từ ngữ văn nói bằng cách diễn đạt học thuật.",
        ],
        "mid": [
            "Đa dạng hoá cách mở đầu câu, tránh lặp cùng một cấu trúc.",
            "Kiểm tra dấu câu và cách viết hoa đầu câu.",
        ],
        "high": ["Diễn đạt tốt; chú ý giữ nhất quán thuật ngữ chuyên ngành."],
    },
    "hinh_thuc": {
        "low": [
            "Bài chưa đạt yêu cầu về độ dài hoặc bố cục đoạn.",
            "Bổ sung trích dẫn nguồn cho các số liệu và nhận định đã dẫn.",
        ],
        "mid": ["Trình bày đúng quy định nhưng cần thống nhất cách ghi nguồn."],
        "high": ["Hình thức đạt yêu cầu."],
    },
}

FLAG_MESSAGES: Dict[str, str] = {
    "BAI_TRONG": "Bài làm trống hoặc quá ngắn, không đủ căn cứ để chấm.",
    "NGHI_LAC_DE": "Nội dung có dấu hiệu không bám sát yêu cầu của đề bài.",
    "CHEP_DE_BAI": "Phần lớn nội dung là chép lại đề bài.",
    "QUA_NGAN": "Bài ngắn hơn nhiều so với độ dài tối thiểu quy định.",
    "NGHI_SAO_CHEP": "Bài trùng lặp cao với một bài khác trong cùng đợt; đã chuyển bộ phận chuyên môn xem xét.",
    "LOI_CHINH_TA_NANG": "Tỉ lệ lỗi chính tả rất cao.",
}


def _band(score: float, max_score: float) -> str:
    r = score / max_score if max_score else 0.0
    return "high" if r >= 0.8 else ("mid" if r >= 0.5 else "low")


def criterion_comment(
    criterion: Criterion, score: float, feats: Dict[str, float]
) -> str:
    band = _band(score, criterion.max_score)
    lv = min(criterion.levels, key=lambda l: abs(l.score - score)) if criterion.levels else None
    parts = [f"{criterion.name}: {score:g}/{criterion.max_score:g}"]
    if lv:
        parts.append(f"({lv.name}) — {lv.descriptor}")
    tips = ADVICE.get(criterion.key, {}).get(band, [])
    if tips:
        parts.append("Gợi ý: " + " ".join(tips[:2]))
    return " ".join(parts)


def build_feedback(
    result: ScoreResult,
    rubric: Rubric,
    text: str,
    feats: Dict[str, float],
    keywords: Optional[Sequence[str]] = None,
    llm_comment: str = "",
    show_evidence: bool = True,
) -> str:
    L: List[str] = []
    L.append(f"**Điểm tổng: {result.total:g}/{rubric.scale_max:g}**")
    L.append("")

    for cs in result.criteria:
        c = rubric.get(cs.key)
        L.append(f"- {criterion_comment(c, cs.score, feats)}")
    L.append("")

    if show_evidence:
        ev: List[str] = []
        errs = list_spell_errors(text, limit=8)
        if errs:
            ev.append("Từ nghi sai chính tả: " + ", ".join(f"*{e}*" for e in errs))
        conf = list_confusions(text)
        if conf:
            ev.append("Cặp từ hay nhầm: " + "; ".join(conf[:5]))
        if keywords:
            miss = missing_keywords(text, keywords)
            if miss:
                ev.append("Ý cốt lõi còn thiếu: " + ", ".join(miss[:8]))
        if feats.get("n_paragraphs", 0) <= 1 and feats.get("n_words", 0) > 150:
            ev.append("Bài chưa tách đoạn; nên chia theo từng luận điểm.")
        if feats.get("mean_sentence_len", 0) > 35:
            ev.append(f"Câu trung bình dài {feats['mean_sentence_len']:.0f} từ — nên rút gọn.")
        if feats.get("connective_variety", 0) < 3 and feats.get("n_words", 0) > 200:
            ev.append("Ít phương tiện liên kết; bổ sung từ nối giữa các đoạn.")
        if ev:
            L.append("**Bằng chứng cụ thể:**")
            L.extend(f"- {e}" for e in ev)
            L.append("")

    if result.flags:
        L.append("**Lưu ý:**")
        for f in result.flags:
            L.append(f"- {FLAG_MESSAGES.get(f, f)}")
        L.append("")

    if llm_comment:
        L.append("**Nhận xét chung:**")
        L.append(llm_comment.strip())
        L.append("")

    if result.needs_human_review:
        L.append(f"> ⚠️ Bài này được chuyển giảng viên phúc tra. Lí do: {result.review_reason}. "
                 f"Điểm hiển thị là điểm tham khảo của hệ thống, chưa phải điểm cuối cùng.")
    else:
        L.append("> Điểm do hệ thống chấm tự động theo rubric. "
                 "Sinh viên có quyền yêu cầu phúc khảo theo quy định của học phần.")
    return "\n".join(L)

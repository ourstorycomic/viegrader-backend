"""Sinh tự động Sổ tay gán nhãn và biểu mẫu chấm từ rubric.

Mục tiêu: mọi giám khảo dùng CHUNG một tài liệu được sinh từ chính file rubric
đang chạy trong hệ thống, tránh tình trạng "rubric trên giấy" khác "rubric trong
code" - lỗi thường gặp làm hỏng kết quả nghiên cứu.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import pandas as pd

from ..schema import Rubric


def build_handbook(rubric: Rubric, anchor_examples: Optional[pd.DataFrame] = None) -> str:
    """Xuất Sổ tay gán nhãn dạng Markdown."""
    L: List[str] = []
    L.append(f"# SỔ TAY GÁN NHÃN — {rubric.name}")
    L.append("")
    L.append(f"- **Mã rubric:** `{rubric.rubric_id}`")
    L.append(f"- **Thang điểm tổng:** {rubric.scale_max} (bước làm tròn {rubric.scale_step})")
    L.append(f"- **Phiên bản:** {rubric.meta.get('version', 'n/a')}")
    L.append("")
    L.append("## 1. Quy trình chấm của giám khảo")
    L.append("")
    L.append("1. Đọc trọn bài **một lượt** để có ấn tượng tổng thể, chưa cho điểm.")
    L.append("2. Đọc lượt hai, chấm **từng tiêu chí độc lập** theo thứ tự trong bảng dưới.")
    L.append("3. Với mỗi tiêu chí, chọn mức mô tả **sát nhất**; nếu phân vân giữa hai mức, "
              "chọn mức thấp hơn rồi ghi chú lí do.")
    L.append("4. Ghi ít nhất **một bằng chứng trích từ bài làm** cho mỗi tiêu chí bị trừ điểm.")
    L.append("5. Không xem điểm của giám khảo khác trước khi nộp (chấm mù đôi).")
    L.append("6. Không suy đoán về người viết; bài đã được ẩn danh.")
    L.append("")
    L.append("### Nguyên tắc bắt buộc")
    L.append("")
    L.append("- **Chấm theo mô tả, không theo cảm tính.** Nếu mô tả mức không khớp bài thực tế, "
             "báo lại để chỉnh rubric, không tự diễn giải riêng.")
    L.append("- **Không phạt hai lần cùng một lỗi** ở hai tiêu chí khác nhau.")
    L.append("- **Không thưởng độ dài.** Bài dài mà loãng không được điểm cao hơn bài ngắn mà chặt.")
    L.append("- **Bỏ qua chữ viết/trình bày** ngoài các yêu cầu đã nêu ở tiêu chí hình thức.")
    L.append("- Nghi vấn gian lận: **không tự trừ điểm**, gắn cờ và chuyển bộ phận xử lí.")
    L.append("")

    L.append("## 2. Bảng tiêu chí")
    L.append("")
    for c in rubric.criteria:
        L.append(f"### {c.name} (`{c.key}`) — trọng số {c.weight:.0%}, tối đa {c.max_score} điểm")
        L.append("")
        if c.description:
            L.append(f"> {c.description.strip()}")
            L.append("")
        L.append("| Mức | Điểm | Mô tả biểu hiện | Dấu hiệu định lượng |")
        L.append("|---|---|---|---|")
        for lv in sorted(c.levels, key=lambda x: -x.score):
            ind = "; ".join(lv.indicators) if lv.indicators else "—"
            L.append(f"| {lv.name} | {lv.score} | {lv.descriptor} | {ind} |")
        L.append("")

    L.append("## 3. Quy định chấm tự động đang áp dụng")
    L.append("")
    L.append("Các quy định này do hệ thống thi hành sau khi mô hình dự đoán điểm. "
             "Giám khảo cần nắm để hiểu vì sao điểm máy khác điểm mình.")
    L.append("")
    L.append("| Mã | Điều kiện | Hành động | Mô tả |")
    L.append("|---|---|---|---|")
    for r in rubric.rules:
        act = r.get("action", "")
        val = r.get("value", r.get("value_expr", ""))
        act_txt = f"`{act}`" + (f" = {val}" if val != "" else "")
        L.append(f"| {r.get('id','')} | `{r.get('when','')}` | {act_txt} | {r.get('description','')} |")
    L.append("")

    L.append("## 4. Quy tắc phân xử bất đồng")
    L.append("")
    L.append(f"- Hai giám khảo lệch **≤ {rubric.scale_max * 0.1:.1f} điểm**: lấy trung bình.")
    L.append(f"- Lệch **> {rubric.scale_max * 0.1:.1f} điểm**: giám khảo thứ ba chấm mù; "
             "điểm cuối là trung bình của hai điểm gần nhau nhất.")
    L.append("- Lệch **> 30% thang điểm**: đưa ra họp hội đồng, ghi biên bản, "
             "cân nhắc sửa mô tả mức trong rubric.")
    L.append("- Mọi ca phân xử phải được lưu để tính lại QWK sau khi hợp nhất nhãn.")
    L.append("")

    L.append("## 5. Ngưỡng chất lượng gán nhãn cần đạt")
    L.append("")
    L.append("| Chỉ số | Ngưỡng tối thiểu | Ý nghĩa |")
    L.append("|---|---|---|")
    L.append("| QWK giữa 2 giám khảo (điểm tổng) | ≥ 0.70 | Rubric đủ rõ để người khác nhau hiểu giống nhau |")
    L.append("| QWK từng tiêu chí | ≥ 0.60 | Tiêu chí không mơ hồ |")
    L.append("| Adjacent agreement (lệch ≤ 1 điểm) | ≥ 0.90 | Không có bất đồng lớn |")
    L.append("| Krippendorff's α | ≥ 0.667 | Ngưỡng dùng được cho kết luận sơ bộ |")
    L.append("| ICC(2,k) | ≥ 0.75 | Điểm trung bình nhiều giám khảo đủ tin cậy |")
    L.append("")
    L.append("Nếu chưa đạt: **sửa rubric và tập huấn lại**, không ép mô hình học nhãn nhiễu.")
    L.append("")

    if anchor_examples is not None and len(anchor_examples):
        L.append("## 6. Bài mẫu neo (anchor essays)")
        L.append("")
        for _, r in anchor_examples.iterrows():
            L.append(f"- **{r.get('essay_id')}** — điểm chuẩn {r.get('mean', r.get('total')):.2f} "
                     f"(mức {r.get('level', '')})")
        L.append("")

    L.append("## 7. Nhật ký phiên chấm")
    L.append("")
    L.append("Giám khảo ghi lại: ngày chấm, số bài, thời gian trung bình mỗi bài, "
             "các trường hợp khó. Hệ thống dùng nhật ký này để phát hiện mệt mỏi "
             "và trôi chuẩn (drift) theo thời gian.")
    return "\n".join(L)


def build_scoring_form(rubric: Rubric, essay_ids: Sequence[str],
                       rater_id: str = "") -> pd.DataFrame:
    """Biểu mẫu chấm dạng bảng, xuất ra .xlsx/.csv cho giám khảo điền."""
    cols = {"essay_id": list(essay_ids)}
    cols["rater_id"] = [rater_id] * len(essay_ids)
    for c in rubric.criteria:
        cols[c.key] = [None] * len(essay_ids)
        cols[f"{c.key}__evidence"] = [""] * len(essay_ids)
    cols["total"] = [None] * len(essay_ids)
    cols["flag"] = [""] * len(essay_ids)
    cols["note"] = [""] * len(essay_ids)
    cols["duration_sec"] = [None] * len(essay_ids)
    return pd.DataFrame(cols)


def validate_annotations(df: pd.DataFrame, rubric: Rubric) -> pd.DataFrame:
    """Kiểm tra tính hợp lệ của bảng nhãn do giám khảo nộp."""
    issues = []
    for _, r in df.iterrows():
        eid = r.get("essay_id")
        for c in rubric.criteria:
            v = r.get(c.key)
            if pd.isna(v):
                issues.append({"essay_id": eid, "rater_id": r.get("rater_id"),
                               "issue": f"Thiếu điểm tiêu chí '{c.key}'"})
                continue
            if not (0 <= float(v) <= c.max_score):
                issues.append({"essay_id": eid, "rater_id": r.get("rater_id"),
                               "issue": f"Điểm '{c.key}'={v} ngoài khoảng [0,{c.max_score}]"})
            allowed = c.level_scores()
            if allowed and float(v) not in allowed:
                issues.append({"essay_id": eid, "rater_id": r.get("rater_id"),
                               "issue": f"Điểm '{c.key}'={v} không thuộc mức rubric {allowed}"})
    return pd.DataFrame(issues)

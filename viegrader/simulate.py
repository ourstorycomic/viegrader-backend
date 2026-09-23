"""Sinh dữ liệu mô phỏng để chạy thử pipeline khi chưa có bộ dữ liệu thật.

Dữ liệu sinh ra CHỈ dùng để kiểm thử kĩ thuật (unit test, demo, đo tốc độ).
KHÔNG được dùng để báo cáo kết quả nghiên cứu - kết quả trên dữ liệu mô phỏng
không phản ánh năng lực thật của mô hình.

Cơ chế: mỗi bài được sinh từ một "mức năng lực" ẩn q ∈ [0,1]; q chi phối độ dài,
độ phủ ý cốt lõi, mật độ từ nối, tỉ lệ lỗi chính tả và mức dùng văn nói. Điểm
vàng được tính từ q theo đúng công thức rubric, có thêm nhiễu giám khảo.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .schema import Rubric

PROMPTS = {
    "P01": (
        "Trình bày khái niệm chuyển đổi số và phân tích tác động của chuyển đổi số "
        "đến hoạt động quản trị doanh nghiệp Việt Nam hiện nay.",
        ["chuyển đổi số", "dữ liệu|cơ sở dữ liệu", "quy trình", "công nghệ",
         "quản trị|quản lý", "nhân lực", "khách hàng", "hiệu quả", "rủi ro", "thể chế|chính sách"],
    ),
    "P02": (
        "Phân tích vai trò của vốn con người đối với tăng trưởng kinh tế. "
        "Liên hệ thực tiễn Việt Nam.",
        ["vốn con người", "giáo dục|đào tạo", "năng suất", "tăng trưởng",
         "kỹ năng", "y tế|sức khoẻ", "thu nhập", "đầu tư", "thị trường lao động", "chính sách"],
    ),
    "P03": (
        "Trình bày nguyên tắc kế toán dồn tích và so sánh với kế toán tiền mặt.",
        ["dồn tích", "tiền mặt", "doanh thu", "chi phí", "ghi nhận",
         "kỳ kế toán", "báo cáo tài chính", "dòng tiền", "nguyên tắc phù hợp", "chuẩn mực"],
    ),
}

_INTRO = [
    "Trong bối cảnh hiện nay, vấn đề này ngày càng được quan tâm.",
    "Có thể thấy đây là một nội dung quan trọng của học phần.",
    "Bài viết dưới đây trình bày các nội dung chính của vấn đề.",
]
_BODY = [
    "Trước hết, cần làm rõ nội hàm của khái niệm {kw}.",
    "Bên cạnh đó, {kw} giữ vai trò quan trọng trong toàn bộ quá trình.",
    "Cụ thể là, việc triển khai {kw} đòi hỏi nguồn lực và lộ trình phù hợp.",
    "Tuy nhiên, {kw} cũng đặt ra không ít thách thức cần được xử lí.",
    "Thực tế cho thấy {kw} tác động trực tiếp tới kết quả hoạt động.",
    "Vì vậy, {kw} cần được xem xét trong mối quan hệ với các yếu tố khác.",
]
_ANALYSIS = [
    "Điều này chứng tỏ mối quan hệ giữa các yếu tố là chặt chẽ và có tính hệ thống.",
    "Số liệu khảo sát gần đây cho thấy xu hướng này tiếp tục được duy trì.",
    "Ngược lại, nếu thiếu điều kiện cần thiết thì kết quả sẽ không như kì vọng.",
    "Từ đó có thể rút ra bài học về cách tổ chức thực hiện trong thực tiễn.",
]
_CONCL = [
    "Tóm lại, vấn đề trên có ý nghĩa quan trọng cả về lí luận và thực tiễn.",
    "Như vậy, có thể khẳng định nội dung đã phân tích là cần thiết.",
]
_FILLER = [
    "Đây là một vấn đề rất đáng chú ý.",
    "Nói chung là mọi thứ đều có liên quan tới nhau.",
    "Em nghĩ là điều này khá quan trọng ạ.",
]
_TYPO = {"và": "vaf", "của": "cuar", "được": "dc", "không": "ko", "những": "nhg",
         "chia sẻ": "chia sẽ", "bổ sung": "bổ xung", "xuất sắc": "xuất xắc"}


def _make_essay(q: float, keywords: Sequence[str], rng: random.Random) -> str:
    n_kw = max(1, int(round(q * len(keywords))))
    kws = rng.sample(list(keywords), n_kw)
    n_para = max(1, int(round(1 + q * 4)))

    paras: List[str] = []
    if q > 0.3:
        paras.append(rng.choice(_INTRO))
    per = max(1, len(kws) // max(1, n_para - 1)) if n_para > 1 else len(kws)
    for i in range(max(1, n_para - 1)):
        chunk = kws[i * per:(i + 1) * per] or [rng.choice(list(keywords))]
        sents = []
        for kw in chunk:
            sents.append(rng.choice(_BODY).format(kw=kw.split("|")[0]))
            if q > 0.5:
                sents.append(rng.choice(_ANALYSIS))
        if q < 0.4:
            sents.append(rng.choice(_FILLER))
        paras.append(" ".join(sents))
    if q > 0.55:
        paras.append(rng.choice(_CONCL))

    text = "\n\n".join(paras) if q > 0.35 else " ".join(paras)

    # Chèn lỗi chính tả theo mức năng lực
    err_rate = max(0.0, 0.28 * (1 - q))
    for good, bad in _TYPO.items():
        if rng.random() < err_rate * 2:
            text = text.replace(good, bad, rng.randint(1, 3))
    if q < 0.35 and rng.random() < 0.5:
        text += " Nói chung là vậy thôi nhé, mình nghĩ là đủ rồi ạ."
    return text


def make_dataset(
    rubric: Rubric,
    n: int = 300,
    seed: int = 42,
    rater_noise: float = 0.35,
    n_raters: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trả về (essays, gold, annotations)."""
    rng = random.Random(seed)
    nprng = np.random.RandomState(seed)

    rows, gold_rows, ann_rows = [], [], []
    pids = list(PROMPTS.keys())
    for i in range(n):
        pid = pids[i % len(pids)]
        prompt_text, kws = PROMPTS[pid]
        q = float(np.clip(nprng.beta(4, 3), 0.05, 0.98))
        text = _make_essay(q, kws, rng)
        eid = f"E{i:05d}"
        rows.append({
            "essay_id": eid, "text": text, "prompt_id": pid,
            "prompt_text": prompt_text, "course_id": "MH101",
            "student_id": f"SV{i:05d}", "keywords": ";".join(kws),
        })

        g: Dict[str, float] = {"essay_id": eid}
        total = 0.0
        for c in rubric.criteria:
            allowed = np.array(c.level_scores(), dtype=float)
            target = q * c.max_score
            s = float(allowed[np.argmin(np.abs(allowed - target))])
            g[c.key] = s
            g[f"gold_{c.key}"] = s
            total += (s / c.max_score) * c.weight * rubric.scale_max
        g["gold_total"] = round(total / rubric.scale_step) * rubric.scale_step
        g["total"] = g["gold_total"]
        g["latent_q"] = round(q, 4)
        gold_rows.append(g)

        for r in range(n_raters):
            noisy = {}
            t = 0.0
            for c in rubric.criteria:
                allowed = np.array(c.level_scores(), dtype=float)
                v = g[c.key] + nprng.normal(0, rater_noise * c.max_score / 4)
                v = float(allowed[np.argmin(np.abs(allowed - v))])
                noisy[c.key] = v
                t += (v / c.max_score) * c.weight * rubric.scale_max
            ann_rows.append({
                "essay_id": eid, "rater_id": f"GK{r+1}", **noisy,
                "total": round(t / rubric.scale_step) * rubric.scale_step,
                "batch": i // 50,
            })

    # Chèn vài trường hợp biên để kiểm thử quy định chấm
    edge = [
        {"essay_id": "E90001", "text": "", "note": "bài trống"},
        {"essay_id": "E90002", "text": "Em không biết làm ạ.", "note": "quá ngắn"},
        {"essay_id": "E90003", "text": PROMPTS["P01"][0] * 5, "note": "chép đề bài"},
        {"essay_id": "E90004", "text": "toi khong biet lam bai nay nhung toi se co gang "
                                       "viet mot doan van khong dau de kiem tra he thong "
                                       "co phat hien duoc hay khong " * 8,
         "note": "không dấu"},
    ]
    for e in edge:
        rows.append({
            "essay_id": e["essay_id"], "text": e["text"], "prompt_id": "P01",
            "prompt_text": PROMPTS["P01"][0], "course_id": "MH101",
            "student_id": e["essay_id"], "keywords": ";".join(PROMPTS["P01"][1]),
        })

    # Một cặp bài trùng lặp để kiểm thử phát hiện sao chép
    if len(rows) > 5:
        src = rows[3]
        rows.append({**src, "essay_id": "E90005", "student_id": "SV90005"})

    return (
        pd.DataFrame(rows),
        pd.DataFrame(gold_rows),
        pd.DataFrame(ann_rows),
    )

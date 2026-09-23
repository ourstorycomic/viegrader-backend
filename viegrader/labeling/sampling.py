"""Chiến lược lấy mẫu để gán nhãn - tối ưu công sức của giám khảo.

Bài toán: giảng viên chỉ chấm tay được vài trăm bài, nhưng mô hình cần dữ liệu
phủ đều dải điểm. Ba chiến lược được cài đặt:

    1. ``stratified_pilot``  - vòng thử: phân tầng theo độ dài để phủ dải năng lực,
       toàn bộ do >= 2 giám khảo chấm độc lập -> dùng đo QWK và chốt rubric.
    2. ``double_scoring_plan`` - vòng chính: mọi bài 1 giám khảo, 20-30% bài chấm
       kép để giám sát chất lượng liên tục, kèm 5% "bài kiểm tra ngầm"
       (seeded anchors) để phát hiện giám khảo trôi chuẩn.
    3. ``uncertainty_sampling`` - vòng mở rộng (active learning): ưu tiên gán nhãn
       những bài mô hình dự đoán kém chắc chắn nhất.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def stratified_pilot(
    df: pd.DataFrame,
    n: int = 200,
    strat_col: str = "n_words",
    n_bins: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Chọn mẫu thử phân tầng, mỗi tầng lấy đều."""
    d = df.copy()
    if strat_col not in d.columns:
        d[strat_col] = d["text"].str.split().str.len()
    d["_stratum"] = pd.qcut(d[strat_col].rank(method="first"), n_bins, labels=False)
    per = max(1, n // n_bins)
    out = (
        d.groupby("_stratum", group_keys=False)
        .apply(lambda g: g.sample(min(per, len(g)), random_state=seed))
        .drop(columns=["_stratum"])
    )
    return out.reset_index(drop=True)


def double_scoring_plan(
    df: pd.DataFrame,
    raters: Sequence[str],
    double_rate: float = 0.25,
    anchor_ids: Optional[Sequence[str]] = None,
    anchor_rate: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Sinh bảng phân công chấm: mỗi dòng = (essay_id, rater_id, vai trò).

    Vai trò: primary | secondary | anchor
    Bài anchor là bài đã có "điểm chuẩn" được hội đồng thống nhất, chèn ngầm vào
    danh sách của từng giám khảo để đo độ trôi chuẩn theo thời gian.
    """
    rng = np.random.RandomState(seed)
    raters = list(raters)
    if len(raters) < 2:
        raise ValueError("Cần ít nhất 2 giám khảo để lập kế hoạch chấm kép.")
    ids = list(df["essay_id"])
    rng.shuffle(ids)

    rows: List[Dict[str, object]] = []
    for k, eid in enumerate(ids):
        primary = raters[k % len(raters)]
        rows.append({"essay_id": eid, "rater_id": primary, "role": "primary", "batch": k // 50})
        if rng.rand() < double_rate:
            others = [r for r in raters if r != primary]
            rows.append({
                "essay_id": eid,
                "rater_id": others[rng.randint(len(others))],
                "role": "secondary",
                "batch": k // 50,
            })

    if anchor_ids:
        n_anchor = max(1, int(len(ids) * anchor_rate / max(1, len(raters))))
        for r in raters:
            for eid in rng.choice(list(anchor_ids), size=min(n_anchor, len(anchor_ids)),
                                  replace=False):
                rows.append({"essay_id": str(eid), "rater_id": r, "role": "anchor",
                             "batch": -1})

    plan = pd.DataFrame(rows)
    return plan.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def uncertainty_sampling(
    df: pd.DataFrame,
    pred_std: Sequence[float],
    n: int = 100,
    diversity_col: Optional[str] = "prompt_id",
) -> pd.DataFrame:
    """Chọn bài để gán nhãn tiếp theo: ưu tiên độ bất định cao, có ràng buộc đa dạng."""
    d = df.copy()
    d["_unc"] = np.asarray(pred_std, dtype=float)
    if diversity_col and diversity_col in d.columns and d[diversity_col].nunique() > 1:
        per = max(1, n // d[diversity_col].nunique())
        out = (
            d.sort_values("_unc", ascending=False)
            .groupby(diversity_col, group_keys=False)
            .head(per)
        )
        if len(out) < n:
            rest = d[~d["essay_id"].isin(out["essay_id"])].sort_values("_unc", ascending=False)
            out = pd.concat([out, rest.head(n - len(out))])
    else:
        out = d.sort_values("_unc", ascending=False).head(n)
    return out.drop(columns=["_unc"]).reset_index(drop=True)


def select_anchor_essays(
    annotations: pd.DataFrame, rubric_scale_max: float = 10.0, per_level: int = 3
) -> pd.DataFrame:
    """Chọn 'bài mẫu neo' (anchor/benchmark essays) cho từng mức điểm.

    Bài neo phục vụ 3 việc:
      - Tập huấn giám khảo trước khi chấm.
      - Kiểm tra ngầm độ trôi chuẩn trong khi chấm.
      - Làm ví dụ few-shot cho mô-đun chấm bằng LLM.
    Tiêu chí chọn: điểm trung bình gần tâm mức VÀ độ lệch giữa giám khảo thấp nhất.
    """
    g = annotations.groupby("essay_id")["total"].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    g = g[g["count"] >= 2]
    g["level"] = np.rint(g["mean"]).astype(int).clip(0, int(rubric_scale_max))
    g["dist_to_center"] = (g["mean"] - g["level"]).abs()
    g["quality"] = g["std"] + g["dist_to_center"]
    return (
        g.sort_values(["level", "quality"])
        .groupby("level", group_keys=False)
        .head(per_level)
        .reset_index(drop=True)
    )

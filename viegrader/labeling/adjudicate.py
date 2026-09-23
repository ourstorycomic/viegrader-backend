"""Hợp nhất nhãn của nhiều giám khảo thành nhãn vàng (gold label)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..schema import Rubric


def resolve(
    annotations: pd.DataFrame,
    rubric: Rubric,
    tol_ratio: float = 0.10,
    method: str = "closest_pair",
) -> pd.DataFrame:
    """Sinh nhãn vàng cho từng bài.

    Quy tắc (mặc định ``closest_pair``):
      - 1 giám khảo            -> lấy nguyên điểm, đánh dấu single_scored.
      - 2 giám khảo, lệch ≤ tol -> trung bình.
      - 2 giám khảo, lệch > tol -> chờ phân xử (needs_adjudication=True).
      - ≥3 giám khảo            -> trung bình của hai điểm gần nhau nhất.
    ``tol`` = tol_ratio * scale_max (mặc định 10% thang điểm).
    """
    tol = tol_ratio * rubric.scale_max
    score_cols = [c.key for c in rubric.criteria if c.key in annotations.columns]
    out: List[Dict[str, object]] = []

    for eid, g in annotations.groupby("essay_id"):
        rec: Dict[str, object] = {"essay_id": eid, "n_raters": len(g)}
        totals = g["total"].dropna().values if "total" in g.columns else np.array([])
        if len(totals) == 0 and score_cols:
            totals = g[score_cols].sum(axis=1).values

        if len(totals) == 1:
            gold, status = float(totals[0]), "single_scored"
        elif len(totals) == 2:
            if abs(totals[0] - totals[1]) <= tol:
                gold, status = float(np.mean(totals)), "agreed"
            else:
                gold, status = float(np.mean(totals)), "needs_adjudication"
        else:
            s = np.sort(totals)
            if method == "median":
                gold, status = float(np.median(s)), "median"
            else:
                diffs = np.diff(s)
                i = int(np.argmin(diffs))
                gold = float((s[i] + s[i + 1]) / 2)
                status = "closest_pair" if diffs[i] <= tol else "needs_adjudication"

        rec["gold_total"] = round(gold, 4)
        rec["status"] = status
        rec["rater_range"] = float(np.ptp(totals)) if len(totals) else 0.0
        rec["rater_std"] = float(np.std(totals, ddof=1)) if len(totals) > 1 else 0.0
        for c in score_cols:
            vals = g[c].dropna().values
            rec[f"gold_{c}"] = round(float(np.mean(vals)), 4) if len(vals) else np.nan
            rec[f"range_{c}"] = float(np.ptp(vals)) if len(vals) else 0.0
        out.append(rec)

    df = pd.DataFrame(out)
    df["needs_adjudication"] = df["status"] == "needs_adjudication"
    return df.sort_values("rater_range", ascending=False).reset_index(drop=True)


def snap_to_rubric(df: pd.DataFrame, rubric: Rubric) -> pd.DataFrame:
    """Neo điểm vàng từng tiêu chí về đúng các mức rubric cho phép."""
    d = df.copy()
    for c in rubric.criteria:
        col = f"gold_{c.key}"
        if col not in d.columns:
            continue
        allowed = np.array(c.level_scores(), dtype=float)
        if allowed.size == 0:
            continue
        d[col] = d[col].apply(
            lambda v: float(allowed[np.argmin(np.abs(allowed - v))]) if pd.notna(v) else v
        )
    step = rubric.scale_step
    d["gold_total"] = (d["gold_total"] / step).round() * step
    return d


def rater_drift(annotations: pd.DataFrame, anchor_gold: Dict[str, float]) -> pd.DataFrame:
    """Đo độ trôi chuẩn của từng giám khảo trên các bài neo theo thời gian/lô."""
    a = annotations[annotations["essay_id"].isin(anchor_gold.keys())].copy()
    if a.empty:
        return pd.DataFrame(columns=["rater_id", "batch", "bias", "mae", "n"])
    a["gold"] = a["essay_id"].map(anchor_gold)
    a["err"] = a["total"] - a["gold"]
    key = ["rater_id"] + (["batch"] if "batch" in a.columns else [])
    g = a.groupby(key)["err"].agg(bias="mean", mae=lambda x: x.abs().mean(), n="count")
    return g.reset_index()


def rater_severity(annotations: pd.DataFrame) -> pd.DataFrame:
    """Ước lượng độ 'khắt khe/dễ dãi' của từng giám khảo (Many-Facet Rasch rút gọn).

    Dùng mô hình cộng tính: score_ij = mu + theta_i (năng lực bài) + beta_j (độ dễ
    dãi của giám khảo). Ước lượng luân phiên vài vòng; beta_j âm = khắt khe.
    """
    a = annotations.dropna(subset=["total"]).copy()
    mu = a["total"].mean()
    theta = {e: 0.0 for e in a["essay_id"].unique()}
    beta = {r: 0.0 for r in a["rater_id"].unique()}
    for _ in range(30):
        for e, g in a.groupby("essay_id"):
            theta[e] = float((g["total"] - mu - g["rater_id"].map(beta)).mean())
        for r, g in a.groupby("rater_id"):
            beta[r] = float((g["total"] - mu - g["essay_id"].map(theta)).mean())
    out = pd.DataFrame({"rater_id": list(beta.keys()), "leniency": list(beta.values())})
    out["interpretation"] = np.where(
        out["leniency"] > 0.25, "Dễ dãi",
        np.where(out["leniency"] < -0.25, "Khắt khe", "Cân bằng"),
    )
    n = a.groupby("rater_id").size().rename("n_scored")
    return out.merge(n, on="rater_id", how="left").sort_values("leniency")

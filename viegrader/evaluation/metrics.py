"""Đánh giá hiệu năng hệ thống chấm tự động.

Bộ chỉ số chuẩn của lĩnh vực AES, dùng trực tiếp cho phần thực nghiệm của báo cáo:

    QWK               - chỉ số chính, so sánh được với các công bố quốc tế
    Pearson / Spearman- tương quan với điểm người
    MAE / RMSE        - sai số tuyệt đối trên thang điểm
    Exact / Adjacent  - tỉ lệ trùng khớp và lệch ≤ 1 điểm
    SMD               - chênh lệch trung bình chuẩn hoá (đo thiên lệch hệ thống)
    Human benchmark   - so QWK(máy, người) với QWK(người, người): hệ thống đạt
                        chuẩn khi tỉ số ≥ 0.90
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..labeling.agreement import (
    adjacent_agreement,
    exact_agreement,
    pearson,
    quadratic_weighted_kappa,
    spearman,
)


def standardized_mean_difference(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    a, b = np.asarray(y_true, float), np.asarray(y_pred, float)
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((b.mean() - a.mean()) / s) if s > 0 else 0.0


def score_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    step: float = 0.25,
    adjacent_tol: float = 1.0,
) -> Dict[str, float]:
    a, b = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "n": float(len(a)),
        "QWK": quadratic_weighted_kappa(a, b, step),
        "pearson": pearson(a, b),
        "spearman": spearman(a, b),
        "MAE": float(np.mean(np.abs(a - b))),
        "RMSE": float(np.sqrt(np.mean((a - b) ** 2))),
        "exact_agreement": exact_agreement(a, b, step),
        "adjacent_agreement": adjacent_agreement(a, b, adjacent_tol),
        "SMD": standardized_mean_difference(a, b),
        "bias": float(np.mean(b - a)),
        "pred_std": float(b.std()),
        "true_std": float(a.std()),
        "std_ratio": float(b.std() / a.std()) if a.std() > 0 else np.nan,
    }


def per_criterion_metrics(
    gold: pd.DataFrame, pred: pd.DataFrame, keys: Sequence[str], step: float = 0.25
) -> pd.DataFrame:
    rows = []
    for k in keys:
        gk = k if k in gold.columns else f"gold_{k}"
        if gk not in gold.columns or k not in pred.columns:
            continue
        mask = gold[gk].notna() & pred[k].notna()
        if mask.sum() < 5:
            continue
        m = score_metrics(gold.loc[mask, gk], pred.loc[mask, k], step)
        m["criterion"] = k
        rows.append(m)
    df = pd.DataFrame(rows)
    return df[["criterion"] + [c for c in df.columns if c != "criterion"]] if len(df) else df


def human_benchmark(
    qwk_machine_human: float, qwk_human_human: float
) -> Dict[str, float]:
    """Hệ thống được coi là 'đạt chuẩn thay thế giám khảo thứ hai' khi tỉ số ≥ 0.90."""
    ratio = qwk_machine_human / qwk_human_human if qwk_human_human else np.nan
    return {
        "QWK_machine_human": qwk_machine_human,
        "QWK_human_human": qwk_human_human,
        "ratio": float(ratio),
        "passed": float(ratio >= 0.90) if not np.isnan(ratio) else 0.0,
    }


def review_efficiency(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    needs_review: Sequence[bool],
    tol: float = 1.0,
) -> Dict[str, float]:
    """Đánh giá cơ chế chuyển phúc tra.

    Mục tiêu: các bài mô hình chấm sai nhiều PHẢI nằm trong nhóm được chuyển
    phúc tra. Chỉ số quan trọng nhất là ``recall_of_errors``.
    """
    a, b = np.asarray(y_true, float), np.asarray(y_pred, float)
    r = np.asarray(needs_review, bool)
    err = np.abs(a - b) > tol
    n = len(a)
    return {
        "review_rate": float(r.mean()),
        "error_rate_overall": float(err.mean()),
        "error_rate_auto_accepted": float(err[~r].mean()) if (~r).sum() else 0.0,
        "recall_of_errors": float((err & r).sum() / err.sum()) if err.sum() else 1.0,
        "precision_of_review": float((err & r).sum() / r.sum()) if r.sum() else 0.0,
        "n_auto_accepted": float((~r).sum()),
        "workload_saved": float((~r).sum() / n) if n else 0.0,
    }


def review_threshold_curve(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    confidence: Sequence[float],
    tol: float = 1.0,
    grid: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Đường đánh đổi giữa khối lượng phúc tra và độ an toàn.

    Dùng để chọn ngưỡng ``confidence`` trong quy định R07 sao cho vừa giữ được
    tỉ lệ phúc tra chấp nhận được, vừa bảo đảm hầu hết bài bị chấm sai đều lọt
    vào nhóm phúc tra.
    """
    grid = grid if grid is not None else np.arange(0.30, 0.96, 0.05)
    conf = np.asarray(confidence, float)
    rows = []
    for t in grid:
        m = review_efficiency(y_true, y_pred, conf < t, tol)
        m["threshold"] = float(t)
        rows.append(m)
    df = pd.DataFrame(rows)
    cols = ["threshold", "review_rate", "recall_of_errors",
            "error_rate_auto_accepted", "workload_saved"]
    return df[[c for c in cols if c in df.columns]]


def suggest_review_threshold(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    confidence: Sequence[float],
    target_review_rate: float = 0.15,
    min_recall: float = 0.80,
    tol: float = 1.0,
) -> Dict[str, float]:
    """Đề xuất ngưỡng: ưu tiên đạt ``min_recall`` với khối lượng phúc tra nhỏ nhất."""
    curve = review_threshold_curve(y_true, y_pred, confidence, tol)
    ok = curve[curve["recall_of_errors"] >= min_recall]
    if len(ok):
        best = ok.loc[ok["review_rate"].idxmin()]
    else:
        best = curve.loc[(curve["review_rate"] - target_review_rate).abs().idxmin()]
    return {k: float(v) for k, v in best.items()}


def fairness_by_group(
    df: pd.DataFrame, group_col: str, true_col: str = "gold_total", pred_col: str = "total"
) -> pd.DataFrame:
    """Kiểm tra thiên lệch theo nhóm (độ dài bài, lớp, đề bài...).

    Cảnh báo khi |SMD| > 0.25 hoặc chênh MAE giữa các nhóm > 0.5 điểm - dấu hiệu
    mô hình đối xử không đồng đều.
    """
    rows = []
    for g, sub in df.groupby(group_col):
        if len(sub) < 10:
            continue
        m = score_metrics(sub[true_col], sub[pred_col])
        m["group"] = g
        m["n"] = len(sub)
        rows.append(m)
    out = pd.DataFrame(rows)
    if len(out):
        out["warn_bias"] = out["SMD"].abs() > 0.25
        out["warn_mae"] = out["MAE"] > out["MAE"].min() + 0.5
        cols = ["group", "n", "QWK", "MAE", "SMD", "bias", "warn_bias", "warn_mae"]
        out = out[[c for c in cols if c in out.columns]]
    return out


def length_bias(df: pd.DataFrame, n_bins: int = 5,
                true_col: str = "gold_total", pred_col: str = "total",
                len_col: str = "n_words") -> pd.DataFrame:
    """Kiểm tra 'thiên lệch độ dài' - lỗi kinh điển của AES: mô hình học rằng
    bài dài = điểm cao."""
    d = df.copy()
    if len_col not in d.columns:
        d[len_col] = d["text"].str.split().str.len()
    d["_bin"] = pd.qcut(d[len_col].rank(method="first"), n_bins, labels=False)
    out = fairness_by_group(d, "_bin", true_col, pred_col)
    if len(out):
        out = out.rename(columns={"group": "length_bin"})
    return out


def evaluation_report(
    gold: pd.DataFrame,
    pred: pd.DataFrame,
    rubric_keys: Sequence[str],
    step: float = 0.25,
    qwk_human_human: Optional[float] = None,
) -> str:
    gt = "gold_total" if "gold_total" in gold.columns else "total"
    overall = score_metrics(gold[gt], pred["total"], step)
    L = ["===== BÁO CÁO ĐÁNH GIÁ HỆ THỐNG CHẤM TỰ ĐỘNG =====", "", "## Điểm tổng"]
    for k, v in overall.items():
        L.append(f"  {k:<20}: {v:.4f}")
    pc = per_criterion_metrics(gold, pred, rubric_keys, step)
    if len(pc):
        L.append("")
        L.append("## Theo từng tiêu chí")
        L.append(pc.round(4).to_string(index=False))
    if qwk_human_human:
        L.append("")
        L.append("## So với chuẩn giám khảo")
        hb = human_benchmark(overall["QWK"], qwk_human_human)
        for k, v in hb.items():
            L.append(f"  {k:<20}: {v:.4f}")
        L.append("  => " + ("ĐẠT chuẩn thay thế giám khảo thứ hai."
                            if hb["passed"] else "CHƯA đạt, cần cải thiện mô hình."))
    L.append("")
    L.append("## Ngưỡng tham chiếu của lĩnh vực")
    L.append("  QWK ≥ 0.70: dùng được cho chấm hỗ trợ")
    L.append("  QWK ≥ 0.80: tương đương giám khảo thứ hai trong nhiều nghiên cứu")
    L.append("  |SMD| ≤ 0.15: không thiên lệch hệ thống")
    L.append("  std_ratio ∈ [0.85, 1.15]: không bị dồn điểm về trung bình")
    return "\n".join(L)

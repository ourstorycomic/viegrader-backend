"""Đo độ tin cậy của quá trình gán nhãn.

Các chỉ số dùng trong AES (Automated Essay Scoring):
    - QWK (Quadratic Weighted Kappa): chỉ số chuẩn của lĩnh vực, phạt nặng khi
      lệch nhiều bậc. Ngưỡng chấp nhận trong nghiên cứu: >= 0.70 giữa 2 giám khảo.
    - Krippendorff's alpha (ordinal): dùng được khi >2 giám khảo và có ô trống.
    - ICC(2,k): độ tin cậy của điểm trung bình nhiều giám khảo.
    - Adjacent agreement: tỉ lệ hai giám khảo lệch <= 1 bậc.
    - Pearson / Spearman: tương quan tuyến tính / thứ bậc.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _to_bins(x: Sequence[float], step: float) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return np.rint(a / step).astype(int)


def quadratic_weighted_kappa(
    y1: Sequence[float], y2: Sequence[float], step: float = 0.25
) -> float:
    """QWK cho điểm liên tục: rời rạc hoá theo bước ``step`` rồi tính kappa."""
    a, b = _to_bins(y1, step), _to_bins(y2, step)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    n = hi - lo + 1
    if n <= 1:
        return 1.0
    O = np.zeros((n, n))
    for i, j in zip(a - lo, b - lo):
        O[i, j] += 1
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            W[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)
    ha = np.bincount(a - lo, minlength=n).astype(float)
    hb = np.bincount(b - lo, minlength=n).astype(float)
    E = np.outer(ha, hb)
    E = E * (O.sum() / E.sum())
    denom = (W * E).sum()
    if denom == 0:
        return 1.0
    return float(1.0 - (W * O).sum() / denom)


def adjacent_agreement(
    y1: Sequence[float], y2: Sequence[float], tol: float = 1.0
) -> float:
    a, b = np.asarray(y1, float), np.asarray(y2, float)
    return float(np.mean(np.abs(a - b) <= tol))


def exact_agreement(y1: Sequence[float], y2: Sequence[float], step: float = 0.25) -> float:
    return float(np.mean(_to_bins(y1, step) == _to_bins(y2, step)))


def krippendorff_alpha_ordinal(matrix: np.ndarray) -> float:
    """Krippendorff's alpha cho thang thứ bậc.

    ``matrix`` : mảng (n_raters, n_items), giá trị NaN cho ô chưa chấm.
    """
    m = np.asarray(matrix, dtype=float)
    vals = m[~np.isnan(m)]
    if vals.size == 0:
        return float("nan")
    levels = np.unique(vals)
    idx = {v: i for i, v in enumerate(levels)}
    K = len(levels)
    if K <= 1:
        return 1.0

    # Ma trận trùng khớp (coincidence matrix)
    coinc = np.zeros((K, K))
    for item in range(m.shape[1]):
        col = m[:, item]
        col = col[~np.isnan(col)]
        mu = len(col)
        if mu < 2:
            continue
        for c, d in combinations(range(mu), 2):
            i, j = idx[col[c]], idx[col[d]]
            coinc[i, j] += 1 / (mu - 1)
            coinc[j, i] += 1 / (mu - 1)
    n_c = coinc.sum(axis=1)
    n_total = coinc.sum()
    if n_total == 0:
        return float("nan")

    # Hàm khoảng cách thứ bậc
    def d_ord(i: int, j: int) -> float:
        lo, hi = (i, j) if i <= j else (j, i)
        s = n_c[lo] / 2 + n_c[hi] / 2 + n_c[lo + 1:hi].sum()
        return float(s ** 2)

    D = np.array([[d_ord(i, j) for j in range(K)] for i in range(K)])
    Do = (coinc * D).sum() / n_total
    De = 0.0
    for i in range(K):
        for j in range(K):
            De += n_c[i] * n_c[j] * D[i, j]
    De /= max(n_total * (n_total - 1), 1e-12)
    if De == 0:
        return 1.0
    return float(1 - Do / De)


def icc_2k(matrix: np.ndarray) -> float:
    """ICC(2,k) - two-way random, absolute agreement, average measures."""
    m = np.asarray(matrix, dtype=float)
    m = m[:, ~np.isnan(m).any(axis=0)]
    k, n = m.shape                     # k giám khảo, n bài
    if n < 2 or k < 2:
        return float("nan")
    grand = m.mean()
    ms_r = k * ((m.mean(axis=0) - grand) ** 2).sum() / (n - 1)          # giữa bài
    ms_c = n * ((m.mean(axis=1) - grand) ** 2).sum() / (k - 1)          # giữa giám khảo
    resid = m - m.mean(axis=0)[None, :] - m.mean(axis=1)[:, None] + grand
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_r + (ms_c - ms_e) / n
    if denom <= 0:
        return float("nan")
    return float((ms_r - ms_e) / denom)


def pearson(y1: Sequence[float], y2: Sequence[float]) -> float:
    a, b = np.asarray(y1, float), np.asarray(y2, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(y1: Sequence[float], y2: Sequence[float]) -> float:
    a = pd.Series(y1).rank()
    b = pd.Series(y2).rank()
    return pearson(a, b)


# --------------------------------------------------------------------------- #
def rater_matrix(
    annotations: pd.DataFrame, criterion: str = "total", essay_col: str = "essay_id"
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Chuyển bảng nhãn dài -> ma trận (n_raters, n_essays)."""
    piv = annotations.pivot_table(
        index="rater_id", columns=essay_col, values=criterion, aggfunc="mean"
    )
    return piv.values, list(piv.index), list(piv.columns)


def agreement_report(
    annotations: pd.DataFrame,
    criteria: Sequence[str] = ("total",),
    step: float = 0.25,
) -> pd.DataFrame:
    """Bảng tổng hợp độ đồng thuận cho từng tiêu chí.

    ``annotations`` phải có cột: essay_id, rater_id, và các cột điểm.
    """
    rows = []
    for crit in criteria:
        if crit not in annotations.columns:
            continue
        M, raters, essays = rater_matrix(annotations, crit)
        pairwise_qwk, pairwise_adj, pairwise_r = [], [], []
        for i, j in combinations(range(len(raters)), 2):
            mask = ~np.isnan(M[i]) & ~np.isnan(M[j])
            if mask.sum() < 5:
                continue
            pairwise_qwk.append(quadratic_weighted_kappa(M[i][mask], M[j][mask], step))
            pairwise_adj.append(adjacent_agreement(M[i][mask], M[j][mask], tol=step * 4))
            pairwise_r.append(pearson(M[i][mask], M[j][mask]))
        rows.append({
            "criterion": crit,
            "n_raters": len(raters),
            "n_essays": len(essays),
            "n_double_scored": int((~np.isnan(M)).sum(axis=0).__ge__(2).sum()),
            "QWK_mean": float(np.nanmean(pairwise_qwk)) if pairwise_qwk else np.nan,
            "QWK_min": float(np.nanmin(pairwise_qwk)) if pairwise_qwk else np.nan,
            "adjacent_agreement": float(np.nanmean(pairwise_adj)) if pairwise_adj else np.nan,
            "pearson_mean": float(np.nanmean(pairwise_r)) if pairwise_r else np.nan,
            "krippendorff_alpha": krippendorff_alpha_ordinal(np.rint(M / step)),
            "ICC_2k": icc_2k(M),
        })
    return pd.DataFrame(rows)


def disagreements(
    annotations: pd.DataFrame, criterion: str = "total", tol: float = 1.0
) -> pd.DataFrame:
    """Danh sách bài cần phân xử: hai giám khảo lệch quá ``tol``."""
    g = annotations.groupby("essay_id")[criterion].agg(["count", "min", "max", "mean", "std"])
    g["range"] = g["max"] - g["min"]
    out = g[(g["count"] >= 2) & (g["range"] > tol)].copy()
    out["need_adjudication"] = True
    return out.sort_values("range", ascending=False).reset_index()

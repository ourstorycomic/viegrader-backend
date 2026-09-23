from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

SUS_ITEMS = [f"sus_{i}" for i in range(1, 11)]


def score_sus(row: Sequence[float]) -> float:
    values = np.asarray(row, dtype=float)
    if values.shape != (10,) or np.isnan(values).any() or ((values < 1) | (values > 5)).any():
        raise ValueError("SUS cần đúng 10 câu, mỗi câu từ 1 đến 5.")
    adjusted = np.where(np.arange(10) % 2 == 0, values - 1, 5 - values)
    return float(adjusted.sum() * 2.5)


def _cronbach_alpha(matrix: np.ndarray) -> float:
    k = matrix.shape[1]
    total_var = matrix.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - matrix.var(axis=0, ddof=1).sum() / total_var)) if total_var else 0.0


def summarize_sus(df: pd.DataFrame, item_cols=SUS_ITEMS) -> Dict[str, float]:
    missing = [c for c in item_cols if c not in df]
    if missing:
        raise ValueError(f"Thiếu cột SUS: {missing}")
    clean = df[list(item_cols)].dropna().astype(float)
    scores = clean.apply(lambda r: score_sus(r.values), axis=1)
    return {
        "n": float(len(scores)), "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "median": float(scores.median()), "min": float(scores.min()),
        "max": float(scores.max()), "cronbach_alpha": _cronbach_alpha(clean.to_numpy()),
    }

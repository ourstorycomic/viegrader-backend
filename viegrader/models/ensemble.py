"""Tổng hợp các nguồn điểm thành điểm cuối cùng.

Ba nguồn:
    F - mô hình đặc trưng tường minh (TraitModel)
    S - mô hình ngữ nghĩa PhoBERT (đã nằm trong đặc trưng của TraitModel dưới dạng
        vector giảm chiều, hoặc dùng riêng)
    L - LLM chấm theo rubric

Cách hợp nhất: hồi quy tuyến tính không âm (NNLS) trên tập kiểm định, học riêng
cho TỪNG tiêu chí. Trọng số không âm và tổng bằng 1 nên vẫn giải thích được:
"điểm nội dung = 0.42 × mô hình đặc trưng + 0.58 × LLM".

Nếu không có tập kiểm định hoặc thiếu nguồn, hệ thống lùi về trung bình có trọng
số mặc định đã khai báo trong ``EnsembleConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from ..schema import Rubric


@dataclass
class EnsembleConfig:
    default_weights: Dict[str, float] = field(
        default_factory=lambda: {"feature": 0.55, "llm": 0.45}
    )
    min_val_size: int = 30
    shrink_to_mean: float = 0.0     # >0: kéo điểm về trung bình (giảm phương sai)


class ScoreEnsemble:
    def __init__(self, rubric: Rubric, cfg: Optional[EnsembleConfig] = None):
        self.rubric = rubric
        self.cfg = cfg or EnsembleConfig()
        self.weights: Dict[str, Dict[str, float]] = {}   # {criterion: {source: w}}
        self.fitted = False

    def fit(
        self,
        sources: Dict[str, pd.DataFrame],
        y: pd.DataFrame,
    ) -> "ScoreEnsemble":
        """``sources`` = {'feature': df_pred, 'llm': df_pred}, mỗi df có cột = key tiêu chí.
        ``y`` = điểm vàng từng tiêu chí."""
        names = [k for k, v in sources.items() if v is not None and len(v)]
        for c in self.rubric.criteria:
            cols = [n for n in names if c.key in sources[n].columns]
            gold_col = c.key if c.key in y.columns else f"gold_{c.key}"
            if not cols or gold_col not in y.columns:
                continue
            mask = y[gold_col].notna()
            for n in cols:
                mask &= sources[n][c.key].notna().values
            if mask.sum() < self.cfg.min_val_size:
                self.weights[c.key] = self._default(cols)
                continue
            A = np.column_stack([sources[n].loc[mask.values, c.key].values for n in cols])
            b = y.loc[mask, gold_col].values
            # Ràng buộc tổng trọng số = 1 bằng cách thêm hàng phạt
            lam = np.sqrt(len(b)) * b.std() if b.std() > 0 else 1.0
            A_aug = np.vstack([A, lam * np.ones((1, A.shape[1]))])
            b_aug = np.concatenate([b, [lam]])
            w, _ = nnls(A_aug, b_aug)
            if w.sum() <= 1e-9:
                self.weights[c.key] = self._default(cols)
            else:
                w = w / w.sum()
                self.weights[c.key] = {n: float(wi) for n, wi in zip(cols, w)}
        self.fitted = True
        return self

    def _default(self, cols: Sequence[str]) -> Dict[str, float]:
        d = {c: self.cfg.default_weights.get(c, 1.0) for c in cols}
        s = sum(d.values()) or 1.0
        return {k: v / s for k, v in d.items()}

    def predict(self, sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        names = [k for k, v in sources.items() if v is not None and len(v)]
        if not names:
            raise ValueError("Không có nguồn điểm nào để tổng hợp.")
        index = sources[names[0]].index
        out = pd.DataFrame(index=index)
        for c in self.rubric.criteria:
            cols = [n for n in names if c.key in sources[n].columns]
            if not cols:
                continue
            w = self.weights.get(c.key) or self._default(cols)
            w = {k: v for k, v in w.items() if k in cols}
            s = sum(w.values()) or 1.0
            acc = np.zeros(len(index))
            for n, wi in w.items():
                acc += (wi / s) * sources[n][c.key].fillna(
                    sources[n][c.key].mean()
                ).values
            if self.cfg.shrink_to_mean > 0:
                acc = (1 - self.cfg.shrink_to_mean) * acc + self.cfg.shrink_to_mean * acc.mean()
            out[c.key] = np.clip(acc, 0.0, c.max_score)
        return out

    def explain(self) -> pd.DataFrame:
        rows = []
        for c in self.rubric.criteria:
            w = self.weights.get(c.key, {})
            row = {"criterion": c.key, "name": c.name}
            row.update({f"w_{k}": round(v, 4) for k, v in w.items()})
            rows.append(row)
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def snap_to_levels(scores: pd.DataFrame, rubric: Rubric) -> pd.DataFrame:
    """Neo điểm thành phần về đúng mức rubric cho phép."""
    out = scores.copy()
    for c in rubric.criteria:
        if c.key not in out.columns:
            continue
        allowed = np.array(c.level_scores(), dtype=float)
        if allowed.size == 0:
            continue
        out[c.key] = out[c.key].apply(
            lambda v: float(allowed[np.argmin(np.abs(allowed - v))])
        )
    return out


def level_name(criterion, score: float) -> str:
    if not criterion.levels:
        return ""
    return min(criterion.levels, key=lambda lv: abs(lv.score - score)).name


def round_total(total: float, rubric: Rubric) -> float:
    step = rubric.scale_step or 0.25
    return float(np.clip(round(total / step) * step, 0.0, rubric.scale_max))

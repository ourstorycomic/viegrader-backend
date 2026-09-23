"""Mô hình chấm theo từng tiêu chí (per-trait scoring).

Thiết kế:
    - Mỗi tiêu chí rubric có một bộ hồi quy riêng -> điểm thành phần giải thích
      được, đúng tinh thần chấm theo rubric (khác cách chỉ dự đoán điểm tổng).
    - Đầu vào = đặc trưng tường minh (bắt buộc) + vector ngữ nghĩa (tuỳ chọn).
    - Ước lượng độ bất định bằng bagging: huấn luyện K mô hình trên các mẫu
      bootstrap, độ lệch chuẩn của K dự đoán chính là tín hiệu để quyết định
      "chuyển giám khảo phúc tra".
    - Hiệu chỉnh (calibration) bằng hồi quy đẳng hướng để phân phối điểm máy
      khớp phân phối điểm người - tránh hiện tượng mô hình dồn điểm về trung bình.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..schema import Criterion, Rubric


@dataclass
class TraitModelConfig:
    algo: str = "gbr"                 # 'ridge' | 'gbr'
    n_bags: int = 5                   # số mô hình bootstrap để ước lượng bất định
    use_semantic: bool = True
    semantic_dim: int = 64            # giảm chiều vector ngữ nghĩa trước khi ghép
    calibrate: bool = True
    random_state: int = 42


def _make_regressor(algo: str, seed: int):
    if algo == "ridge":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ])
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=seed)),
    ])


class TraitModel:
    """Bộ hồi quy cho MỘT tiêu chí."""

    def __init__(self, criterion: Criterion, cfg: Optional[TraitModelConfig] = None):
        self.criterion = criterion
        self.cfg = cfg or TraitModelConfig()
        self.feature_names: List[str] = []
        self.models: List[Any] = []
        self.calibrator: Optional[IsotonicRegression] = None
        self.train_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ưu tiên đặc trưng được rubric gợi ý, nhưng vẫn giữ toàn bộ đặc trưng
        số để mô hình tự chọn - tránh bỏ sót tín hiệu."""
        num = X.select_dtypes(include=[np.number]).copy()
        hints = [h for h in self.criterion.feature_hints if h in num.columns]
        if hints:                       # nhân đôi trọng số bằng cách thêm bản sao
            for h in hints:
                num[f"__hint__{h}"] = num[h]
        return num

    def fit(self, X: pd.DataFrame, y: Sequence[float]) -> "TraitModel":
        Xn = self._select_features(X)
        self.feature_names = list(Xn.columns)
        Xv, yv = Xn.values.astype(float), np.asarray(y, dtype=float)
        rng = np.random.RandomState(self.cfg.random_state)
        n = len(yv)

        self.models = []
        oof = np.zeros(n)
        oof_cnt = np.zeros(n)
        for b in range(self.cfg.n_bags):
            idx = rng.choice(n, n, replace=True)
            oob = np.setdiff1d(np.arange(n), np.unique(idx))
            m = _make_regressor(self.cfg.algo, self.cfg.random_state + b)
            m.fit(Xv[idx], yv[idx])
            self.models.append(m)
            if len(oob):
                oof[oob] += m.predict(Xv[oob])
                oof_cnt[oob] += 1

        mask = oof_cnt > 0
        oof_pred = np.where(mask, oof / np.maximum(oof_cnt, 1), yv.mean())

        if self.cfg.calibrate and mask.sum() > 10:
            self.calibrator = IsotonicRegression(
                y_min=0.0, y_max=self.criterion.max_score, out_of_bounds="clip"
            ).fit(oof_pred[mask], yv[mask])

        err = np.abs(oof_pred[mask] - yv[mask]) if mask.sum() else np.array([0.0])
        self.train_stats = {
            "n_train": float(n),
            "oof_mae": float(err.mean()),
            "y_mean": float(yv.mean()),
            "y_std": float(yv.std()),
        }
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Trả về (điểm dự đoán, độ lệch chuẩn giữa các mô hình bootstrap)."""
        Xn = self._select_features(X)
        for c in self.feature_names:
            if c not in Xn.columns:
                Xn[c] = np.nan
        Xv = Xn[self.feature_names].values.astype(float)
        preds = np.vstack([m.predict(Xv) for m in self.models])
        mean, std = preds.mean(axis=0), preds.std(axis=0)
        if self.calibrator is not None:
            mean = self.calibrator.predict(mean)
        return np.clip(mean, 0.0, self.criterion.max_score), std

    def feature_importance(self, top: int = 15) -> pd.DataFrame:
        rows: Dict[str, float] = {}
        for m in self.models:
            est = m.named_steps["m"]
            if hasattr(est, "feature_importances_"):
                imp = est.feature_importances_
            elif hasattr(est, "coef_"):
                imp = np.abs(est.coef_)
            else:
                continue
            for name, v in zip(self.feature_names, imp):
                rows[name] = rows.get(name, 0.0) + float(v)
        if not rows:
            return pd.DataFrame(columns=["feature", "importance"])
        s = pd.Series(rows).sort_values(ascending=False) / max(1, len(self.models))
        return s.head(top).rename("importance").reset_index().rename(
            columns={"index": "feature"}
        )


# --------------------------------------------------------------------------- #
class RubricScorer:
    """Tập hợp các TraitModel theo rubric + quy tắc tổng hợp điểm."""

    def __init__(self, rubric: Rubric, cfg: Optional[TraitModelConfig] = None):
        self.rubric = rubric
        self.cfg = cfg or TraitModelConfig()
        self.traits: Dict[str, TraitModel] = {}
        self.encoder_name: str = ""
        self.meta: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "RubricScorer":
        """``y`` là DataFrame có cột trùng key tiêu chí (điểm vàng từng tiêu chí)."""
        for c in self.rubric.criteria:
            col = c.key if c.key in y.columns else f"gold_{c.key}"
            if col not in y.columns:
                continue
            mask = y[col].notna()
            if mask.sum() < 20:
                import warnings
                warnings.warn(
                    f"Tiêu chí '{c.key}' chỉ có {int(mask.sum())} nhãn - "
                    "mô hình sẽ kém tin cậy (khuyến nghị ≥ 150)."
                )
            if mask.sum() < 5:
                continue
            tm = TraitModel(c, self.cfg).fit(X[mask.values], y.loc[mask, col].values)
            self.traits[c.key] = tm
        return self

    def predict_traits(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        scores, stds = {}, {}
        for k, tm in self.traits.items():
            m, s = tm.predict(X)
            scores[k], stds[k] = m, s
        return pd.DataFrame(scores, index=X.index), pd.DataFrame(stds, index=X.index)

    def aggregate(self, trait_scores: pd.DataFrame) -> np.ndarray:
        """Quy đổi điểm thành phần về thang điểm tổng theo trọng số rubric."""
        total = np.zeros(len(trait_scores))
        for c in self.rubric.criteria:
            if c.key not in trait_scores.columns:
                continue
            total += (trait_scores[c.key].values / c.max_score) * c.weight * self.rubric.scale_max
        return total

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(
                {"rubric_id": self.rubric.rubric_id, "cfg": self.cfg,
                 "traits": self.traits, "encoder_name": self.encoder_name,
                 "meta": self.meta, "rubric": self.rubric},
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "RubricScorer":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(d["rubric"], d["cfg"])
        obj.traits = d["traits"]
        obj.encoder_name = d.get("encoder_name", "")
        obj.meta = d.get("meta", {})
        return obj

    def report(self) -> str:
        L = [f"Mô hình chấm theo rubric '{self.rubric.name}'", ""]
        for k, tm in self.traits.items():
            L.append(f"— {tm.criterion.name} ({k}): "
                     f"n={tm.train_stats.get('n_train', 0):.0f}, "
                     f"OOF MAE={tm.train_stats.get('oof_mae', float('nan')):.3f}")
            fi = tm.feature_importance(8)
            if len(fi):
                L.append("    Đặc trưng quan trọng: " +
                         ", ".join(f"{r.feature}({r.importance:.3f})" for r in fi.itertuples()))
        return "\n".join(L)

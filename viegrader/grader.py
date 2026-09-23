"""Lớp điều phối cấp cao: huấn luyện và chấm điểm đầu-cuối.

    from viegrader import Grader, load_rubric
    g = Grader(load_rubric("rubrics/bai_kiem_tra_mon_hoc.yaml"))
    g.fit(df_train, gold_df)          # df_train có cột text; gold có điểm từng tiêu chí
    results = g.score(df_test)        # -> list[ScoreResult]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .cleaning.dedup import prompt_copy_ratio
from .cleaning.normalize import clean_text
from .config import thresholds
from .features.extractor import extract_all
from .features.semantic import get_encoder, semantic_features
from .models.ensemble import EnsembleConfig, ScoreEnsemble, level_name, round_total, snap_to_levels
from .models.llm_judge import LLMConfig
from .backends.base import ScoringBackend
from .backends.remote_llm import RemoteLLMBackend
from .models.trait_model import RubricScorer, TraitModelConfig
from .schema import CriterionScore, Rubric, ScoreResult
from .scoring.feedback import build_feedback
from .scoring.rules import apply_rules


@dataclass
class GraderConfig:
    encoder: str = "auto"                 # auto | phobert | tfidf | none
    use_llm: bool = False
    llm: LLMConfig = field(default_factory=LLMConfig)
    trait: TraitModelConfig = field(default_factory=TraitModelConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    semantic_dim: int = 48                # số chiều vector ngữ nghĩa ghép vào đặc trưng
    clean_input: bool = True
    review_confidence: float = 0.55
    n_anchors_for_llm: int = 3


class Grader:
    def __init__(self, rubric: Rubric, cfg: Optional[GraderConfig] = None):
        self.rubric = rubric
        self.cfg = cfg or GraderConfig()
        self.scorer = RubricScorer(rubric, self.cfg.trait)
        self.ensemble = ScoreEnsemble(rubric, self.cfg.ensemble)
        self.encoder = None
        self._svd = None
        self.anchors: List[Dict[str, Any]] = []
        self.keywords: Dict[str, List[str]] = {}     # {prompt_id: [ý cốt lõi]}
        self.fitted = False
        # Backend có client mạng được khởi tạo lười và không đưa vào pickle.
        self._remote_llm_backend: Optional[RemoteLLMBackend] = None
        # Backend cục bộ/tùy biến không được pickle; đăng ký lại khi triển khai.
        self.backends: Dict[str, ScoringBackend] = {}

    # ------------------------------------------------------------------ #
    # Chuẩn bị đặc trưng
    # ------------------------------------------------------------------ #
    def _prepare_texts(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if "text" not in d.columns:
            raise ValueError("DataFrame cần cột 'text'.")
        d["text"] = d["text"].fillna("").astype(str)
        for c in ("prompt_text", "prompt_id"):
            if c not in d.columns:
                d[c] = ""
            d[c] = d[c].fillna("").astype(str)
        if self.cfg.clean_input:
            d["text"] = [clean_text(t)[0] for t in d["text"]]
        return d

    def _semantic_matrix(self, d: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        if self.cfg.encoder == "none":
            return pd.DataFrame(index=d.index)
        if self.encoder is None:
            self.encoder = get_encoder(self.cfg.encoder)
        texts = list(d["text"])
        prompts = list(d["prompt_text"])
        E = self.encoder.encode(texts)
        P = self.encoder.encode(prompts) if any(p.strip() for p in prompts) else None
        ref = None
        if self.anchors:
            ref = self.encoder.encode([a["text"] for a in self.anchors])
        sem = pd.DataFrame(semantic_features(E, P, ref), index=d.index)

        # Giảm chiều vector bài để ghép vào bảng đặc trưng
        k = min(self.cfg.semantic_dim, max(2, E.shape[1]), max(2, E.shape[0] - 1))
        if fit or self._svd is None:
            from sklearn.decomposition import TruncatedSVD
            self._svd = TruncatedSVD(n_components=k, random_state=42).fit(E)
        emb = self._svd.transform(E)
        emb_df = pd.DataFrame(
            emb, index=d.index, columns=[f"emb_{i}" for i in range(emb.shape[1])]
        )
        return pd.concat([sem, emb_df], axis=1)

    def build_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        d = self._prepare_texts(df)
        rows: List[Dict[str, float]] = []
        for _, r in d.iterrows():
            kws = self.keywords.get(str(r.get("prompt_id", "")), [])
            if not kws and isinstance(r.get("keywords"), str) and r["keywords"]:
                kws = [x.strip() for x in str(r["keywords"]).split(";") if x.strip()]
            f = extract_all(r["text"], r["prompt_text"], keywords=kws)
            f["prompt_copy_ratio"] = prompt_copy_ratio(r["text"], r["prompt_text"])
            for extra in ("dup_ratio", "teencode_rate"):
                if extra in d.columns and pd.notna(r.get(extra)):
                    f[extra] = float(r[extra])
                else:
                    f.setdefault(extra, 0.0)
            rows.append(f)
        X = pd.DataFrame(rows, index=d.index)
        sem = self._semantic_matrix(d, fit=fit)
        if len(sem.columns):
            X = pd.concat([X, sem], axis=1)
        else:
            X["sim_prompt"] = 0.5
            X["sim_reference"] = 0.0
        return X

    # ------------------------------------------------------------------ #
    # Huấn luyện
    # ------------------------------------------------------------------ #
    def set_keywords(self, mapping: Dict[str, Sequence[str]]) -> None:
        """Khai báo ý cốt lõi (đáp án) cho từng đề bài."""
        self.keywords = {str(k): list(v) for k, v in mapping.items()}

    def set_anchors(self, anchors: Sequence[Dict[str, Any]]) -> None:
        """Bài mẫu neo: [{'text':..., 'total':..., 'scores':{...}}, ...]"""
        self.anchors = list(anchors)[: max(1, self.cfg.n_anchors_for_llm)]

    def add_backend(self, backend: ScoringBackend, name: Optional[str] = None) -> None:
        """Đăng ký Vistral hoặc backend tùy biến vào ensemble hiện tại."""
        backend_name = name or backend.name
        if backend_name in {"feature", "llm"}:
            raise ValueError("Tên backend 'feature' và 'llm' được dành riêng.")
        self.backends[backend_name] = backend

    def fit(
        self,
        df: pd.DataFrame,
        gold: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        val_gold: Optional[pd.DataFrame] = None,
    ) -> "Grader":
        """``gold`` cần các cột trùng key tiêu chí (hoặc 'gold_<key>')."""
        X = self.build_features(df, fit=True)
        self.scorer.fit(X, gold.reset_index(drop=True))
        for backend in self.backends.values():
            backend.fit(self._prepare_texts(df), gold.reset_index(drop=True))

        # Học trọng số tổng hợp trên tập kiểm định (nếu có và có LLM)
        if val_df is not None and val_gold is not None:
            Xv = self.build_features(val_df, fit=False)
            f_pred, _ = self.scorer.predict_traits(Xv)
            sources = {"feature": f_pred}
            if self.cfg.use_llm:
                llm_pred = self._llm_traits(val_df)
                if llm_pred is not None:
                    sources["llm"] = llm_pred
            for name, backend in self.backends.items():
                if backend.available:
                    sources[name] = backend.predict(self._prepare_texts(val_df)).scores
            self.ensemble.fit(sources, val_gold.reset_index(drop=True))

        self.fitted = True
        return self

    # ------------------------------------------------------------------ #
    def _llm_traits(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self._remote_llm_backend is None:
            self._remote_llm_backend = RemoteLLMBackend(self.rubric, self.cfg.llm)
        backend = self._remote_llm_backend
        backend.keywords = self.keywords
        backend.anchors = self.anchors
        if not backend.available:
            return None
        prediction = backend.predict(self._prepare_texts(df))
        out = prediction.scores
        out.attrs["comments"] = prediction.comments
        out.attrs["flags"] = prediction.flags
        out.attrs["confidence"] = prediction.confidence
        out.attrs["evidence"] = prediction.evidence
        return out

    # ------------------------------------------------------------------ #
    # Chấm điểm
    # ------------------------------------------------------------------ #
    def score(self, df: pd.DataFrame, with_feedback: bool = True) -> List[ScoreResult]:
        if not self.fitted and not self.cfg.use_llm and not self.backends:
            raise RuntimeError(
                "Grader chưa được huấn luyện. Hãy gọi fit(), nạp mô hình đã lưu, "
                "hoặc bật cfg.use_llm=True để chấm thuần bằng LLM."
            )
        d = self._prepare_texts(df)
        X = self.build_features(df, fit=False)

        sources: Dict[str, pd.DataFrame] = {}
        stds = pd.DataFrame(index=X.index)
        if self.fitted and self.scorer.traits:
            f_pred, stds = self.scorer.predict_traits(X)
            sources["feature"] = f_pred

        llm_comments: List[str] = [""] * len(d)
        llm_flags: List[List[str]] = [[] for _ in range(len(d))]
        if self.cfg.use_llm:
            llm_pred = self._llm_traits(df)
            if llm_pred is not None and len(llm_pred.columns):
                sources["llm"] = llm_pred
                llm_comments = llm_pred.attrs.get("comments", llm_comments)
                llm_flags = llm_pred.attrs.get("flags", llm_flags)

        for name, backend in self.backends.items():
            if not backend.available:
                continue
            prediction = backend.predict(d)
            sources[name] = prediction.scores
            for i in range(len(d)):
                if prediction.comments[i]:
                    llm_comments[i] = "\n".join(
                        x for x in (llm_comments[i], prediction.comments[i]) if x
                    )
                llm_flags[i].extend(prediction.flags[i])

        if not sources:
            raise RuntimeError("Không có nguồn điểm nào khả dụng.")

        raw = self.ensemble.predict(sources) if len(sources) > 1 else list(sources.values())[0]
        snapped = snap_to_levels(raw, self.rubric)

        th = thresholds(self.rubric)
        results: List[ScoreResult] = []
        for i, (idx, row) in enumerate(d.iterrows()):
            feats = X.loc[idx].to_dict()
            crit_scores: List[CriterionScore] = []
            snap_gaps: List[float] = []
            for c in self.rubric.criteria:
                if c.key not in snapped.columns:
                    continue
                s = float(snapped.loc[idx, c.key])
                rawv = float(raw.loc[idx, c.key])
                sd = float(stds.loc[idx, c.key]) if c.key in stds.columns else 0.0
                conf, gap = self._confidence(c, rawv, s, sd)
                snap_gaps.append(gap)
                crit_scores.append(CriterionScore(
                    key=c.key, name=c.name, raw_score=round(rawv, 4), score=s,
                    level=level_name(c, s), weight=c.weight, confidence=round(conf, 3),
                ))

            total_raw = sum(
                (cs.score / self.rubric.get(cs.key).max_score) * cs.weight * self.rubric.scale_max
                for cs in crit_scores
            )
            conf_all = float(np.mean([cs.confidence for cs in crit_scores])) if crit_scores else 0.5
            if "llm" in sources and len(sources) > 1:
                conf_all = min(1.0, conf_all + 0.05)

            res = ScoreResult(
                essay_id=str(row.get("essay_id", idx)),
                rubric_id=self.rubric.rubric_id,
                total=round_total(total_raw, self.rubric),
                total_before_rules=round(total_raw, 4),
                criteria=crit_scores,
                confidence=round(conf_all, 3),
                flags=[f for f in llm_flags[i] if f],
                meta={"sources": list(sources.keys())},
            )

            env = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in feats.items()}
            env.update({
                "min_words": th["min_words"],
                "max_words": th["max_words"],
                # Độ "chông chênh" của điểm: 0 = rơi đúng tâm mức, 1 = nằm giữa hai mức
                "snap_gap": float(np.mean(snap_gaps)) if snap_gaps else 0.0,
                "max_snap_gap": float(np.max(snap_gaps)) if snap_gaps else 0.0,
                "pred_std": float(np.mean([
                    stds.loc[idx, c.key] for c in self.rubric.criteria
                    if c.key in stds.columns
                ])) if len(stds.columns) else 0.0,
            })
            res, _ = apply_rules(res, self.rubric, env)
            res.total = round_total(res.total, self.rubric)

            if with_feedback:
                kws = self.keywords.get(str(row.get("prompt_id", "")), [])
                res.feedback = build_feedback(
                    res, self.rubric, row["text"], feats, kws, llm_comments[i]
                )
            results.append(res)
        return results

    # ------------------------------------------------------------------ #
    @staticmethod
    def _confidence(criterion, raw: float, snapped: float, sd: float) -> tuple:
        """Độ tin cậy của điểm một tiêu chí.

        Kết hợp hai nguồn bất định:
          - ``sd``  : phân tán giữa các mô hình bootstrap (bất định của mô hình)
          - ``gap`` : khoảng cách từ điểm thô tới mức rubric gần nhất, chuẩn hoá
                      theo nửa khoảng cách giữa hai mức. gap → 1 nghĩa là bài nằm
                      đúng ranh giới hai mức, việc neo về mức nào là tuỳ tiện.
        Đây chính là loại bài cần giám khảo xem lại, dù mô hình rất "chắc".
        """
        levels = criterion.level_scores()
        if len(levels) >= 2:
            half = min(abs(b - a) for a, b in zip(levels, levels[1:])) / 2.0
        else:
            half = max(criterion.max_score * 0.25, 1e-9)
        gap = float(np.clip(abs(raw - snapped) / max(half, 1e-9), 0.0, 1.0))
        model_conf = float(np.clip(1.0 - sd / max(half, 1e-9), 0.0, 1.0))
        conf = float(np.clip(0.5 * model_conf + 0.5 * (1.0 - gap), 0.0, 1.0))
        return conf, gap

    def score_to_frame(self, df: pd.DataFrame, with_feedback: bool = True) -> pd.DataFrame:
        res = self.score(df, with_feedback)
        rows = []
        for r in res:
            row: Dict[str, Any] = {
                "essay_id": r.essay_id, "total": r.total,
                "total_before_rules": r.total_before_rules,
                "confidence": r.confidence,
                "needs_human_review": r.needs_human_review,
                "review_reason": r.review_reason,
                "flags": "|".join(r.flags),
                "applied_rules": "|".join(r.applied_rules),
            }
            for cs in r.criteria:
                row[cs.key] = cs.score
                row[f"{cs.key}__level"] = cs.level
            if with_feedback:
                row["feedback"] = r.feedback
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        import pickle
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({
                "rubric": self.rubric, "cfg": self.cfg, "scorer": self.scorer,
                "ensemble": self.ensemble, "svd": self._svd, "encoder": self.encoder,
                "anchors": self.anchors, "keywords": self.keywords, "fitted": self.fitted,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "Grader":
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        g = cls(d["rubric"], d["cfg"])
        g.scorer, g.ensemble = d["scorer"], d["ensemble"]
        g._svd, g.encoder = d["svd"], d["encoder"]
        g.anchors, g.keywords, g.fitted = d["anchors"], d["keywords"], d["fitted"]
        g._remote_llm_backend = None
        g.backends = {}
        return g

    def close(self) -> None:
        """Giải phóng tài nguyên backend; hữu ích khi chạy tuần tự ablation A–H."""
        if self._remote_llm_backend is not None:
            self._remote_llm_backend.close()
            self._remote_llm_backend = None
        for backend in self.backends.values():
            backend.close()

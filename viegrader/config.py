"""Nạp rubric và cấu hình hệ thống."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .schema import Criterion, Level, Rubric

PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent
DEFAULT_RUBRIC_DIR = PROJECT_DIR / "rubrics"


def load_rubric(path: str | os.PathLike) -> Rubric:
    """Nạp rubric từ file YAML/JSON và kiểm tra tính hợp lệ."""
    p = Path(path)
    if not p.exists():
        cand = DEFAULT_RUBRIC_DIR / p.name
        if cand.exists():
            p = cand
        else:
            raise FileNotFoundError(f"Không tìm thấy rubric: {path}")

    with open(p, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    criteria: List[Criterion] = []
    for c in raw.get("criteria", []):
        levels = [
            Level(
                name=lv["name"],
                score=float(lv["score"]),
                descriptor=lv.get("descriptor", ""),
                indicators=list(lv.get("indicators", []) or []),
            )
            for lv in c.get("levels", [])
        ]
        criteria.append(
            Criterion(
                key=c["key"],
                name=c.get("name", c["key"]),
                weight=float(c["weight"]),
                max_score=float(c["max_score"]),
                description=(c.get("description") or "").strip(),
                levels=levels,
                feature_hints=list(c.get("feature_hints", []) or []),
            )
        )

    meta = dict(raw.get("meta", {}) or {})
    meta["thresholds"] = dict(raw.get("thresholds", {}) or {})

    rubric = Rubric(
        rubric_id=raw.get("rubric_id", p.stem),
        name=raw.get("name", p.stem),
        scale_max=float(raw.get("scale_max", 10.0)),
        scale_step=float(raw.get("scale_step", 0.25)),
        criteria=criteria,
        rules=list(raw.get("rules", []) or []),
        meta=meta,
    )

    warns = rubric.validate()
    if warns:
        import warnings

        warnings.warn("Rubric có cảnh báo:\n - " + "\n - ".join(warns))
    return rubric


def default_rubric_path() -> Optional[Path]:
    for name in ("bai_kiem_tra_mon_hoc.yaml",):
        p = DEFAULT_RUBRIC_DIR / name
        if p.exists():
            return p
    return None


def thresholds(rubric: Rubric) -> Dict[str, float]:
    """Ngưỡng cấu hình, có giá trị mặc định an toàn."""
    t = {
        "min_words": 300.0,
        "max_words": 1500.0,
        "review_rate_target": 0.15,
    }
    t.update({k: float(v) for k, v in (rubric.meta.get("thresholds") or {}).items()})
    return t

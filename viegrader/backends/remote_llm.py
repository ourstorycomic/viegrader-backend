from __future__ import annotations

from typing import Any, Dict, Sequence

import pandas as pd

from .base import BackendPrediction, ScoringBackend
from ..models.llm_judge import LLMConfig, LLMJudge
from ..schema import Rubric


class RemoteLLMBackend(ScoringBackend):
    name = "remote_llm"

    def __init__(self, rubric: Rubric, cfg: LLMConfig | None = None):
        self.rubric = rubric
        self.judge = LLMJudge(rubric, cfg)
        self.keywords: Dict[str, Sequence[str]] = {}
        self.anchors: list[Dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return self.judge.available

    def predict(self, essays: pd.DataFrame) -> BackendPrediction:
        rows, confidence, comments, flags, evidence = [], [], [], [], []
        for _, row in essays.iterrows():
            pid = str(row.get("prompt_id", ""))
            result = self.judge.score(
                str(row.get("text", "")), str(row.get("prompt_text", "")),
                "; ".join(self.keywords.get(pid, [])), self.anchors,
            )
            if not result:
                rows.append({}); confidence.append(0.0); comments.append("")
                flags.append(["LLM_FAILED"]); evidence.append({})
                continue
            criteria = result.get("criteria", {})
            rows.append({k: v.get("score") for k, v in criteria.items()})
            evidence.append({k: list(v.get("evidence", [])) for k, v in criteria.items()})
            confidence.append(float(result.get("confidence", 0.5)))
            comments.append(str(result.get("overall_comment", "")))
            flags.append(list(result.get("flags", [])))
        return BackendPrediction(
            pd.DataFrame(rows, index=essays.index), confidence, comments, flags,
            evidence, {"backend": self.name},
        )

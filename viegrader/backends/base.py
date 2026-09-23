from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import pandas as pd


@dataclass
class BackendPrediction:
    """Đầu ra chuẩn của mọi nguồn chấm điểm."""

    scores: pd.DataFrame
    confidence: List[float] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    flags: List[List[str]] = field(default_factory=list)
    evidence: List[Dict[str, List[str]]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.scores)
        self.confidence = self.confidence or [0.5] * n
        self.comments = self.comments or [""] * n
        self.flags = self.flags or [[] for _ in range(n)]
        self.evidence = self.evidence or [{} for _ in range(n)]


class ScoringBackend(ABC):
    """Hợp đồng dùng chung cho baseline, API LLM và LLM cục bộ."""

    name = "backend"

    @property
    @abstractmethod
    def available(self) -> bool: ...

    def fit(self, essays: pd.DataFrame, gold: pd.DataFrame) -> "ScoringBackend":
        return self

    @abstractmethod
    def predict(self, essays: pd.DataFrame) -> BackendPrediction: ...

    def close(self) -> None:
        return None

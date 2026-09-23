"""Các backend chấm điểm có cùng hợp đồng đầu ra."""

from .base import BackendPrediction, ScoringBackend
from .remote_llm import RemoteLLMBackend
from .vistral import VistralBackend, VistralConfig

__all__ = [
    "BackendPrediction", "ScoringBackend", "RemoteLLMBackend",
    "VistralBackend", "VistralConfig",
]

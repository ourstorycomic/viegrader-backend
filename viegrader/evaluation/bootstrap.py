from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np


def bootstrap_ci(
    y_true: Sequence[float], y_pred: Sequence[float], metric: Callable,
    n_boot: int = 1000, confidence: float = 0.95, seed: int = 42,
) -> Dict[str, float]:
    a, b = np.asarray(y_true), np.asarray(y_pred)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Cần hai dãy cùng độ dài và ít nhất 2 quan sát.")
    rng, values = np.random.default_rng(seed), []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        try:
            value = float(metric(a[idx], b[idx]))
            if np.isfinite(value):
                values.append(value)
        except (ValueError, ZeroDivisionError):
            continue
    if not values:
        raise ValueError("Không tính được bootstrap CI.")
    alpha = (1 - confidence) / 2
    return {
        "estimate": float(metric(a, b)),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1 - alpha)),
        "n_boot_valid": float(len(values)),
    }

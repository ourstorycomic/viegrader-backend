from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, Iterable

import numpy as np
import pandas as pd

from .ablation import AblationConfig
from ..evaluation.metrics import score_metrics


class AblationRunner:
    """Chạy A–H trên cùng test set qua factory do notebook/ứng dụng cung cấp."""

    def __init__(self, scorer_factory: Callable[[AblationConfig], object], output_dir="reports/ablation"):
        self.scorer_factory = scorer_factory
        self.output_dir = Path(output_dir)

    def run(self, configs: Iterable[AblationConfig], test_df: pd.DataFrame,
            gold: pd.DataFrame, gold_total_col="gold_total") -> pd.DataFrame:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for cfg in configs:
            random.seed(cfg.seed); np.random.seed(cfg.seed)
            scorer = self.scorer_factory(cfg)
            pred = scorer.score_to_frame(test_df) if hasattr(scorer, "score_to_frame") else scorer(test_df)
            metrics = score_metrics(gold[gold_total_col], pred["total"])
            rows.append({"experiment_id": cfg.experiment_id, **cfg.to_dict(), **metrics})
            pred.to_csv(self.output_dir / f"pred_{cfg.experiment_id}.csv", index=False)
            (self.output_dir / f"config_{cfg.experiment_id}.json").write_text(
                json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if hasattr(scorer, "close"):
                scorer.close()
        result = pd.DataFrame(rows).sort_values("experiment_id")
        result.to_csv(self.output_dir / "ablation_summary.csv", index=False)
        return result

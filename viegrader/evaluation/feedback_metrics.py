from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def bertscore_feedback(
    generated: Sequence[str], references: Sequence[str],
    model_type: str = "bert-base-multilingual-cased", batch_size: int = 8,
) -> Dict[str, float]:
    if len(generated) != len(references):
        raise ValueError("generated và references phải có cùng số phần tử.")
    if not generated:
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    try:
        from bert_score import score
    except ImportError as exc:
        raise ImportError("Cài viegrader[evaluation] để tính BERTScore.") from exc
    p, r, f1 = score(
        list(generated), list(references), model_type=model_type,
        batch_size=batch_size, verbose=False,
    )
    return {
        "bertscore_precision": float(np.mean(p.cpu().numpy())),
        "bertscore_recall": float(np.mean(r.cpu().numpy())),
        "bertscore_f1": float(np.mean(f1.cpu().numpy())),
    }

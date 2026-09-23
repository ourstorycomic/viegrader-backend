from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd

from ..models.llm_judge import SYSTEM_PROMPT, build_user_prompt
from ..schema import Rubric


def build_instruction_records(
    essays: pd.DataFrame, gold: pd.DataFrame, rubric: Rubric,
    feedback_col: str = "teacher_feedback",
) -> List[Dict[str, str]]:
    """Chuyển nhãn vàng thành hội thoại SFT, không đưa định danh sinh viên vào model."""
    if len(essays) != len(gold):
        raise ValueError("essays và gold phải có cùng số dòng.")
    records = []
    for (_, essay), (_, label) in zip(essays.iterrows(), gold.iterrows()):
        criteria = {}
        for criterion in rubric.criteria:
            key = criterion.key if criterion.key in label else f"gold_{criterion.key}"
            if key in label and pd.notna(label[key]):
                criteria[criterion.key] = {"score": float(label[key]), "evidence": [], "comment": ""}
        target = {
            "criteria": criteria,
            "total": float(label.get("gold_total", label.get("total", 0.0))),
            "overall_comment": str(label.get(feedback_col, "")), "flags": [],
        }
        user = build_user_prompt(
            str(essay.get("text", "")), rubric, str(essay.get("prompt_text", "")),
            str(essay.get("rag_context", essay.get("answer_key", ""))), None,
        )
        records.append({
            "system": SYSTEM_PROMPT, "user": user,
            "assistant": json.dumps(target, ensure_ascii=False),
        })
    return records

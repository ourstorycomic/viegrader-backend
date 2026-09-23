import json
from pathlib import Path

import pandas as pd

from viegrader.cli import build_parser
from viegrader.config import load_rubric
from viegrader.discrete_math import (
    EXAMS,
    build_total_instruction_records,
    infer_exam,
)


ROOT = Path(__file__).resolve().parents[1]


def test_all_discrete_math_rubrics_are_valid():
    for exam in EXAMS:
        rubric = load_rubric(ROOT / "data/toan_roi_rac/rubrics" / f"rubric_de_{exam[-1]}.yaml")
        assert rubric.validate() == []
        assert len(rubric.criteria) == 10
        assert abs(sum(c.weight for c in rubric.criteria) - 1.0) < 1e-9


def test_infer_exam_uses_distinctive_content():
    exam, hits, _ = infer_exam("Chọn 4 tầng liền nhau; 2 bít đầu là số 1; chia hết cho 2 hoặc 3 hoặc hoặc 7")
    assert exam == "De_4"
    assert hits == 3


def test_total_instruction_records_do_not_invent_trait_labels(tmp_path):
    essays = pd.DataFrame([{
        "essay_id": "DM-x", "text": "Bài làm", "exam_id": "De_1",
        "prompt_text": "Đề", "answer_key": "cau_1a: 4536", "split": "train",
    }])
    gold = pd.DataFrame([{
        "essay_id": "DM-x", "gold_total": 8.0, "teacher_feedback": "Khá",
        "split": "train",
    }])
    ep, gp, op = tmp_path / "e.csv", tmp_path / "g.csv", tmp_path / "out.jsonl"
    essays.to_csv(ep, index=False)
    gold.to_csv(gp, index=False)
    records = build_total_instruction_records(ep, gp, op)
    target = json.loads(records[0]["assistant"])
    assert target["total"] == 8.0
    assert target["label_granularity"] == "total_only"
    assert "criteria" not in target


def test_cli_exposes_discrete_math_commands():
    parser = build_parser()
    args = parser.parse_args(["dm-build-qlora", "-i", "e.csv", "-g", "g.csv"])
    assert args.cmd == "dm-build-qlora"


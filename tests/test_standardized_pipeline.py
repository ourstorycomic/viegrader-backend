import json
from pathlib import Path

import pandas as pd

from viegrader.cli import build_parser
from viegrader.standardized_pipeline import (
    prepare_standardized_dataset,
    score_rubric_items,
    split_answer_sections,
    validate_rubrics,
)


ROOT = Path(__file__).resolve().parents[1]
RUBRICS = ROOT / "data/toan_roi_rac/rubrics"


def test_cli_exposes_it04_pipeline():
    parser = build_parser()
    args = parser.parse_args(["it04-run-all", "--source", "dataset.jsonl"])
    assert args.cmd == "it04-run-all"


def test_split_and_exact_rubric_score():
    text = "Câu 1a: Kết quả 4536.\nCâu 1b: Kết quả 4464."
    sections = split_answer_sections(text)
    assert "cau_1a" in sections and "cau_1b" in sections
    items, meta = score_rubric_items(text, RUBRICS / "rubric_de_1.yaml")
    indexed = {item.criterion: item for item in items}
    assert indexed["cau_1a"].score == 1.0
    assert indexed["cau_1b"].score == 1.0
    assert indexed["cau_1a"].location == "cau_1a"
    assert meta["rubric_total"] >= 2.0


def test_prepare_keeps_locked_split_and_excludes_not_ready(tmp_path):
    source = tmp_path / "dataset.jsonl"
    rows = [
        {"essay_id": "A", "exam_code": "DE01", "answer_text": "Câu 1a: đáp án 4536 và giải thích",
         "split": "train", "training_eligible": True, "viegrader_ready": True,
         "labels": {"score_10": 8.0}},
        {"essay_id": "B", "exam_code": "DE02", "answer_text": "Câu 1a: bài làm đủ dài để kiểm tra",
         "split": "test", "training_eligible": False, "viegrader_ready": True,
         "labels": {"score_10": 6.0}},
        {"essay_id": "C", "exam_code": "UNKNOWN", "answer_text": "Văn bản này đủ dài nhưng không rõ đề",
         "split": "excluded_conflict", "training_eligible": False, "viegrader_ready": False,
         "labels": {"score_10": 4.0}},
    ]
    source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    report = prepare_standardized_dataset(source, RUBRICS, tmp_path / "out")
    essays = pd.read_csv(tmp_path / "out/essays_split.csv")
    assert report["n_ready"] == 2
    assert dict(zip(essays.essay_id, essays.split)) == {"A": "train", "B": "test"}
    assert essays.loc[essays.essay_id.eq("A"), "training_eligible"].iloc[0]


def test_rubric_inventory_has_five_exams_and_fifty_items(tmp_path):
    report = validate_rubrics(RUBRICS, tmp_path)
    assert report["valid"] is True
    assert report["n_rubrics"] == 5
    assert report["n_items"] == 50

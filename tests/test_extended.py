import json

import numpy as np
import pandas as pd
import pytest

from viegrader.artifacts import ArtifactManifest
from viegrader.backends.base import BackendPrediction, ScoringBackend
from viegrader.config import load_rubric
from viegrader.evaluation.bootstrap import bootstrap_ci
from viegrader.experiments import default_ablation_configs
from viegrader.integrations.moodle import MoodleClient, MoodleConfig
from viegrader.labeling.agreement import quadratic_weighted_kappa
from viegrader.qlora.dataset import build_instruction_records
from viegrader.datasets import prepare_vietnamese_it_dataset, split_research_dataset
from viegrader.rag import DocumentChunk, TfidfRAGIndex, attach_rag_context
from viegrader.usability import score_sus, summarize_sus


@pytest.fixture
def rubric():
    return load_rubric("rubrics/bai_kiem_tra_mon_hoc.yaml")


def test_rag_retrieval_and_audit():
    index = TfidfRAGIndex().fit([
        DocumentChunk("c1", "Mã hóa AES sử dụng khóa đối xứng", "doc1", prompt_id="P1"),
        DocumentChunk("c2", "Tường lửa kiểm soát lưu lượng mạng", "doc2", prompt_id="P2"),
    ])
    hits = index.search("trình bày mã hóa đối xứng AES", prompt_id="P1")
    assert hits[0]["chunk_id"] == "c1"
    enriched, audit = attach_rag_context(pd.DataFrame([{
        "essay_id": "E1", "prompt_id": "P1", "text": "AES", "prompt_text": "Mã hóa"
    }]), index)
    assert "[c1]" in enriched.loc[0, "rag_context"]
    assert audit.loc[0, "chunk_id"] == "c1"


def test_qlora_records_are_json(rubric):
    essays = pd.DataFrame([{"text": "Bài làm", "prompt_text": "Câu hỏi"}])
    scores = {c.key: c.level_scores()[0] for c in rubric.criteria}
    gold = pd.DataFrame([{**scores, "gold_total": 0.0, "teacher_feedback": "Cần bổ sung"}])
    records = build_instruction_records(essays, gold, rubric)
    target = json.loads(records[0]["assistant"])
    assert set(target["criteria"]) == {c.key for c in rubric.criteria}


def test_ablation_has_a_to_h():
    configs = default_ablation_configs()
    assert [c.experiment_id for c in configs] == list("ABCDEFGH")
    assert configs[0].encoder == "tfidf"
    assert configs[1].encoder == "phobert"
    assert configs[-1].use_qlora and configs[-1].use_rag


def test_bootstrap_ci_contains_estimate():
    y = np.array([0, 1, 2, 3, 4] * 5, dtype=float)
    result = bootstrap_ci(y, y, quadratic_weighted_kappa, n_boot=50)
    assert result["estimate"] == pytest.approx(1.0)
    assert result["lower"] <= result["estimate"] <= result["upper"]


def test_sus_scoring_and_summary():
    row = [5, 1, 5, 1, 5, 1, 5, 1, 5, 1]
    assert score_sus(row) == 100.0
    df = pd.DataFrame([row, row], columns=[f"sus_{i}" for i in range(1, 11)])
    result = summarize_sus(df)
    assert result["mean"] == 100.0


def test_moodle_requires_approval():
    client = MoodleClient(MoodleConfig("https://moodle.invalid", "secret"))
    with pytest.raises(PermissionError):
        client.save_grade(1, 2, 8.0, approved=False)


def test_artifact_manifest_roundtrip(tmp_path):
    path = tmp_path / "manifest.json"
    ArtifactManifest(model_type="qlora", base_model="vistral").save(path)
    manifest = ArtifactManifest.load(path)
    assert manifest.version == "0.5.1"
    assert manifest.model_type == "qlora"


def test_backend_contract():
    class DummyBackend(ScoringBackend):
        name = "dummy"

        @property
        def available(self):
            return True

        def predict(self, essays):
            return BackendPrediction(pd.DataFrame({"criterion": [1.0] * len(essays)}))

    output = DummyBackend().predict(pd.DataFrame({"text": ["a", "b"]}))
    assert len(output.scores) == 2
    assert output.confidence == [0.5, 0.5]


def test_prepare_external_dataset_does_not_invent_labels(tmp_path):
    source = tmp_path / "download"
    source.mkdir()
    pd.DataFrame({
        "id": [1, 2],
        "essay": ["Bài luận tiếng Việt thứ nhất.", "Bài luận tiếng Việt thứ hai."],
        "question": ["Phân tích vấn đề", "Phân tích vấn đề"],
    }).to_csv(source / "essays.csv", index=False)
    output = tmp_path / "prepared"
    report = prepare_vietnamese_it_dataset(source, output)
    assert report["labels_status"] == "human_labeling_required"
    assert (output / "essays_raw.csv").is_file()
    assert not (output / "labels_unverified.csv").exists()


def test_group_split_has_no_student_leakage(tmp_path):
    n = 60
    essays = pd.DataFrame({
        "essay_id": [f"E{i}" for i in range(n)],
        "text": ["Nội dung bài luận " + str(i) for i in range(n)],
        "student_hash": [f"S{i // 2}" for i in range(n)],
        "prompt_id": [f"P{i % 5}" for i in range(n)],
    })
    gold = pd.DataFrame({"essay_id": essays["essay_id"], "gold_total": np.arange(n) % 31})
    report = split_research_dataset(essays, gold, tmp_path / "splits", seed=42)
    assert report["group_leakage_count"] == 0
    manifest = pd.read_csv(tmp_path / "splits" / "split_manifest.csv")
    checked = essays.merge(manifest, on="essay_id").groupby("student_hash")["split"].nunique()
    assert checked.max() == 1

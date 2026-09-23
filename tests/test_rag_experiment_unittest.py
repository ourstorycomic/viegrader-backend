"""CPU-only integration check for paired inference data flow and report."""

import json
import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from viegrader.rag.experiment import prepare, report, run, run_zero_shot
from viegrader.rag.index import DocumentChunk, TfidfRAGIndex


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True):
        return [0] * (len(messages[1]["content"]) // 5)


class RagExperimentTest(unittest.TestCase):
    def test_paired_report_and_gold_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            essays = root / "essays.csv"
            gold = root / "gold.csv"
            index_path = root / "index.json"
            output = root / "run"
            pd.DataFrame([
                {"essay_id": f"E{i}", "text": "nguyên lý bù trừ và tổ hợp",
                 "prompt_text": "Trình bày nguyên lý bù trừ", "answer_key": "Đáp án",
                 "exam_id": "De_1", "split": "test"} for i in (1, 2)
            ]).to_csv(essays, index=False)
            pd.DataFrame({"essay_id": ["E1", "E2"], "gold_total": [5, 7],
                          "split": ["test", "test"]}).to_csv(gold, index=False)
            TfidfRAGIndex().fit([DocumentChunk("d1", "Nguyên lý bù trừ và tổ hợp. " * 5,
                                                 "book.pdf", course_id="IT04", page=3,
                                                 source_sha256="a" * 64)]).save_documents(index_path)
            transformers = types.SimpleNamespace(
                AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()))
            with patch.dict(sys.modules, {"transformers": transformers}):
                manifest = prepare(essays, index_path, output, "test-model", max_tokens=512,
                                   expected_test=2)
            self.assertEqual(manifest["n_paired_eligible"], 2)
            self.assertNotIn("gold_total", pd.read_csv(output / "input_rag.csv"))
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")

            def fake_score(input_path, output_path, **kwargs):
                inp = pd.read_csv(input_path)
                rows = []
                for entry in inp.itertuples():
                    rows.append({"essay_id": entry.essay_id, "exam_id": entry.exam_id,
                                 "parse_ok": True, "total": 6 if entry.variant == "plain" else
                                 (5 if entry.essay_id.endswith("E1") else 7)})
                result = pd.DataFrame(rows)
                result.to_csv(output_path, index=False)
                return result

            with patch("viegrader.rag.experiment.score_total_qlora", side_effect=fake_score):
                run(output, model="test-model", adapter=adapter, max_tokens=512)
            def fake_zero(input_path, output_path, **kwargs):
                self.assertIsNone(kwargs["adapter_path"])
                frame = pd.read_csv(input_path)
                pred = pd.DataFrame({"essay_id": frame.essay_id, "exam_id": frame.exam_id,
                                     "parse_ok": True, "total": [6.0, 6.0]})
                pred.to_csv(output_path, index=False)
                return pred

            with patch("viegrader.rag.experiment.score_total_qlora", side_effect=fake_zero):
                run_zero_shot(output, model="test-model", max_tokens=512)
            hybrid = root / "hybrid.csv"
            pd.DataFrame({"essay_id": ["E1", "E2"], "final_total": [5, 7],
                          "grading_mode": ["hybrid_rubric_model", "model_fallback"]}).to_csv(hybrid, index=False)
            baseline = root / "baseline.csv"
            pd.DataFrame({"essay_id": ["E1", "E2"], "total": [6, 6]}).to_csv(baseline, index=False)
            metrics = report(output, gold, baseline=baseline, hybrid=hybrid, bootstrap=100)
            self.assertEqual(metrics["n_common_valid"], 2)
            self.assertAlmostEqual(metrics["plain"]["MAE"], 1.0)
            self.assertAlmostEqual(metrics["rag"]["MAE"], 0.0)
            self.assertAlmostEqual(metrics["delta_MAE_rag_minus_plain"], -1.0)
            self.assertEqual(metrics["n_common_three_qlora_plain_rag_zero_shot"], 2)
            self.assertEqual(metrics["hybrid_route_counts"]["hybrid_rubric_model"], 1)
            self.assertIn("tfidf_ridge", metrics)
            self.assertTrue((output / "retrieval_audit.csv").is_file())
            self.assertIn("QWK_41", json.loads((output / "report.json").read_text())["rag"])


if __name__ == "__main__":
    unittest.main()

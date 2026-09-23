import json
import zipfile

import pandas as pd

from viegrader.reporting import ReportBuilder, build_report_bundle, detect_stage


def test_detect_report_stages():
    assert detect_stage("clean_report.json") == "data"
    assert detect_stage("agreement.csv") == "labeling"
    assert detect_stage("baseline_tfidf.csv") == "baseline"
    assert detect_stage("training_config_qlora.json") == "qlora"
    assert detect_stage("rag_context_audit.csv") == "rag"
    assert detect_stage("ablation_summary.csv") == "ablation"
    assert detect_stage("sus_report.json") == "sus"


def test_report_builder_ingests_tables_and_json(tmp_path):
    clean = tmp_path / "clean_report.json"
    clean.write_text(json.dumps({"input": 100, "kept": 92}), encoding="utf-8")
    metrics = tmp_path / "evaluation_metrics.csv"
    pd.DataFrame([{"QWK": 0.81, "MAE": 0.42}]).to_csv(metrics, index=False)
    builder = ReportBuilder("NCKH 2026", "Nhóm nghiên cứu").ingest([clean, metrics])
    summary = builder.summary_frame()
    assert set(summary[summary.status == "Đã có kết quả"].stage) == {"data", "evaluation"}
    assert len(builder.manifest()["stages"]) == 2


def test_export_all_report_formats(tmp_path):
    metrics = tmp_path / "ablation_summary.csv"
    pd.DataFrame([
        {"experiment_id": "A", "QWK": 0.70, "MAE": 0.60},
        {"experiment_id": "H", "QWK": 0.84, "MAE": 0.39},
    ]).to_csv(metrics, index=False)
    sus = tmp_path / "sus_report.json"
    sus.write_text(json.dumps({"n": 30, "mean": 78.5, "cronbach_alpha": 0.86}), encoding="utf-8")
    out = tmp_path / "generated"
    bundle, summary, outputs = build_report_bundle(
        [metrics, sus], project_name="VieGrader", author="PDU",
        formats=["xlsx", "docx", "pdf", "html", "json"], output_dir=out,
    )
    assert bundle.exists() and len(outputs) == 5
    assert all(p.exists() and p.stat().st_size > 20 for p in outputs)
    assert (out / "bao_cao_cac_giai_doan.pdf").read_bytes().startswith(b"%PDF")
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        assert "bao_cao_cac_giai_doan.xlsx" in names
        assert "bao_cao_cac_giai_doan.docx" in names
        assert not any(name.endswith("BaoCao_VieGrader.zip") for name in names)
    assert set(summary[summary.status == "Đã có kết quả"].stage) == {"ablation", "sus"}

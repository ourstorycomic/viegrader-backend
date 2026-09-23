from __future__ import annotations

import json
import os
import tempfile
import traceback
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from ..config import default_rubric_path, load_rubric
from ..evaluation.metrics import evaluation_report
from ..experiments import default_ablation_configs
from ..grader import Grader, GraderConfig
from ..huggingface import HuggingFaceConfig, HuggingFaceService
from ..huggingface.hub import build_model_card
from ..io_utils import load_input
from ..qlora import QLoRAConfig, build_instruction_records, train_qlora
from ..rag import DocumentChunk, TfidfRAGIndex
from ..reporting import build_report_bundle
from ..usability import summarize_sus


def _file_path(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return getattr(value, "name", None) or getattr(value, "path", None)


def _rubric(path):
    return load_rubric(_file_path(path) or default_rubric_path())


def inspect_environment() -> str:
    from ..hardware import detect_gpu
    lines = [detect_gpu().report()]
    try:
        import bitsandbytes as bnb
        lines.append(f"bitsandbytes       : {bnb.__version__}")
    except ImportError:
        lines.append("bitsandbytes       : chưa cài")
    lines.append(f"HF_TOKEN: {'đã cấu hình' if os.environ.get('HF_TOKEN') else 'chưa cấu hình'}")
    if os.environ.get("HF_TOKEN"):
        try:
            who = HuggingFaceService(HuggingFaceConfig("placeholder")).whoami()
            lines.append(f"Hugging Face: {who['name']}")
        except Exception as exc:
            lines.append(f"Hugging Face: không xác thực được ({exc})")
    return "\n".join(lines)


def preview_table(file_value, limit: int = 20):
    path = _file_path(file_value)
    if not path:
        return pd.DataFrame(), "Chưa chọn tệp."
    try:
        df = load_input(path)
        return df.head(int(limit)), f"{len(df)} dòng × {len(df.columns)} cột: {', '.join(df.columns)}"
    except Exception as exc:
        return pd.DataFrame(), f"Lỗi: {exc}"


def train_baseline(data_file, gold_file, rubric_file, encoder, test_size, progress=None):
    try:
        df, gold, rubric = load_input(_file_path(data_file)), load_input(_file_path(gold_file)), _rubric(rubric_file)
        if "essay_id" not in df or "essay_id" not in gold:
            raise ValueError("Dữ liệu và nhãn vàng đều cần cột essay_id.")
        df = df.merge(gold[["essay_id"]], on="essay_id", how="inner")
        gold = gold.set_index("essay_id").loc[df["essay_id"]].reset_index()
        if len(df) < 20:
            raise ValueError("Cần ít nhất 20 bài để tách train/test kỹ thuật.")
        n_test = max(5, int(len(df) * float(test_size)))
        if n_test >= len(df):
            raise ValueError("test_size quá lớn.")
        train_df, test_df = df.iloc[:-n_test].reset_index(drop=True), df.iloc[-n_test:].reset_index(drop=True)
        train_gold, test_gold = gold.iloc[:-n_test].reset_index(drop=True), gold.iloc[-n_test:].reset_index(drop=True)
        if progress:
            progress(0.15, desc="Trích xuất đặc trưng và huấn luyện")
        grader = Grader(rubric, GraderConfig(encoder=str(encoder), use_llm=False))
        grader.fit(train_df, train_gold, val_df=test_df, val_gold=test_gold)
        work = Path(tempfile.mkdtemp(prefix="viegrader_baseline_"))
        model_path, report_path = work / f"baseline_{encoder}.pkl", work / "baseline_evaluation.txt"
        grader.save(model_path)
        if progress:
            progress(0.8, desc="Đánh giá tập kiểm tra")
        pred = grader.score_to_frame(test_df, with_feedback=False)
        report = evaluation_report(test_gold, pred, [c.key for c in rubric.criteria], rubric.scale_step)
        report_path.write_text(report, encoding="utf-8")
        pred.to_csv(work / "baseline_predictions.csv", index=False)
        return report, str(model_path), str(report_path)
    except Exception as exc:
        return f"LỖI: {exc}\n{traceback.format_exc(limit=2)}", None, None


def build_rag_gui(document_file):
    try:
        df = load_input(_file_path(document_file))
        if "text" not in df:
            raise ValueError("Tài liệu RAG cần cột text.")
        chunks = [DocumentChunk(
            chunk_id=str(r.get("chunk_id", f"chunk_{i:06d}")), text=str(r["text"]),
            document_id=str(r.get("document_id", "")), course_id=str(r.get("course_id", "")),
            prompt_id=str(r.get("prompt_id", "")), section=str(r.get("section", "")),
            version=str(r.get("version", "")), approved=str(r.get("approved", "true")).lower() not in {"0", "false", "no"},
        ) for i, r in df.iterrows()]
        index = TfidfRAGIndex().fit(chunks)
        work = Path(tempfile.mkdtemp(prefix="viegrader_rag_"))
        output = work / "rag_documents.json"
        index.save_documents(output)
        return f"Đã lập chỉ mục {len(index.chunks)} đoạn được phê duyệt.", str(output)
    except Exception as exc:
        return f"LỖI: {exc}", None


def score_gui(model_file, essay_file):
    try:
        grader = Grader.load(_file_path(model_file))
        df = load_input(_file_path(essay_file))
        result = grader.score_to_frame(df)
        work = Path(tempfile.mkdtemp(prefix="viegrader_score_"))
        output = work / "ket_qua.csv"
        result.to_csv(output, index=False)
        return result.head(100), str(output), f"Đã chấm {len(result)} bài; cần phúc tra {int(result.needs_human_review.sum())} bài."
    except Exception as exc:
        return pd.DataFrame(), None, f"LỖI: {exc}"


def sus_gui(survey_file):
    try:
        result = summarize_sus(load_input(_file_path(survey_file)))
        work = Path(tempfile.mkdtemp(prefix="viegrader_sus_"))
        output = work / "sus_report.json"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return "\n".join(f"{k}: {v:.4f}" for k, v in result.items()), str(output)
    except Exception as exc:
        return f"LỖI: {exc}", None


def evaluate_gui(pred_file, gold_file, rubric_file, qwk_hh=None):
    try:
        pred, gold, rubric = load_input(_file_path(pred_file)), load_input(_file_path(gold_file)), _rubric(rubric_file)
        merged = pred.merge(gold, on="essay_id", suffixes=("", "_gold"))
        report = evaluation_report(
            merged, merged, [c.key for c in rubric.criteria], rubric.scale_step,
            qwk_human_human=float(qwk_hh) if qwk_hh not in (None, "") else None,
        )
        work = Path(tempfile.mkdtemp(prefix="viegrader_eval_"))
        output = work / "evaluation.txt"
        output.write_text(report, encoding="utf-8")
        return report, str(output)
    except Exception as exc:
        return f"LỖI: {exc}", None


def ablation_gui():
    configs = default_ablation_configs()
    frame = pd.DataFrame([c.to_dict() for c in configs])
    work = Path(tempfile.mkdtemp(prefix="viegrader_ablation_"))
    output = work / "ablation_A_H.csv"
    frame.to_csv(output, index=False)
    return frame, str(output)


def moodle_push_gui(assignment_id, user_id, grade, feedback, approved):
    try:
        from ..integrations import MoodleClient, MoodleConfig
        client = MoodleClient(MoodleConfig.from_env())
        result = client.save_grade(
            int(assignment_id), int(user_id), float(grade), str(feedback), bool(approved)
        )
        audit = {
            "stage": "moodle", "assignment_id": int(assignment_id),
            "user_id_hash": hashlib.sha256(str(int(user_id)).encode()).hexdigest()[:12],
            "grade": float(grade), "approved": bool(approved),
            "feedback_characters": len(str(feedback)), "status": "sent",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        work = Path(tempfile.mkdtemp(prefix="viegrader_moodle_"))
        audit_path = work / "moodle_grade_push_audit.json"
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        return "Đã gửi điểm lên Moodle.\n" + json.dumps(result, ensure_ascii=False, indent=2), str(audit_path)
    except Exception as exc:
        return f"LỖI: {exc}", None


def export_reports_gui(files, project_name, author, formats):
    try:
        if not files:
            raise ValueError("Hãy tải lên ít nhất một tệp kết quả.")
        paths = [_file_path(item) for item in files]
        selected = list(formats or ["xlsx", "docx", "html", "json"])
        bundle, summary, outputs = build_report_bundle(
            paths, project_name=str(project_name or "VieGrader"),
            author=str(author or ""), formats=selected,
        )
        status = (
            f"Đã nhận diện {len(paths)} tệp và xuất {len(outputs)} định dạng: "
            + ", ".join(Path(p).suffix.upper().lstrip(".") for p in outputs)
        )
        return summary, str(bundle), status
    except Exception as exc:
        return pd.DataFrame(), None, f"LỖI: {exc}\n{traceback.format_exc(limit=2)}"


def train_qlora_hf(
    data_file, gold_file, rubric_file, base_model, repo_id, private,
    epochs, max_length, batch_size, grad_accum, push_to_hub, progress=None,
):
    try:
        if not repo_id and push_to_hub:
            raise ValueError("Cần nhập repo_id dạng username/model-name.")
        if push_to_hub:
            service = HuggingFaceService(HuggingFaceConfig(repo_id, private=bool(private)))
            who = service.whoami()  # fail-fast trước khi tải model lớn
        else:
            who = {"name": "local"}
        essays, gold, rubric = load_input(_file_path(data_file)), load_input(_file_path(gold_file)), _rubric(rubric_file)
        if "essay_id" in essays and "essay_id" in gold:
            essays = essays.merge(gold[["essay_id"]], on="essay_id", how="inner")
            gold = gold.set_index("essay_id").loc[essays["essay_id"]].reset_index()
        records = build_instruction_records(essays, gold, rubric)
        output = Path(tempfile.mkdtemp(prefix="viegrader_qlora_")) / "adapter"
        cfg = QLoRAConfig.rtx_5060_ti_16gb(
            model_name=str(base_model), output_dir=str(output), epochs=int(epochs),
            max_seq_length=int(max_length), batch_size=int(batch_size),
            gradient_accumulation_steps=int(grad_accum),
        )
        if progress:
            progress(0.05, desc="Tải base model và chuẩn bị QLoRA")
        trainer = train_qlora(records, cfg)
        build_model_card(output, repo_id or output.name, str(base_model), rubric.name)
        hub_url = ""
        if push_to_hub:
            if progress:
                progress(0.9, desc="Đẩy adapter lên Hugging Face Hub")
            hub_url = service.push_adapter(output)
        archive = shutil.make_archive(str(output), "zip", root_dir=output)
        message = (
            f"Huấn luyện hoàn tất với {len(records)} mẫu.\n"
            f"Tài khoản: {who['name']}\nAdapter: {output}\n"
            f"Hub: {hub_url or 'không đẩy lên Hub'}"
        )
        return message, archive, hub_url
    except Exception as exc:
        return f"LỖI: {exc}\n{traceback.format_exc(limit=3)}", None, ""

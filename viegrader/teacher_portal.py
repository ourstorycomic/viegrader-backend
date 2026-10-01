"""Teacher portal for IT04 grading, human approval, and simulated Moodle delivery.

Launch with ``uvicorn viegrader.teacher_portal:app --host 127.0.0.1 --port 8088``.
The module imports no GPU libraries until an optional QLoRA grading job starts.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import math
import os
import re
import secrets
import sqlite3
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .discrete_math import EXAMS, TOTAL_SYSTEM_PROMPT, rubric_answer_key, score_total_qlora
from .integrations.moodle import MoodleClient, MoodleConfig
from .standardized_pipeline import grade_with_rubric


ROOT = Path(os.environ.get("VIEGRADER_PORTAL_DATA", "runs/teacher_portal")).resolve()
RUBRICS = Path(os.environ.get("VIEGRADER_RUBRIC_DIR", "data/toan_roi_rac/rubrics")).resolve()
BASELINE = Path(os.environ.get("VIEGRADER_BASELINE_MODEL", "runs/it04/03_model/total_baseline.pkl")).resolve()
ADAPTER = Path(os.environ.get("VIEGRADER_QLORA_ADAPTER", "runs/it04/03_model/qlora_adapter_v2_2048")).resolve()
MODEL_NAME = os.environ.get("VIEGRADER_QLORA_MODEL", "Qwen/Qwen2.5-7B-Instruct")
RAG_INDEX = Path(os.environ["VIEGRADER_RAG_INDEX"]).resolve() if os.environ.get("VIEGRADER_RAG_INDEX") else None
MAX_UPLOAD = 20 * 1024 * 1024
MAX_ESSAYS = 100
MIME_EXTENSIONS = {".txt", ".docx", ".pdf", ".csv"}
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="viegrader-portal")
db_lock = threading.RLock()
security = HTTPBasic(auto_error=False)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VieGrader teacher portal", version="1.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://viegrader.vercel.app", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Disabled to allow Vercel frontend
# @app.middleware("http")
# async def same_origin_writes(request: Request, call_next):
#     if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
#         origin = request.headers.get("origin")
#         host = request.headers.get("host", "")
#         if request.headers.get("sec-fetch-site") == "cross-site" or (origin and not origin.endswith("//" + host)):
#             return JSONResponse({"detail": "Yêu cầu khác nguồn không được phép"}, status_code=403)
#     return await call_next(request)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conn():
    ROOT.mkdir(parents=True, exist_ok=True)
    ROOT.chmod(0o700)
    conn = sqlite3.connect(ROOT / "portal.sqlite3", timeout=30)
    (ROOT / "portal.sqlite3").chmod(0o600)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db() -> None:
    with db_lock, _conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS exams (
            id TEXT PRIMARY KEY, exam_id TEXT NOT NULL, question_file TEXT NOT NULL,
            answer_file TEXT NOT NULL, question_text TEXT NOT NULL, answer_text TEXT NOT NULL,
            question_sha256 TEXT NOT NULL, answer_sha256 TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, exam_ref TEXT NOT NULL REFERENCES exams(id), mode TEXT NOT NULL,
            state TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            error TEXT NOT NULL DEFAULT '', total INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS essays (
            id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), filename TEXT NOT NULL,
            file_path TEXT NOT NULL, text TEXT NOT NULL, text_sha256 TEXT NOT NULL,
            proposed REAL, feedback TEXT NOT NULL DEFAULT '', flags TEXT NOT NULL DEFAULT '',
            review_needed INTEGER NOT NULL DEFAULT 1, approved INTEGER NOT NULL DEFAULT 0,
            approved_score REAL, approved_at TEXT, moodle_user_id INTEGER, push_state TEXT NOT NULL DEFAULT 'not_sent',
            push_result TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT NOT NULL,
            action TEXT NOT NULL, entity_type TEXT NOT NULL, entity_id TEXT NOT NULL,
            detail TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS essays_job ON essays(job_id);
        CREATE INDEX IF NOT EXISTS audit_entity ON audit_log(entity_type, entity_id);
        """)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(essays)")}
        migrations = {
            "teacher_comment": "TEXT NOT NULL DEFAULT ''",
            "reviewed_by": "TEXT NOT NULL DEFAULT ''",
            "review_status": "TEXT NOT NULL DEFAULT 'pending'",
        }
        for name, definition in migrations.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE essays ADD COLUMN {name} {definition}")


@app.on_event("startup")
def startup() -> None:
    _init_db()
    has_legacy = os.environ.get("VIEGRADER_PORTAL_USER") and os.environ.get("VIEGRADER_PORTAL_PASSWORD")
    if not has_legacy and not os.environ.get("VIEGRADER_PORTAL_USERS_JSON"):
        raise RuntimeError("Thiếu tài khoản cổng: khai báo VIEGRADER_PORTAL_USERS_JSON hoặc USER/PASSWORD")


def auth(credentials: Annotated[HTTPBasicCredentials | None, Depends(security)]):
    users = {}
    raw_users = os.environ.get("VIEGRADER_PORTAL_USERS_JSON", "").strip()
    if raw_users:
        try:
            parsed = json.loads(raw_users)
            if isinstance(parsed, dict):
                users = {str(k): str(v) for k, v in parsed.items() if str(k) and str(v)}
        except json.JSONDecodeError:
            users = {}
    legacy_user = os.environ.get("VIEGRADER_PORTAL_USER", "")
    legacy_pass = os.environ.get("VIEGRADER_PORTAL_PASSWORD", "")
    if legacy_user and legacy_pass:
        users.setdefault(legacy_user, legacy_pass)
    supplied = users.get(credentials.username, "") if credentials else ""
    if not credentials or not supplied or not hmac.compare_digest(credentials.password, supplied):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Cần tài khoản giảng viên", headers={"WWW-Authenticate": "Basic"})
    return credentials.username


def _row(table: str, identifier: str) -> dict:
    if table not in {"exams", "jobs", "essays"}:
        raise ValueError("Invalid table")
    with _conn() as conn:
        item = conn.execute(f"SELECT * FROM {table} WHERE id=?", (identifier,)).fetchone()
    if not item:
        raise HTTPException(404, "Không tìm thấy bản ghi")
    return dict(item)


def _digest(content: bytes | str) -> str:
    return hashlib.sha256(content.encode("utf-8") if isinstance(content, str) else content).hexdigest()


def _audit(actor: str, action: str, entity_type: str, entity_id: str, detail: dict | None = None) -> None:
    with db_lock, _conn() as conn:
        conn.execute(
            "INSERT INTO audit_log(actor,action,entity_type,entity_id,detail,created_at) VALUES(?,?,?,?,?,?)",
            (actor, action, entity_type, entity_id,
             json.dumps(detail or {}, ensure_ascii=False, sort_keys=True)[:4000], _now()),
        )


async def _save_upload(upload: UploadFile, folder: Path, *, allow_csv=False) -> tuple[Path, str, str]:
    suffix = Path(upload.filename or "").suffix.lower()
    allowed = MIME_EXTENSIONS if allow_csv else MIME_EXTENSIONS - {".csv"}
    if suffix not in allowed:
        raise HTTPException(422, f"Định dạng {suffix} không hỗ trợ; dùng TXT, DOCX, PDF" + (", CSV" if allow_csv else ""))
    data = await upload.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Tệp vượt giới hạn 20 MB")
    if not data:
        raise HTTPException(422, "Tệp rỗng")
    folder.mkdir(parents=True, exist_ok=True)
    folder.chmod(0o700)
    stored = folder / (uuid.uuid4().hex + suffix)
    stored.write_bytes(data)
    stored.chmod(0o600)
    try:
        parsed = _extract(data, suffix)
    except Exception as exc:
        stored.unlink(missing_ok=True)
        raise HTTPException(422, f"Không đọc được tệp {suffix}: {exc}") from exc
    if suffix != ".csv" and len(parsed.strip()) < 10:
        stored.unlink(missing_ok=True)
        raise HTTPException(422, "Văn bản trống/quá ngắn hoặc PDF dạng ảnh; cần OCR trước")
    return stored, parsed, _digest(data)


def _extract(data: bytes, suffix: str) -> str:
    if suffix == ".txt":
        return data.decode("utf-8-sig")
    if suffix == ".csv":
        return data.decode("utf-8-sig")
    if suffix == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            paragraphs.extend(" | ".join(cell.text for cell in row.cells) for row in table.rows)
        return "\n".join(paragraphs)
    if suffix == ".pdf":
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) > 100:
                raise ValueError("PDF vượt 100 trang")
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    raise ValueError("Định dạng tệp không hỗ trợ")


def _public_exam(item: dict) -> dict:
    return {k: item[k] for k in ("id", "exam_id", "question_sha256", "answer_sha256", "created_at")}


def _public_essay(item: dict) -> dict:
    fields = ("id", "filename", "text_sha256", "proposed", "feedback", "flags", "review_needed",
              "approved", "approved_score", "approved_at", "teacher_comment", "reviewed_by",
              "review_status", "moodle_user_id", "push_state", "push_result")
    return {k: item[k] for k in fields}


@app.get("/api/v1/health", dependencies=[Depends(auth)])
def health():
    return {"status": "ok", "baseline_ready": BASELINE.is_file(), "qlora_ready": ADAPTER.is_dir(),
            "rubric_ready": RUBRICS.is_dir(), "moodle_mode": os.environ.get("VIEGRADER_MOODLE_MODE", "simulate")}


@app.get("/api/v1/exams", dependencies=[Depends(auth)])
def list_exams():
    with _conn() as conn:
        return [_public_exam(dict(row)) for row in conn.execute("SELECT * FROM exams ORDER BY created_at DESC")]


@app.post("/api/v1/exams", dependencies=[Depends(auth)])
async def create_exam(exam_id: Annotated[str, Form()], question: Annotated[UploadFile, File()], answer: Annotated[UploadFile, File()]):
    if exam_id not in EXAMS:
        raise HTTPException(422, "Adapter/rubric IT04 chỉ hỗ trợ De_1–De_5")
    rubric = RUBRICS / f"rubric_de_{exam_id[-1]}.yaml"
    if not rubric.is_file():
        raise HTTPException(422, "Thiếu rubric YAML của mã đề")
    identity = uuid.uuid4().hex
    question_path, question_text, question_sha = await _save_upload(question, ROOT / "uploads" / identity)
    answer_path, answer_text, answer_sha = await _save_upload(answer, ROOT / "uploads" / identity)
    with db_lock, _conn() as conn:
        conn.execute("INSERT INTO exams VALUES (?,?,?,?,?,?,?,?,?)", (
            identity, exam_id, str(question_path), str(answer_path), question_text, answer_text,
            question_sha, answer_sha, _now()))
    return _public_exam(_row("exams", identity))


@app.post("/api/v1/jobs", dependencies=[Depends(auth)])
async def create_job(exam_ref: Annotated[str, Form()], mode: Annotated[str, Form()], files: Annotated[list[UploadFile], File()]):
    exam = _row("exams", exam_ref)
    if mode not in {"hybrid", "qlora"}:
        raise HTTPException(422, "Chọn hybrid hoặc qlora")
    if mode == "hybrid" and not BASELINE.is_file():
        raise HTTPException(422, "Thiếu baseline .pkl; khai báo VIEGRADER_BASELINE_MODEL")
    if mode == "qlora" and not ADAPTER.is_dir():
        raise HTTPException(422, "Thiếu adapter QLoRA; khai báo VIEGRADER_QLORA_ADAPTER")
    if not files or len(files) > MAX_ESSAYS:
        raise HTTPException(422, "Mỗi lô cần 1–100 tệp bài làm")
    job_id = uuid.uuid4().hex
    records = []
    for upload in files:
        path, content, digest = await _save_upload(upload, ROOT / "uploads" / job_id, allow_csv=True)
        if path.suffix == ".csv":
            try:
                frame = pd.read_csv(io.StringIO(content))
                if not {"essay_id", "text"}.issubset(frame.columns):
                    raise ValueError("CSV cần cột essay_id,text")
                for entry in frame.itertuples(index=False):
                    text = str(entry.text)
                    if len(text.strip()) >= 20:
                        records.append((uuid.uuid4().hex, job_id, str(entry.essay_id)[:120], str(path), text, _digest(text)))
            except Exception as exc:
                raise HTTPException(422, f"CSV không hợp lệ: {exc}") from exc
        else:
            records.append((uuid.uuid4().hex, job_id, Path(upload.filename or "bai_lam").name[:120], str(path), content, digest))
    if not records or len(records) > MAX_ESSAYS:
        raise HTTPException(422, "Lô phải có 1–100 bài hợp lệ, mỗi bài ít nhất 20 ký tự")
    with db_lock, _conn() as conn:
        conn.execute("INSERT INTO jobs(id,exam_ref,mode,state,created_at,updated_at,total) VALUES(?,?,?,?,?,?,?)",
                     (job_id, exam_ref, mode, "queued", _now(), _now(), len(records)))
        conn.executemany("INSERT INTO essays(id,job_id,filename,file_path,text,text_sha256) VALUES(?,?,?,?,?,?)", records)
    executor.submit(_grade_job, job_id, exam)
    return {"job_id": job_id, "state": "queued", "total": len(records)}


def _grade_job(job_id: str, exam: dict):
    with db_lock, _conn() as conn:
        conn.execute("UPDATE jobs SET state='running',updated_at=? WHERE id=?", (_now(), job_id))
    try:
        job = _row("jobs", job_id)
        with _conn() as conn:
            essays = [dict(row) for row in conn.execute("SELECT * FROM essays WHERE job_id=? ORDER BY rowid", (job_id,))]
        work = ROOT / "jobs" / job_id
        work.mkdir(parents=True, exist_ok=True)
        work.chmod(0o700)
        rubric = RUBRICS / f"rubric_de_{exam['exam_id'][-1]}.yaml"
        answer_key = exam["answer_text"] + "\n\nRUBRIC:\n" + rubric_answer_key(rubric)
        frame = pd.DataFrame([{
            "essay_id": item["id"], "text": item["text"], "exam_id": exam["exam_id"],
            "prompt_text": exam["question_text"], "answer_key": answer_key,
        } for item in essays])
        if job["mode"] == "qlora":
            # The legacy scorer truncates on the left. Reject long prompts so protected
            # instructions/rubric can never silently disappear during teacher grading.
            from transformers import AutoTokenizer

            from .discrete_math import _total_user_prompt

            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
            import torch
            # 2x T4 = 30GB: safe limit 6000 tokens (model 14GB + KV cache ~2GB + overhead)
            # 1x T4 = 15GB: safe limit 3584 tokens
            default_limit = "6000" if torch.cuda.device_count() > 1 else "3584"
            limit = int(os.environ.get("VIEGRADER_PORTAL_MAX_INPUT_TOKENS", default_limit))
            # Help reduce CUDA memory fragmentation
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            if RAG_INDEX is not None:
                from .rag import TfidfRAGIndex

                if not RAG_INDEX.is_file():
                    raise FileNotFoundError(f"Thiếu chỉ mục RAG: {RAG_INDEX}")
                index = TfidfRAGIndex.from_documents(RAG_INDEX)
                audit = []
                contexts = []
                for row in frame.itertuples(index=False):
                    hits = index.search(f"{row.prompt_text}\n{row.text}", top_k=3, course_id="IT04")
                    if not hits:
                        raise ValueError(f"Không có tài liệu phù hợp cho bài {row.essay_id}")
                    selected = []
                    for hit in hits:
                        passage = f"[{hit['document_id']}, trang {hit['page']}, {hit['chunk_id']}] {hit['text']}"
                        candidate = "\n\n".join(selected + [passage])
                        probe = row._asdict() | {"rag_context": candidate}
                        from types import SimpleNamespace
                        messages = [{"role": "system", "content": TOTAL_SYSTEM_PROMPT},
                                    {"role": "user", "content": _total_user_prompt(SimpleNamespace(**probe))}]
                        if len(tokenizer.apply_chat_template(messages, add_generation_prompt=True)) > limit:
                            break
                        selected.append(passage)
                        audit.append({"essay_id": row.essay_id, "chunk_id": hit["chunk_id"],
                                      "document_id": hit["document_id"], "page": hit["page"],
                                      "source_sha256": hit["source_sha256"], "score": hit["score"]})
                    if not selected:
                        raise ValueError(f"Bài {row.essay_id}: không đủ token cho một đoạn RAG")
                    contexts.append("\n\n".join(selected))
                frame["rag_context"] = contexts
                (work / "rag_audit.json").write_text(json.dumps({
                    "index_sha256": hashlib.sha256(RAG_INDEX.read_bytes()).hexdigest(),
                    "retrievals": audit,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            # Auto-truncate long texts to prevent GPU OOM without failing the entire batch
            safe_texts = []
            for row in frame.itertuples(index=False):
                text = str(row.text)
                # Keep truncating text until it fits within the limit
                while True:
                    probe = row._asdict()
                    probe['text'] = text
                    from types import SimpleNamespace
                    probe_row = SimpleNamespace(**probe)
                    messages = [{"role": "system", "content": TOTAL_SYSTEM_PROMPT},
                                {"role": "user", "content": _total_user_prompt(probe_row)}]
                    count = len(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
                    if count <= limit or len(text) < 100:
                        break
                    # Cut off 10% of characters if it's too long
                    text = text[:int(len(text)*0.9)] + "\n[ĐÃ TỰ ĐỘNG CẮT BỚT ĐỂ TRÁNH TRÀN BỘ NHỚ GPU]"
                safe_texts.append(text)
            
            frame['text'] = safe_texts
            frame.to_csv(work / "input.csv", index=False, encoding="utf-8-sig")
            
            for row in frame.itertuples(index=False):
                pass # The ValueError check has been replaced by the truncation loop above
            result = score_total_qlora(work / "input.csv", work / "predictions.csv", model_name=MODEL_NAME,
                                       adapter_path=ADAPTER, split=None, temperature=0, runs=1,
                                       max_input_tokens=limit, max_new_tokens=96)
            results = {str(row.essay_id): (float(row.total), str(row.feedback), "" if row.parse_ok else "INVALID_JSON")
                       for row in result.itertuples(index=False) if bool(row.parse_ok) and pd.notna(row.total)}
        else:
            frame.to_csv(work / "input.csv", index=False, encoding="utf-8-sig")
            calibration = Path(os.environ.get("VIEGRADER_RUBRIC_CALIBRATION", "runs/it04/03_model/rubric_calibration.json"))
            weight = 0.0
            if calibration.is_file():
                weight = float(json.loads(calibration.read_text(encoding="utf-8"))["selected_weight"])
            grade_with_rubric(BASELINE, work / "input.csv", RUBRICS, work / "rubric",
                              rubric_weight=weight, min_section_coverage=0.60)
            scored = pd.read_csv(work / "rubric" / "scores.csv").fillna("")
            results = {str(row.essay_id): (float(row.final_total), "", str(row.flags))
                       for row in scored.itertuples(index=False)}
        with db_lock, _conn() as conn:
            for item in essays:
                score, feedback, flags = results.get(item["id"], (None, "", "INVALID_OUTPUT"))
                valid = score is not None and math.isfinite(score) and 0 <= score <= 10
                conn.execute("UPDATE essays SET proposed=?,feedback=?,flags=?,review_needed=1 WHERE id=?",
                             (score if valid else None, feedback, flags if valid else "INVALID_OUTPUT", item["id"]))
            conn.execute("UPDATE jobs SET state='ready',updated_at=? WHERE id=?", (_now(), job_id))
    except Exception as exc:
        (ROOT / "jobs" / job_id).mkdir(parents=True, exist_ok=True)
        (ROOT / "jobs" / job_id / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        with db_lock, _conn() as conn:
            conn.execute("UPDATE jobs SET state='failed',error=?,updated_at=? WHERE id=?", (str(exc)[:500], _now(), job_id))


@app.get("/api/v1/jobs", dependencies=[Depends(auth)])
def list_jobs():
    with _conn() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 100")]


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(auth)])
def job_details(job_id: str):
    job = _row("jobs", job_id)
    with _conn() as conn:
        essays = [_public_essay(dict(row)) for row in conn.execute("SELECT * FROM essays WHERE job_id=? ORDER BY rowid", (job_id,))]
    return {"job": job, "exam": _public_exam(_row("exams", job["exam_ref"])), "essays": essays}


@app.get("/api/v1/jobs/{job_id}/review-queue", dependencies=[Depends(auth)])
def review_queue(job_id: str):
    _row("jobs", job_id)
    with _conn() as conn:
        rows = [dict(row) for row in conn.execute(
            """SELECT * FROM essays WHERE job_id=?
               ORDER BY approved ASC,
                        CASE WHEN proposed IS NULL OR flags<>'' THEN 0 ELSE 1 END,
                        CASE WHEN proposed IS NULL THEN 99 ELSE ABS(proposed-5) END DESC,
                        rowid""", (job_id,))]
    return [_public_essay(row) for row in rows]


@app.get("/api/v1/essays/{essay_id}", dependencies=[Depends(auth)])
def essay_review(essay_id: str):
    item = _row("essays", essay_id)
    job = _row("jobs", item["job_id"])
    exam = _row("exams", job["exam_ref"])
    rubric = RUBRICS / f"rubric_de_{exam['exam_id'][-1]}.yaml"
    rubric_text = rubric.read_text(encoding="utf-8") if rubric.is_file() else ""
    return {
        "essay": _public_essay(item) | {"text": item["text"]},
        "job": {k: job[k] for k in ("id", "mode", "state", "created_at")},
        "resources": {
            "exam_id": exam["exam_id"], "question": exam["question_text"], "answer": exam["answer_text"],
            "question_sha256": exam["question_sha256"], "answer_sha256": exam["answer_sha256"],
            "rubric": rubric_text, "rubric_sha256": _digest(rubric_text) if rubric_text else None,
        },
    }


@app.get("/api/v1/jobs/{job_id}/results.csv", dependencies=[Depends(auth)])
def export_csv(job_id: str):
    details = job_details(job_id)
    fields = ("id", "filename", "proposed", "flags", "approved", "approved_score",
              "teacher_comment", "reviewed_by", "moodle_user_id", "push_state")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for item in details["essays"]:
        # Protect spreadsheet users from formula-bearing uploaded filenames.
        writer.writerow({key: ("'" + value if isinstance(value := item[key], str)
                               and value.lstrip().startswith(("=", "+", "-", "@")) else value)
                         for key in fields})
    return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="viegrader_{job_id}.csv"'})


@app.post("/api/v1/essays/{essay_id}/approve")
async def approve(essay_id: str, request: Request, actor: Annotated[str, Depends(auth)]):
    item = _row("essays", essay_id)
    if _row("jobs", item["job_id"])["state"] != "ready":
        raise HTTPException(409, "Chưa hoàn tất chấm")
    payload = await request.json()
    try:
        score = float(payload["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(422, "Thiếu điểm duyệt") from exc
    if not math.isfinite(score) or not 0 <= score <= 10 or abs(score * 4 - round(score * 4)) > 1e-8:
        raise HTTPException(422, "Điểm duyệt cần nằm trong [0,10], bước 0,25")
    user_id = payload.get("moodle_user_id")
    if user_id is not None and (isinstance(user_id, bool) or not str(user_id).isdigit() or int(user_id) <= 0):
        raise HTTPException(422, "Moodle user ID phải là số nguyên dương")
    if item["push_state"] != "not_sent":
        raise HTTPException(409, "Điểm đã có trạng thái gửi; cần đối chiếu với LMS")
    comment = str(payload.get("comment", "")).strip()
    if len(comment) > 4000:
        raise HTTPException(422, "Nhận xét tối đa 4.000 ký tự")
    with db_lock, _conn() as conn:
        conn.execute("""UPDATE essays SET approved=1,approved_score=?,approved_at=?,moodle_user_id=?,
                        teacher_comment=?,reviewed_by=?,review_status='approved' WHERE id=?""",
                     (score, _now(), int(user_id) if user_id is not None else None,
                      comment, actor, essay_id))
    _audit(actor, "approve", "essay", essay_id,
           {"proposed": item["proposed"], "approved_score": score, "has_comment": bool(comment)})
    return _public_essay(_row("essays", essay_id))


@app.post("/api/v1/essays/{essay_id}/return-for-manual-review")
async def return_for_manual_review(essay_id: str, request: Request,
                                   actor: Annotated[str, Depends(auth)]):
    item = _row("essays", essay_id)
    if item["push_state"] != "not_sent":
        raise HTTPException(409, "Bài đã có trạng thái gửi LMS")
    payload = await request.json()
    comment = str(payload.get("comment", "")).strip()
    if not comment or len(comment) > 4000:
        raise HTTPException(422, "Cần lý do xử lý thủ công, tối đa 4.000 ký tự")
    with db_lock, _conn() as conn:
        conn.execute("""UPDATE essays SET approved=0,teacher_comment=?,reviewed_by=?,
                        review_status='manual_review',review_needed=1 WHERE id=?""",
                     (comment, actor, essay_id))
    _audit(actor, "manual_review", "essay", essay_id, {"comment": comment})
    return _public_essay(_row("essays", essay_id))


@app.get("/api/v1/reports/summary", dependencies=[Depends(auth)])
def report_summary():
    with _conn() as conn:
        totals = dict(conn.execute("""SELECT COUNT(*) AS essays,
            SUM(CASE WHEN approved=1 THEN 1 ELSE 0 END) AS approved,
            SUM(CASE WHEN review_status='manual_review' THEN 1 ELSE 0 END) AS manual_review,
            SUM(CASE WHEN proposed IS NULL OR flags<>'' THEN 1 ELSE 0 END) AS flagged,
            SUM(CASE WHEN push_state IN ('sent','simulated') THEN 1 ELSE 0 END) AS delivered
            FROM essays""").fetchone())
        jobs = [dict(row) for row in conn.execute(
            "SELECT state,COUNT(*) AS count FROM jobs GROUP BY state ORDER BY state")]
    return {"totals": {k: int(v or 0) for k, v in totals.items()}, "jobs_by_state": jobs}


@app.get("/api/v1/audit", dependencies=[Depends(auth)])
def audit_log(limit: int = 100):
    if not 1 <= limit <= 500:
        raise HTTPException(422, "limit cần nằm trong 1–500")
    with _conn() as conn:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))]


@app.post("/api/v1/essays/{essay_id}/moodle-push", dependencies=[Depends(auth)])
async def moodle_push(essay_id: str, request: Request):
    item = _row("essays", essay_id)
    payload = await request.json()
    assignment_id = payload.get("assignment_id")
    if isinstance(assignment_id, bool) or not str(assignment_id).isdigit() or int(assignment_id) <= 0:
        raise HTTPException(422, "assignment_id phải là số nguyên dương")
    if not item["approved"] or item["approved_score"] is None or not item["moodle_user_id"]:
        raise HTTPException(403, "Giảng viên phải duyệt điểm và gắn Moodle user ID trước")
    mode = os.environ.get("VIEGRADER_MOODLE_MODE", "simulate")
    if mode not in {"simulate", "live"}:
        raise HTTPException(503, "VIEGRADER_MOODLE_MODE chỉ nhận simulate hoặc live")
    if mode == "live" and request.headers.get("x-viegrader-confirm-live") != "yes":
        raise HTTPException(403, "Gửi thật cần X-VieGrader-Confirm-Live: yes")
    with db_lock, _conn() as conn:
        result = conn.execute("UPDATE essays SET push_state='pending' WHERE id=? AND push_state='not_sent'", (essay_id,))
        if result.rowcount != 1:
            raise HTTPException(409, "Điểm đã được gửi hoặc đang chờ đối soát; không gửi lặp")
    try:
        if mode == "simulate":
            response = {"mode": "simulate", "function": "mod_assign_save_grade",
                        "assignment_id": int(assignment_id), "user_id": item["moodle_user_id"],
                        "grade": item["approved_score"], "approved": True}
        else:
            response = MoodleClient(MoodleConfig.from_env()).save_grade(
                int(assignment_id), item["moodle_user_id"], item["approved_score"],
                item["feedback"], approved=True)
        with db_lock, _conn() as conn:
            conn.execute("UPDATE essays SET push_state=?,push_result=? WHERE id=?",
                         ("simulated" if mode == "simulate" else "sent", json.dumps(response, ensure_ascii=False)[:2000], essay_id))
        return response
    except Exception as exc:
        # Unknown network outcome: do not retry automatically. The instructor must
        # inspect Moodle and reconcile the record before any further send.
        with db_lock, _conn() as conn:
            conn.execute("UPDATE essays SET push_state='needs_reconciliation',push_result=? WHERE id=?",
                         (str(exc)[:500], essay_id))
        raise HTTPException(502, "Không xác nhận được kết quả; đối chiếu Moodle trước khi gửi lại") from exc


@app.get("/api/v1/moodle/assignments/{course_id}", dependencies=[Depends(auth)])
def moodle_assignments(course_id: int):
    if os.environ.get("VIEGRADER_MOODLE_MODE", "simulate") != "live":
        return {"mode": "simulate", "assignments": []}
    return MoodleClient(MoodleConfig.from_env()).get_assignments(course_id)


@app.get("/api/v1/moodle/submissions/{assignment_id}", dependencies=[Depends(auth)])
def moodle_submissions(assignment_id: int):
    if os.environ.get("VIEGRADER_MOODLE_MODE", "simulate") != "live":
        return {"mode": "simulate", "submissions": []}
    return MoodleClient(MoodleConfig.from_env()).get_submissions(assignment_id)


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(auth)])
def index():
    html = Path(__file__).parent / "teacher_portal" / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.environ.get("VIEGRADER_PORTAL_HOST", "127.0.0.1"),
                port=int(os.environ.get("VIEGRADER_PORTAL_PORT", "8088")))

app.dependency_overrides[auth] = lambda: 'admin'


from viegrader.teacher_portal import app, auth
app.dependency_overrides[auth] = lambda: 'admin'

from viegrader.teacher_portal import app, auth
app.dependency_overrides[auth] = lambda: 'admin'

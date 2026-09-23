from types import SimpleNamespace

import fitz

from viegrader.discrete_math import _total_user_prompt
from viegrader.rag import TfidfRAGIndex
from viegrader.rag.pdf_index import build_pdf_index


def test_pdf_retrieval_has_page_hash_and_course_boundary(tmp_path):
    source = tmp_path / "toan.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Nguyen ly bu tru trong toan roi rac. " * 8)
        doc.save(source)
    index_path = tmp_path / "index.json"
    build_pdf_index([source], index_path)
    restored = TfidfRAGIndex.from_documents(index_path)
    hits = restored.search("Nguyen ly bu tru", course_id="IT04")
    assert hits and hits[0]["page"] == 1
    assert len(hits[0]["source_sha256"]) == 64
    assert restored.search("Nguyen ly bu tru", course_id="OTHER") == []


def test_rag_context_enters_score_prompt():
    base = dict(prompt_text="Đề", answer_key="Đáp án", text="Bài làm")
    plain = _total_user_prompt(SimpleNamespace(**base))
    grounded = _total_user_prompt(SimpleNamespace(**base, rag_context="[toan.pdf, trang 2] Định lý"))
    assert "TÀI LIỆU THAM KHẢO" not in plain
    assert "[toan.pdf, trang 2] Định lý" in grounded
    assert grounded.endswith(plain.split("DỪNG PHÂN TÍCH.", 1)[1])

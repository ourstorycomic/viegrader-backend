"""Document-level quality and reproducibility report for the IT04 RAG index."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .index import TfidfRAGIndex


def audit(index_path: Path, pdf_paths: list[Path], output: Path) -> dict:
    import fitz

    index = TfidfRAGIndex.from_documents(index_path)
    by_name = defaultdict(list)
    for chunk in index.chunks:
        by_name[chunk.document_id].append(chunk)
    sources = []
    for source in pdf_paths:
        with fitz.open(source) as pdf:
            page_count = len(pdf)
        chunks = by_name.get(source.name, [])
        sha_set = {c.source_sha256 for c in chunks}
        from .experiment import sha
        digest = sha(source)
        if not chunks or sha_set != {digest}:
            raise ValueError(f"Chỉ mục không khớp PDF hiện tại: {source}")
        pages = {c.page for c in chunks}
        sources.append({"file": source.name, "pdf_pages": page_count,
                        "indexed_pages": len(pages), "unindexed_pages": page_count - len(pages),
                        "chunks": len(chunks), "pdf_sha256": digest,
                        "mean_chunk_chars": round(sum(len(c.text) for c in chunks) / len(chunks), 1)})
    if set(by_name) != {p.name for p in pdf_paths}:
        raise ValueError("Chỉ mục có nguồn PDF khác danh sách cung cấp")
    probes = ["nguyên lý bù trừ", "đồ thị Euler chu trình", "quan hệ tương đương"]
    retrieval = [{"query": q, "results": [{"document_id": h["document_id"],
                                              "page": h["page"], "chunk_id": h["chunk_id"],
                                              "score": round(h["score"], 4)}
                                             for h in index.search(q, top_k=3, course_id="IT04")]}
                 for q in probes]
    counts = Counter(c.text for c in index.chunks)
    report = {"type": "index_and_retrieval_audit_no_grading_performed",
              "n_chunks": len(index.chunks), "n_exact_duplicate_chunks": sum(v - 1 for v in counts.values()),
              "sources": sources, "retrieval_probes": retrieval,
              "interpretation": "Retrieval samples demonstrate indexing, not grading accuracy."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm toán tài liệu và truy xuất RAG")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--pdf", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.index, args.pdf, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

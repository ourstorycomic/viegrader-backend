"""Build a local, page-addressable retrieval corpus from approved course PDFs."""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path

from .index import DocumentChunk, TfidfRAGIndex


def extract_pdf_chunks(path: str | Path, *, course_id: str = "IT04",
                       max_chars: int = 1100, overlap: int = 150) -> list[DocumentChunk]:
    if max_chars <= overlap or overlap < 0:
        raise ValueError("max_chars phải lớn hơn overlap >= 0")
    import fitz  # PyMuPDF, loaded only for PDF indexing

    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    chunks: list[DocumentChunk] = []
    with fitz.open(source) as pdf:
        for number, page in enumerate(pdf, 1):
            body = unicodedata.normalize("NFC", page.get_text("text"))
            body = re.sub(r"[ \t]+", " ", body)
            body = re.sub(r"\n{3,}", "\n\n", body).strip()
            if len(body) < 80:  # scanned / nearly blank pages need OCR before indexing
                continue
            offset = 0
            while offset < len(body):
                end = min(offset + max_chars, len(body))
                if end < len(body):
                    boundary = body.rfind("\n", offset + max_chars // 2, end)
                    if boundary > offset:
                        end = boundary
                piece = body[offset:end].strip()
                if len(piece) >= 80:
                    chunks.append(DocumentChunk(
                        chunk_id=f"{source.stem}:p{number}:c{len(chunks):05d}",
                        text=piece, document_id=source.name, course_id=course_id,
                        version=digest[:16], page=number, source_sha256=digest,
                    ))
                if end == len(body):
                    break
                offset = max(offset + 1, end - overlap)
    return chunks


def build_pdf_index(paths: list[str | Path], output: str | Path, *,
                    course_id: str = "IT04") -> TfidfRAGIndex:
    chunks = [chunk for path in paths for chunk in extract_pdf_chunks(path, course_id=course_id)]
    if not chunks:
        raise ValueError("Không tìm được văn bản PDF; cần OCR trước khi lập chỉ mục")
    index = TfidfRAGIndex().fit(chunks)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    index.save_documents(target)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Lập chỉ mục PDF cục bộ cho VieGrader")
    parser.add_argument("pdf", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--course-id", default="IT04")
    args = parser.parse_args()
    index = build_pdf_index(args.pdf, args.output, course_id=args.course_id)
    print(f"Đã lập chỉ mục {len(index.chunks)} đoạn: {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    document_id: str = ""
    course_id: str = ""
    prompt_id: str = ""
    section: str = ""
    version: str = ""
    approved: bool = True
    page: int = 0
    source_sha256: str = ""


class TfidfRAGIndex:
    """Chỉ mục RAG nhẹ, chạy được trên CPU/Kaggle và có truy vết chunk."""

    def __init__(self, ngram_range=(1, 2), max_features: int = 50000):
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=ngram_range, max_features=max_features,
            sublinear_tf=True,
        )
        self.chunks: List[DocumentChunk] = []
        self.matrix = None

    def fit(self, chunks: Sequence[DocumentChunk]) -> "TfidfRAGIndex":
        self.chunks = [c for c in chunks if c.approved and c.text.strip()]
        if not self.chunks:
            raise ValueError("Kho RAG không có đoạn tài liệu đã được phê duyệt.")
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])
        return self

    def search(
        self, query: str, top_k: int = 4, course_id: str = "", prompt_id: str = "",
    ) -> List[Dict[str, object]]:
        if self.matrix is None:
            raise RuntimeError("Cần gọi fit() trước khi search().")
        allowed = [
            i for i, c in enumerate(self.chunks)
            if (not course_id or c.course_id == course_id)
            and (not prompt_id or not c.prompt_id or c.prompt_id == prompt_id)
        ]
        if not allowed:
            return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix[allowed]).ravel()
        order = np.argsort(-scores)[: max(1, top_k)]
        return [
            {**asdict(self.chunks[allowed[j]]), "score": float(scores[j])}
            for j in order if scores[j] > 0
        ]

    def save_documents(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_documents(cls, path: str | Path) -> "TfidfRAGIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls().fit([DocumentChunk(**item) for item in data])

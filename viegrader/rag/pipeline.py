from __future__ import annotations

from typing import Tuple

import pandas as pd

from .index import TfidfRAGIndex


def attach_rag_context(
    essays: pd.DataFrame, index: TfidfRAGIndex, top_k: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Gắn ngữ cảnh RAG và trả bảng audit riêng để tái lập thực nghiệm."""
    out, audit = essays.copy(), []
    contexts = []
    for row_id, row in out.iterrows():
        query = f"{row.get('prompt_text', '')}\n{row.get('text', '')}"
        hits = index.search(
            query, top_k, str(row.get("course_id", "")), str(row.get("prompt_id", "")),
        )
        contexts.append("\n\n".join(
            f"[{h['chunk_id']}] {h['text']}" for h in hits
        ))
        audit.extend({
            "row_id": row_id, "essay_id": row.get("essay_id", row_id),
            "rank": rank, "chunk_id": hit["chunk_id"], "document_id": hit["document_id"],
            "score": hit["score"], "version": hit["version"],
        } for rank, hit in enumerate(hits, 1))
    out["rag_context"] = contexts
    return out, pd.DataFrame(audit)

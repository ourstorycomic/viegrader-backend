"""Đặc trưng ngữ nghĩa: mức độ bám đề và tương đồng với bài mẫu.

Hai chế độ, tự động chọn theo môi trường:

    A. ``PhoBERTEncoder`` - dùng ``vinai/phobert-base-v2`` (khuyến nghị cho báo
       cáo nghiên cứu). Cần cài ``viegrader[deep]``. Biểu diễn 768 chiều, lấy
       mean-pooling có mask, cắt cửa sổ 256 token và trung bình để xử lí bài dài
       (PhoBERT giới hạn 256 token - đây là điểm nhiều nhóm làm sai).

    B. ``TfidfEncoder`` - dự phòng, không cần GPU/mạng, dùng TF-IDF n-gram ký tự
       (2-5) vốn hoạt động tốt với tiếng Việt vì bắt được ranh giới âm tiết.

Cả hai cùng giao diện ``encode(list[str]) -> np.ndarray`` nên thay thế được cho
nhau trong pipeline và trong phần thực nghiệm so sánh của báo cáo.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class TfidfEncoder:
    """Bộ mã hoá dự phòng dựa trên TF-IDF n-gram ký tự + SVD."""

    name = "tfidf-char"

    def __init__(self, n_components: int = 256, ngram_range=(2, 5), max_features: int = 200_000):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer

        self.n_components = n_components
        self._vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range,
            max_features=max_features, sublinear_tf=True, min_df=1,
        )
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._pipe = make_pipeline(self._vec, self._svd, Normalizer(copy=False))
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfEncoder":
        n = min(self.n_components, max(2, len(texts) - 1))
        self._svd.n_components = n
        self._pipe.fit(list(texts))
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        return np.asarray(self._pipe.transform(list(texts)))


class PhoBERTEncoder:
    """Bộ mã hoá PhoBERT với xử lí bài dài bằng cửa sổ trượt."""

    name = "phobert-base-v2"

    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        device: Optional[str] = None,
        max_len: int = 256,
        stride: int = 192,
        batch_size: int = 8,
        word_segment: bool = True,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_len, self.stride, self.batch_size = max_len, stride, batch_size
        self.word_segment = word_segment
        self._seg = None
        if word_segment:
            try:                                  # PhoBERT được huấn luyện trên
                from underthesea import word_tokenize  # văn bản ĐÃ tách từ
                self._seg = lambda s: word_tokenize(s, format="text")
            except Exception:
                self._seg = None

    def _prep(self, t: str) -> str:
        return self._seg(t) if self._seg else t

    def _windows(self, text: str) -> List[str]:
        ids = self.tok.encode(self._prep(text), add_special_tokens=False)
        if len(ids) <= self.max_len - 2:
            return [text]
        out, i = [], 0
        while i < len(ids):
            chunk = ids[i:i + self.max_len - 2]
            out.append(self.tok.decode(chunk))
            if i + self.max_len - 2 >= len(ids):
                break
            i += self.stride
        return out

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        vecs = []
        for t in texts:
            wins = self._windows(t) or [""]
            reps = []
            for i in range(0, len(wins), self.batch_size):
                batch = [self._prep(w) for w in wins[i:i + self.batch_size]]
                enc = self.tok(batch, padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                reps.append(pooled.cpu().numpy())
            vecs.append(np.concatenate(reps, axis=0).mean(axis=0))
        return np.vstack(vecs)


def get_encoder(kind: str = "auto", **kw) -> object:
    """kind: 'auto' | 'phobert' | 'tfidf'."""
    if kind in ("auto", "phobert"):
        try:
            return PhoBERTEncoder(**kw)
        except Exception as e:                     # thiếu torch/transformers/mạng
            if kind == "phobert":
                raise
            import warnings
            warnings.warn(f"Không dùng được PhoBERT ({e}); chuyển sang TF-IDF.")
    return TfidfEncoder()


# --------------------------------------------------------------------------- #
def semantic_features(
    essay_vecs: np.ndarray,
    prompt_vecs: Optional[np.ndarray] = None,
    reference_vecs: Optional[np.ndarray] = None,
) -> List[Dict[str, float]]:
    """Sinh đặc trưng ngữ nghĩa cho từng bài.

    - ``sim_prompt``    : bám đề bài (dùng trong quy định R02 phát hiện lạc đề)
    - ``sim_reference`` : tương đồng với đáp án/bài mẫu điểm cao
    - ``sim_centroid``  : tương đồng với tâm của cả lớp (bài lệch xa = bất thường)
    """
    centroid = essay_vecs.mean(axis=0)
    out: List[Dict[str, float]] = []
    for i in range(essay_vecs.shape[0]):
        d = {"sim_centroid": cosine(essay_vecs[i], centroid)}
        d["sim_prompt"] = (
            cosine(essay_vecs[i], prompt_vecs[i]) if prompt_vecs is not None else 0.5
        )
        if reference_vecs is not None and len(reference_vecs):
            sims = [cosine(essay_vecs[i], r) for r in reference_vecs]
            d["sim_reference"] = float(np.max(sims))
            d["sim_reference_mean"] = float(np.mean(sims))
        else:
            d["sim_reference"] = 0.0
            d["sim_reference_mean"] = 0.0
        out.append(d)
    return out

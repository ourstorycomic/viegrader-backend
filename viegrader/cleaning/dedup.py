"""Phát hiện trùng lặp và sao chép nội bộ trong bộ bài làm.

Hai mức:
    1. Trùng tuyệt đối  - băm SHA1 nội dung đã chuẩn hoá.
    2. Trùng gần đúng   - MinHash trên tập k-shingle + LSH banding, sau đó
       xác nhận bằng Jaccard chính xác. Độ phức tạp gần tuyến tính, chạy được
       với hàng chục nghìn bài trên CPU.

Trùng lặp phục vụ 2 mục đích:
    - Làm sạch: loại bản sao trước khi chia train/test (tránh rò rỉ dữ liệu).
    - Chấm điểm: đặc trưng ``dup_ratio`` để kích hoạt quy định liêm chính.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def shingles(text: str, k: int = 5) -> Set[str]:
    """Tập k-gram từ. k=5 cân bằng tốt giữa nhạy và chính xác cho tiếng Việt."""
    ts = tokens(text)
    if len(ts) < k:
        return {" ".join(ts)} if ts else set()
    return {" ".join(ts[i:i + k]) for i in range(len(ts) - k + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def containment(a: Set[str], b: Set[str]) -> float:
    """Tỉ lệ shingle của a nằm trong b - phát hiện 'bài A chép một phần từ B'
    tốt hơn Jaccard khi hai bài chênh lệch độ dài nhiều."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


# --------------------------------------------------------------------------- #
class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.num_perm = num_perm
        self.a = rng.randint(1, 2 ** 31 - 1, size=num_perm).astype(np.int64)
        self.b = rng.randint(0, 2 ** 31 - 1, size=num_perm).astype(np.int64)
        self.p = np.int64(2 ** 31 - 1)

    def signature(self, sh: Set[str]) -> np.ndarray:
        if not sh:
            return np.full(self.num_perm, self.p, dtype=np.int64)
        hs = np.array(
            [int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) for s in sh],
            dtype=np.int64,
        )
        # (a*h + b) mod p  -> ma trận (len(hs), num_perm)
        M = (np.outer(hs, self.a) + self.b) % self.p
        return M.min(axis=0)


@dataclass
class DupPair:
    id_a: str
    id_b: str
    jaccard: float
    containment_a_in_b: float
    containment_b_in_a: float


def find_duplicates(
    docs: Dict[str, str],
    *,
    k: int = 5,
    num_perm: int = 128,
    bands: int = 32,
    threshold: float = 0.5,
) -> List[DupPair]:
    """Trả về các cặp bài nghi trùng, đã xác nhận bằng Jaccard thật."""
    ids = list(docs.keys())
    sh = {i: shingles(docs[i], k) for i in ids}
    mh = MinHasher(num_perm)
    sigs = {i: mh.signature(sh[i]) for i in ids}

    rows = max(1, num_perm // bands)
    buckets: Dict[Tuple[int, bytes], List[str]] = defaultdict(list)
    for i in ids:
        s = sigs[i]
        for b in range(bands):
            band = s[b * rows:(b + 1) * rows].tobytes()
            buckets[(b, band)].append(i)

    cand: Set[Tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) < 2 or len(members) > 200:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                cand.add(tuple(sorted((members[x], members[y]))))  # type: ignore

    out: List[DupPair] = []
    for a, b in cand:
        j = jaccard(sh[a], sh[b])
        if j >= threshold:
            out.append(
                DupPair(a, b, round(j, 4),
                        round(containment(sh[a], sh[b]), 4),
                        round(containment(sh[b], sh[a]), 4))
            )
    return sorted(out, key=lambda d: -d.jaccard)


def dup_ratio_per_doc(docs: Dict[str, str], pairs: Sequence[DupPair]) -> Dict[str, float]:
    """Mức trùng lặp cao nhất của mỗi bài với bất kỳ bài nào khác."""
    r = {i: 0.0 for i in docs}
    for p in pairs:
        r[p.id_a] = max(r[p.id_a], p.containment_a_in_b)
        r[p.id_b] = max(r[p.id_b], p.containment_b_in_a)
    return r


def exact_duplicate_groups(docs: Dict[str, str]) -> Dict[str, List[str]]:
    g: Dict[str, List[str]] = defaultdict(list)
    for i, t in docs.items():
        key = hashlib.sha1(" ".join(t.lower().split()).encode("utf-8")).hexdigest()
        g[key].append(i)
    return {k: v for k, v in g.items() if len(v) > 1}


def prompt_copy_ratio(essay: str, prompt: str, k: int = 4) -> float:
    """Tỉ lệ nội dung bài làm là chép lại đề bài."""
    if not prompt or not prompt.strip():
        return 0.0
    return containment(shingles(essay, k), shingles(prompt, k))

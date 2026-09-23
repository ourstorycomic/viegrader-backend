"""Pipeline làm sạch dữ liệu đầu-cuối.

Vào : DataFrame thô (cột tối thiểu: essay_id, text; tuỳ chọn: prompt_id,
      prompt_text, course_id, student_id, score/nhãn có sẵn)
Ra  : (df_clean, df_rejected, report)

Bảy bước, chạy theo thứ tự và ghi vết đầy đủ:
    B1. Chuẩn hoá cấu trúc bảng, ép kiểu, sinh essay_id nếu thiếu
    B2. Giả danh hoá (PII) + cắt phần đầu bài chứa họ tên, MSSV
    B3. Chuẩn hoá văn bản tiếng Việt (Unicode, dấu thanh, teencode, dấu câu)
    B4. Kiểm định chất lượng -> keep / flag / reject
    B5. Phát hiện trùng lặp (tuyệt đối + gần đúng) trong cùng đề bài
    B6. Tính tỉ lệ chép đề bài
    B7. Xuất báo cáo tiền xử lí
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import dedup, pii
from .normalize import clean_text, count_teencode, diacritic_ratio
from .quality import KEEP, REJECT, QualityConfig, assess

REQUIRED_COLS = ["essay_id", "text"]


@dataclass
class CleanConfig:
    secret: str = "viegrader-default-key"
    anonymize: bool = True
    strip_header: bool = True
    tone_style: str = "new"
    expand_teencode: bool = True
    drop_exact_duplicates: bool = True
    dup_threshold: float = 0.50
    shingle_k: int = 5
    quality: QualityConfig = field(default_factory=QualityConfig)


@dataclass
class CleanReport:
    n_input: int = 0
    n_output: int = 0
    n_rejected: int = 0
    n_flagged: int = 0
    reject_reasons: Dict[str, int] = field(default_factory=dict)
    normalize_counts: Dict[str, int] = field(default_factory=dict)
    pii_counts: Dict[str, int] = field(default_factory=dict)
    exact_dup_groups: int = 0
    near_dup_pairs: int = 0
    dup_pairs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["dup_pairs"] = self.dup_pairs[:200]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def summary(self) -> str:
        lines = [
            "===== BÁO CÁO LÀM SẠCH DỮ LIỆU =====",
            f"Số bản ghi đầu vào      : {self.n_input}",
            f"Số bản ghi giữ lại      : {self.n_output}",
            f"Số bản ghi bị loại      : {self.n_rejected}",
            f"Số bản ghi gắn cờ       : {self.n_flagged}",
            f"Nhóm trùng tuyệt đối    : {self.exact_dup_groups}",
            f"Cặp trùng gần đúng      : {self.near_dup_pairs}",
        ]
        if self.reject_reasons:
            lines.append("Lí do loại:")
            for k, v in sorted(self.reject_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"   - {k}: {v}")
        if self.normalize_counts:
            lines.append("Can thiệp chuẩn hoá:")
            for k, v in sorted(self.normalize_counts.items(), key=lambda x: -x[1]):
                lines.append(f"   - {k}: {v}")
        if self.pii_counts:
            lines.append("Thông tin cá nhân đã che:")
            for k, v in sorted(self.pii_counts.items(), key=lambda x: -x[1]):
                lines.append(f"   - {k}: {v}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def clean_dataframe(
    df: pd.DataFrame, cfg: Optional[CleanConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, CleanReport]:
    cfg = cfg or CleanConfig()
    rep = CleanReport(n_input=len(df))
    df = df.copy()

    # ---- B1. Chuẩn hoá cấu trúc bảng ------------------------------------- #
    if "text" not in df.columns:
        for alt in ("essay", "bai_lam", "noi_dung", "content", "body"):
            if alt in df.columns:
                df = df.rename(columns={alt: "text"})
                break
    if "text" not in df.columns:
        raise ValueError("Bảng đầu vào phải có cột 'text' (hoặc essay/bai_lam/noi_dung).")
    if "essay_id" not in df.columns:
        df["essay_id"] = [f"E{uuid.uuid4().hex[:10]}" for _ in range(len(df))]
    df["essay_id"] = df["essay_id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["raw_text"] = df["text"]
    for col in ("prompt_id", "prompt_text", "course_id"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # ---- B2. Giả danh hoá ------------------------------------------------ #
    student_hashes: List[str] = []
    for i, row in df.iterrows():
        t = row["text"]
        sid_raw = str(row.get("student_id", "") or "")
        if cfg.anonymize:
            t, pr = pii.anonymize(t, cfg.secret)
            for k, v in pr.counts.items():
                rep.pii_counts[k] = rep.pii_counts.get(k, 0) + v
            if not sid_raw and pr.student_id_found:
                sid_raw = pr.student_id_found
        if cfg.strip_header:
            t = pii.strip_header(t)
        df.at[i, "text"] = t
        student_hashes.append(pii.hash_id(sid_raw, cfg.secret) if sid_raw else "")
    df["student_hash"] = student_hashes
    if "student_id" in df.columns:
        df = df.drop(columns=["student_id"])

    # ---- B3. Chuẩn hoá văn bản ------------------------------------------- #
    teencode_rates, dia_ratios = [], []
    for i, row in df.iterrows():
        raw = row["text"]
        n_tok = max(1, len(raw.split()))
        teencode_rates.append(count_teencode(raw) / n_tok)
        t, nrep = clean_text(
            raw, tone_style=cfg.tone_style, do_teencode=cfg.expand_teencode
        )
        for k, v in nrep.as_dict().items():
            rep.normalize_counts[k] = rep.normalize_counts.get(k, 0) + v
        df.at[i, "text"] = t
        dia_ratios.append(round(diacritic_ratio(t), 4))
    df["teencode_rate"] = [round(x, 5) for x in teencode_rates]
    df["diacritic_ratio"] = dia_ratios

    # ---- B4. Kiểm định chất lượng ---------------------------------------- #
    decisions, reasons_col, qstats = [], [], []
    for t in df["text"]:
        v = assess(t, cfg.quality)
        decisions.append(v.decision)
        reasons_col.append("; ".join(v.reasons))
        qstats.append(v.stats)
    df["qc_decision"] = decisions
    df["qc_reasons"] = reasons_col
    qdf = pd.DataFrame(qstats, index=df.index)
    for c in qdf.columns:
        if c not in df.columns:
            df[c] = qdf[c]

    for d, r in zip(decisions, reasons_col):
        if d == REJECT:
            for reason in [x.strip() for x in r.split(";") if x.strip()]:
                key = reason.split("(")[0].strip()
                rep.reject_reasons[key] = rep.reject_reasons.get(key, 0) + 1
    rep.n_flagged = int((df["qc_decision"] == "flag").sum())

    rejected = df[df["qc_decision"] == REJECT].copy()
    kept = df[df["qc_decision"] != REJECT].copy()

    # ---- B5. Trùng lặp (trong phạm vi cùng prompt_id) --------------------- #
    kept["dup_ratio"] = 0.0
    kept["dup_with"] = ""
    all_pairs: List[dedup.DupPair] = []
    exact_groups_total = 0

    for pid, grp in kept.groupby("prompt_id"):
        docs = dict(zip(grp["essay_id"], grp["text"]))
        if len(docs) < 2:
            continue
        exact = dedup.exact_duplicate_groups(docs)
        exact_groups_total += len(exact)
        if cfg.drop_exact_duplicates:
            for _, members in exact.items():
                for dupe in members[1:]:
                    kept.loc[kept["essay_id"] == dupe, "qc_decision"] = REJECT
                    kept.loc[kept["essay_id"] == dupe, "qc_reasons"] = (
                        f"Trùng tuyệt đối với {members[0]}"
                    )
        pairs = dedup.find_duplicates(
            docs, k=cfg.shingle_k, threshold=cfg.dup_threshold
        )
        all_pairs.extend(pairs)
        ratios = dedup.dup_ratio_per_doc(docs, pairs)
        partner: Dict[str, str] = {}
        for p in pairs:
            partner.setdefault(p.id_a, p.id_b)
            partner.setdefault(p.id_b, p.id_a)
        for eid, r in ratios.items():
            kept.loc[kept["essay_id"] == eid, "dup_ratio"] = round(r, 4)
            if eid in partner:
                kept.loc[kept["essay_id"] == eid, "dup_with"] = partner[eid]

    rep.exact_dup_groups = exact_groups_total
    rep.near_dup_pairs = len(all_pairs)
    rep.dup_pairs = [p.__dict__ for p in all_pairs]

    newly_rejected = kept[kept["qc_decision"] == REJECT]
    if len(newly_rejected):
        rejected = pd.concat([rejected, newly_rejected], ignore_index=True)
        kept = kept[kept["qc_decision"] != REJECT].copy()
        rep.reject_reasons["Trùng tuyệt đối"] = (
            rep.reject_reasons.get("Trùng tuyệt đối", 0) + len(newly_rejected)
        )

    # ---- B6. Tỉ lệ chép đề bài ------------------------------------------- #
    kept["prompt_copy_ratio"] = [
        round(dedup.prompt_copy_ratio(t, p), 4)
        for t, p in zip(kept["text"], kept["prompt_text"])
    ]

    # ---- B7. Hoàn tất ----------------------------------------------------- #
    kept = kept.reset_index(drop=True)
    rejected = rejected.reset_index(drop=True)
    rep.n_output = len(kept)
    rep.n_rejected = len(rejected)
    return kept, rejected, rep

"""Bộ chuyển đổi dữ liệu cho nghiên cứu chấm bài luận tiếng Việt.

Nguồn được chọn là Kaggle ``vokhoa/vietnamese-it-essays-for-aes-research``.
Module này chỉ chuẩn hóa dữ liệu thật có trong gói tải về; tuyệt đối không sinh
điểm giả hoặc coi một cột điểm không rõ nguồn gốc là điểm của hai giám khảo.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ..io_utils import read_table, write_json, write_table

SELECTED_DATASET_SLUG = "vokhoa/vietnamese-it-essays-for-aes-research"

COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "essay_id": ("essay_id", "essayid", "id", "bai_id", "ma_bai"),
    "text": ("text", "essay", "essay_text", "content", "body", "bai_lam", "noi_dung"),
    "prompt_id": ("prompt_id", "promptid", "topic_id", "question_id", "de_id", "ma_de"),
    "prompt_text": ("prompt_text", "prompt", "topic", "question", "de_bai", "cau_hoi"),
    "student_id": ("student_id", "studentid", "student", "mssv", "ma_sv"),
    "course_id": ("course_id", "course", "subject", "mon_hoc", "ma_hoc_phan"),
}

TRAIT_ALIASES: Dict[str, Sequence[str]] = {
    "noi_dung": ("noi_dung", "content_score", "content", "task_response"),
    "lap_luan": ("lap_luan", "argumentation_score", "argumentation", "reasoning"),
    "to_chuc": ("to_chuc", "organization_score", "organization", "coherence"),
    "tu_vung": ("tu_vung", "vocabulary_score", "vocabulary", "lexical_resource"),
    "ngu_phap": ("ngu_phap", "grammar_score", "grammar", "syntax"),
    "quy_uoc": ("quy_uoc", "mechanics_score", "mechanics", "conventions"),
}

SUPPORTED_TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".xlsm", ".json", ".jsonl", ".parquet"}


def _norm(name: object) -> str:
    value = str(name).strip().lower()
    value = re.sub(r"[^0-9a-zA-Z_\u00c0-\u024f\u1e00-\u1eff]+", "_", value)
    return value.strip("_")


def _rename_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for mapping in (COLUMN_ALIASES, TRAIT_ALIASES):
        normalized = {_norm(c): c for c in d.columns}
        rename: Dict[object, str] = {}
        used = set()
        for canonical, aliases in mapping.items():
            if canonical in d.columns:
                continue
            for alias in aliases:
                original = normalized.get(_norm(alias))
                if original is not None and original not in used:
                    rename[original] = canonical
                    used.add(original)
                    break
        d = d.rename(columns=rename)
    return d


def _candidate_tables(source_dir: Path) -> Iterable[Tuple[Path, pd.DataFrame]]:
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_TABLE_EXTENSIONS:
            continue
        try:
            frame = _rename_known_columns(read_table(path))
        except Exception:
            continue
        if "text" in frame.columns:
            yield path, frame


def _fingerprint(df: pd.DataFrame, columns: Sequence[str]) -> str:
    h = hashlib.sha256()
    for row in df[list(columns)].fillna("").astype(str).itertuples(index=False, name=None):
        h.update("\x1f".join(row).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def prepare_vietnamese_it_dataset(source_dir: str | Path, output_dir: str | Path) -> dict:
    """Chuẩn hóa gói Kaggle đã giải nén và xuất báo cáo kiểm toán.

    Nếu nguồn không chứa đủ sáu nhãn, chỉ xuất bài luận để tổ chức chấm kép;
    hàm không tự suy diễn hay sinh nhãn thay thế.
    """
    root, out = Path(source_dir), Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục dataset: {root}")
    tables = list(_candidate_tables(root))
    if not tables:
        raise ValueError("Không tìm thấy bảng có cột nội dung bài luận trong gói dataset.")

    frames: List[pd.DataFrame] = []
    files: List[dict] = []
    for path, frame in tables:
        d = frame.copy()
        d["source_file"] = str(path.relative_to(root))
        frames.append(d)
        files.append({"path": str(path.relative_to(root)), "rows": int(len(d)),
                      "columns": [str(c) for c in d.columns if c != "source_file"]})
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    data = data[data["text"].ne("")].copy()
    if "essay_id" not in data.columns:
        data["essay_id"] = [f"VIT{i:06d}" for i in range(1, len(data) + 1)]
    else:
        data["essay_id"] = data["essay_id"].fillna("").astype(str)
        missing = data["essay_id"].eq("")
        data.loc[missing, "essay_id"] = [f"VIT{i:06d}" for i in range(1, int(missing.sum()) + 1)]
    if data["essay_id"].duplicated().any():
        data["essay_id"] = [f"VIT{i:06d}" for i in range(1, len(data) + 1)]
    for col in ("prompt_id", "prompt_text", "student_id", "course_id"):
        if col not in data.columns:
            data[col] = ""

    core = ["essay_id", "text", "prompt_id", "prompt_text", "student_id", "course_id", "source_file"]
    extras = [c for c in data.columns if c not in core]
    data = data[core + extras].reset_index(drop=True)
    trait_cols = [key for key in TRAIT_ALIASES if key in data.columns]
    complete_traits = len(trait_cols) == len(TRAIT_ALIASES)

    out.mkdir(parents=True, exist_ok=True)
    write_table(data, out / "essays_raw.csv")
    if complete_traits:
        label_cols = ["essay_id", *TRAIT_ALIASES.keys()]
        labels = data[label_cols].copy()
        labels["total"] = labels[list(TRAIT_ALIASES)].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        write_table(labels, out / "labels_unverified.csv")

    report = {
        "selected_source": SELECTED_DATASET_SLUG,
        "source_role": "Vietnamese domain corpus; human double-scoring is required for article evaluation",
        "n_rows": int(len(data)),
        "n_files": len(files),
        "files": files,
        "recognized_trait_columns": trait_cols,
        "labels_status": "unverified_existing_labels" if complete_traits else "human_labeling_required",
        "dataset_sha256": _fingerprint(data, ["essay_id", "text", "prompt_id"]),
        "warnings": [
            "Xác minh giấy phép và nguồn gốc bài làm trước khi công bố hoặc huấn luyện.",
            "Không dùng labels_unverified.csv làm gold trước khi xác minh quy trình chấm.",
            "Cần ít nhất hai giám khảo độc lập trên evaluation set để tính human benchmark.",
        ],
    }
    write_json(report, out / "dataset_audit.json")
    return report


def split_research_dataset(
    essays: pd.DataFrame,
    gold: pd.DataFrame,
    output_dir: str | Path,
    group_column: str = "student_hash",
    test_size: float = 0.20,
    validation_size: float = 0.10,
    seed: int = 42,
    holdout_prompt: str | None = None,
) -> dict:
    """Chia train/validation/test không rò rỉ theo người học.

    ``validation_size`` và ``test_size`` là tỉ lệ trên toàn bộ dữ liệu. Khi có
    ``holdout_prompt``, toàn bộ đề đó được khóa làm test ngoài đề.
    """
    if not 0 < test_size < 1 or not 0 <= validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError("Cần 0 < test_size, 0 <= validation_size và tổng < 1.")
    if "essay_id" not in essays or "essay_id" not in gold:
        raise ValueError("essays và gold phải có cột essay_id.")
    joined = essays.merge(gold, on="essay_id", how="inner", suffixes=("", "_gold"))
    if len(joined) < 20:
        raise ValueError("Cần ít nhất 20 bài có nhãn để chia dữ liệu nghiên cứu.")
    if group_column not in joined or joined[group_column].fillna("").astype(str).eq("").all():
        group_column = "essay_id"
    groups = joined[group_column].fillna("").astype(str)
    groups = groups.mask(groups.eq(""), joined["essay_id"].astype(str))

    split = pd.Series("", index=joined.index, dtype="object")
    if holdout_prompt:
        if "prompt_id" not in joined:
            raise ValueError("Không có prompt_id để tạo kiểm tra ngoài đề.")
        prompt_mask = joined["prompt_id"].astype(str).eq(str(holdout_prompt))
        held_out_groups = set(groups.loc[prompt_mask])
        test_mask = groups.isin(held_out_groups)
        if not test_mask.any():
            raise ValueError(f"Không thấy holdout prompt {holdout_prompt!r}.")
        split.loc[test_mask] = "test"
        remaining = joined.index[~test_mask].to_numpy()
        rel_val = validation_size / (1.0 - test_mask.mean())
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        remaining_pos, test_pos = next(splitter.split(joined, groups=groups))
        split.iloc[test_pos] = "test"
        remaining = joined.index[remaining_pos].to_numpy()
        rel_val = validation_size / (1.0 - test_size)

    if validation_size > 0:
        sub = joined.loc[remaining]
        sub_groups = groups.loc[remaining]
        splitter = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
        train_pos, val_pos = next(splitter.split(sub, groups=sub_groups))
        split.loc[sub.index[val_pos]] = "validation"
        split.loc[sub.index[train_pos]] = "train"
    else:
        split.loc[remaining] = "train"
    if split.eq("").any():
        raise RuntimeError("Có bản ghi chưa được gán split.")
    joined["split"] = split

    essay_cols = [c for c in essays.columns if c in joined.columns and c != "split"] + ["split"]
    gold_cols = [c for c in gold.columns if c in joined.columns and c != "split"] + ["split"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        part = joined[joined["split"].eq(name)]
        write_table(part[essay_cols], out / f"essays_{name}.csv")
        write_table(part[gold_cols], out / f"gold_{name}.csv")
    write_table(joined[["essay_id", "split"]], out / "split_manifest.csv")

    leakage = 0
    if group_column != "essay_id":
        group_split_counts = joined.assign(_group=groups).groupby("_group")["split"].nunique()
        leakage = int((group_split_counts > 1).sum())
    report = {
        "seed": seed,
        "group_column": group_column,
        "holdout_prompt": holdout_prompt,
        "counts": {k: int(v) for k, v in joined["split"].value_counts().to_dict().items()},
        "group_leakage_count": leakage,
        "split_sha256": _fingerprint(joined.sort_values("essay_id"), ["essay_id", "split"]),
    }
    write_json(report, out / "split_report.json")
    return report

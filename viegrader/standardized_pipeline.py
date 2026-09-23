"""Pipeline IT04 cho bộ dữ liệu chuẩn VieGrader và chấm lai theo rubric.

Nhãn huấn luyện của bộ IT04 là điểm tổng. Điểm từng câu do rule engine dưới
đây ước lượng từ đáp án/rubric và luôn được đánh dấu ``rule_based_estimate``;
chúng không bao giờ được dùng như nhãn vàng khi huấn luyện.
"""

from __future__ import annotations

import html
import json
import math
import pickle
import re
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .discrete_math import _metrics, _model_text, train_total_baseline


SCHEMA = "viegrader.it04.standardized/v1"
EXAM_ALIASES = {
    "de01": "De_1", "de1": "De_1", "de_1": "De_1",
    "de02": "De_2", "de2": "De_2", "de_2": "De_2",
    "de03": "De_3", "de3": "De_3", "de_3": "De_3",
    "de04": "De_4", "de4": "De_4", "de_4": "De_4",
    "de05": "De_5", "de5": "De_5", "de_5": "De_5",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _first(record: Mapping[str, Any], names: Sequence[str], default: Any = "") -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return default


def normalize_exam(value: Any) -> str:
    raw = re.sub(r"[^a-z0-9_]", "", str(value or "").lower())
    if raw in EXAM_ALIASES:
        return EXAM_ALIASES[raw]
    match = re.search(r"([1-5])", raw)
    return f"De_{match.group(1)}" if match else "UNKNOWN"


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "co", "có"}


def _iter_jsonl(source: str | Path) -> Iterator[Dict[str, Any]]:
    path = Path(source)
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".jsonl")]
            preferred = [n for n in names if "dataset" in Path(n).name.lower()]
            if not names:
                raise ValueError("ZIP không chứa tệp JSONL.")
            name = sorted(preferred or names, key=lambda n: (n.count("/"), len(n)))[0]
            with archive.open(name) as handle:
                for number, raw in enumerate(handle, 1):
                    if raw.strip():
                        try:
                            yield json.loads(raw.decode("utf-8-sig"))
                        except Exception as exc:
                            raise ValueError(f"JSONL lỗi tại dòng {number}: {exc}") from exc
        return
    if path.is_dir():
        candidates = list(path.glob("dataset/*.jsonl")) + list(path.glob("*.jsonl"))
        candidates += list(path.glob("**/viegrader_dataset.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"Không tìm thấy JSONL trong {path}")
        path = sorted(set(candidates), key=lambda p: (len(p.parts), len(p.name)))[0]
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as exc:
                    raise ValueError(f"JSONL lỗi tại dòng {number}: {exc}") from exc


def _record_flags(record: Mapping[str, Any]) -> List[str]:
    flags = record.get("flags", [])
    if isinstance(flags, Mapping):
        return sorted(str(k) for k, v in flags.items() if _bool(v))
    if isinstance(flags, str):
        return [x.strip() for x in re.split(r"[,;|]", flags) if x.strip()]
    return sorted(str(x) for x in (flags or []))


def adapt_record(record: Mapping[str, Any], index: int) -> Dict[str, Any]:
    labels = record.get("labels") if isinstance(record.get("labels"), Mapping) else {}
    eligibility = record.get("eligibility") if isinstance(record.get("eligibility"), Mapping) else {}
    score = _first(labels, ("score_10", "gold_total", "total"), None)
    if score is None:
        score = _first(record, ("score_10", "gold_total", "total", "score"), None)
    try:
        score = float(score) if score not in (None, "") else None
    except (TypeError, ValueError):
        score = None
    split = str(_first(record, ("split", "dataset_split"), "unassigned")).lower()
    split = {"val": "validation", "dev": "validation"}.get(split, split)
    flags = _record_flags(record)
    training_eligible = _bool(
        _first(record, ("training_eligible", "train_eligible"), eligibility.get("training", None)),
        default=score is not None and split == "train",
    )
    ready = _bool(
        _first(record, ("viegrader_ready", "viegrader_execution_ready", "execution_ready",
                        "ready_for_viegrader", "ready"), eligibility.get("viegrader", None)),
        default=score is not None and split in {"train", "validation", "test"},
    )
    return {
        "essay_id": str(_first(record, ("essay_id", "record_id", "sample_id", "submission_id", "id"), f"IT04-{index:06d}")),
        "text": normalize_text(_first(record, ("answer_text", "text_clean", "response_text", "text", "content", "essay"), "")),
        "exam_id": normalize_exam(_first(record, ("exam_code", "exam_code_final", "exam_id", "test_code", "de_thi"), "")),
        "split": split,
        "gold_total": score,
        "grade_band": str(_first(labels, ("grade_band",), _first(record, ("grade_band",), ""))),
        "training_eligible": training_eligible,
        "viegrader_ready": ready,
        "flags": flags,
        "source_record": record,
    }


def _rubric_files(rubric_dir: str | Path) -> Dict[str, Path]:
    root = Path(rubric_dir)
    result: Dict[str, Path] = {}
    for path in sorted(root.glob("*.yaml")):
        exam = normalize_exam(path.stem)
        if exam != "UNKNOWN":
            result[exam] = path
    missing = [f"De_{i}" for i in range(1, 6) if f"De_{i}" not in result]
    if missing:
        raise FileNotFoundError(f"Thiếu rubric cho: {', '.join(missing)}")
    return result


def _rubric_prompt_and_key(path: Path) -> tuple[str, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    prompt = str(raw.get("prompt", raw.get("meta", {}).get("applies_to", raw.get("name", ""))))
    key = "\n".join(
        f"{c.get('key')}: {c.get('description', '')}" for c in raw.get("criteria", [])
    )
    return prompt, key


def prepare_standardized_dataset(
    source: str | Path, rubric_dir: str | Path, output_dir: str | Path,
    *, min_chars: int = 20,
) -> Dict[str, Any]:
    """Chuyển JSONL chuẩn hóa sang hai bảng tương thích VieGrader 0.6+.

    Các split/nhãn/cờ loại do bộ chuẩn hóa cung cấp được giữ nguyên. Hàm không
    tự chia lại dữ liệu và không đưa bản trùng/xung đột trở lại tập train.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rubrics = _rubric_files(rubric_dir)
    prompt_key = {exam: _rubric_prompt_and_key(path) for exam, path in rubrics.items()}
    adapted = [adapt_record(r, i) for i, r in enumerate(_iter_jsonl(source), 1)]
    seen: Counter[str] = Counter(r["essay_id"] for r in adapted)
    rows, rejected, issues = [], [], Counter()
    for row in adapted:
        reasons: List[str] = []
        if seen[row["essay_id"]] > 1:
            reasons.append("duplicate_essay_id")
        if len(row["text"]) < min_chars:
            reasons.append("text_too_short")
        if row["exam_id"] == "UNKNOWN":
            reasons.append("unknown_exam")
        if row["split"] not in {"train", "validation", "test", "excluded_duplicate", "excluded_conflict"}:
            reasons.append("invalid_split")
        if row["gold_total"] is None or not 0 <= row["gold_total"] <= 10:
            reasons.append("invalid_or_missing_score")
        for reason in reasons:
            issues[reason] += 1
        if reasons or not (row["training_eligible"] or row["viegrader_ready"]):
            rejected.append({
                "essay_id": row["essay_id"], "exam_id": row["exam_id"],
                "split": row["split"], "reasons": "|".join(reasons or ["not_eligible_or_ready"]),
                "flags": "|".join(row["flags"]),
            })
            continue
        prompt, answer_key = prompt_key[row["exam_id"]]
        rows.append({
            "essay_id": row["essay_id"], "text": row["text"], "exam_id": row["exam_id"],
            "prompt_id": row["exam_id"], "prompt_text": prompt, "answer_key": answer_key,
            "rubric_file": rubrics[row["exam_id"]].name, "course_id": "IT04",
            "student_hash": row["essay_id"], "dataset_status": "keep",
            "split": row["split"], "gold_total": row["gold_total"],
            "grade_band": row["grade_band"], "training_eligible": row["training_eligible"],
            "viegrader_ready": row["viegrader_ready"], "flags": "|".join(row["flags"]),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Không có bản ghi hợp lệ sau kiểm tra schema/chất lượng.")
    # Không cho cờ training_eligible làm rò dữ liệu ngoài train.
    frame["training_eligible"] = frame["training_eligible"] & frame["split"].eq("train")
    essays_cols = [
        "essay_id", "text", "exam_id", "prompt_id", "prompt_text", "answer_key",
        "rubric_file", "course_id", "student_hash", "dataset_status", "split",
        "training_eligible", "viegrader_ready", "flags",
    ]
    gold_cols = ["essay_id", "gold_total", "grade_band", "exam_id", "split",
                 "training_eligible", "viegrader_ready", "flags"]
    frame[essays_cols].to_csv(out / "essays_split.csv", index=False, encoding="utf-8-sig")
    gold = frame[gold_cols].copy()
    gold["teacher_feedback"] = ""
    gold["label_granularity"] = "total_only"
    gold["label_source"] = "standardized_gradebook"
    gold.to_csv(out / "gold_split.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rejected, columns=["essay_id", "exam_id", "split", "reasons", "flags"]).to_csv(
        out / "rejected_or_quarantined.csv", index=False, encoding="utf-8-sig"
    )
    clean_records = frame.to_dict("records")
    with (out / "clean_dataset.jsonl").open("w", encoding="utf-8") as handle:
        for record in clean_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "schema": SCHEMA, "created_at": _now(), "source": str(source),
        "n_input": len(adapted), "n_prepared": len(frame),
        "n_ready": int(frame["viegrader_ready"].sum()), "n_rejected": len(rejected),
        "n_training_eligible": int(frame["training_eligible"].sum()),
        "counts_by_split": frame["split"].value_counts().sort_index().to_dict(),
        "counts_by_exam": frame["exam_id"].value_counts().sort_index().to_dict(),
        "score_summary": frame.groupby("split")["gold_total"].agg(["count", "mean", "std", "min", "max"]).fillna(0).round(4).to_dict("index"),
        "issues": dict(issues), "label_granularity": "total_only",
        "leakage_guard": "Giữ split đã khóa; chỉ training_eligible=true và split=train được train.",
    }
    _dump(out / "data_audit.json", report)
    return report


def validate_rubrics(rubric_dir: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    files = _rubric_files(rubric_dir)
    rows, problems = [], []
    for exam, path in files.items():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        criteria = raw.get("criteria", [])
        for criterion in criteria:
            rows.append({
                "exam_id": exam, "rubric_file": path.name, "criterion": criterion.get("key"),
                "name": criterion.get("name"), "max_score": criterion.get("max_score"),
                "weight": criterion.get("weight"), "answer": _expected_answer(criterion),
            })
        if len(criteria) != 10:
            problems.append(f"{exam}: cần 10 tiêu chí, hiện có {len(criteria)}")
        max_total = sum(float(c.get("max_score", 0)) for c in criteria)
        weight_total = sum(float(c.get("weight", 0)) for c in criteria)
        if not math.isclose(max_total, float(raw.get("scale_max", 10)), abs_tol=1e-8):
            problems.append(f"{exam}: tổng max_score={max_total}")
        if not math.isclose(weight_total, 1.0, abs_tol=1e-8):
            problems.append(f"{exam}: tổng weight={weight_total}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "rubric_items.csv", index=False, encoding="utf-8-sig")
    report = {"schema": "viegrader.rubric.inventory/v1", "created_at": _now(),
              "n_rubrics": len(files), "n_items": len(rows), "valid": not problems,
              "problems": problems, "label_note": "Điểm câu là ước lượng rule-based, không phải nhãn train."}
    _dump(out / "rubric_audit.json", report)
    if problems:
        raise ValueError("Rubric không hợp lệ: " + "; ".join(problems))
    return report


def train_standardized(
    prepared_dir: str | Path, output_dir: str | Path, *, alpha: float = 12.0,
) -> Dict[str, Any]:
    prepared, out = Path(prepared_dir), Path(output_dir)
    essays = pd.read_csv(prepared / "essays_split.csv")
    gold = pd.read_csv(prepared / "gold_split.csv")
    if "training_eligible" in essays:
        allowed = set(essays.loc[essays["training_eligible"].map(_bool), "essay_id"].astype(str))
        # Giữ validation/test để đánh giá, chỉ loại bản train không đủ điều kiện.
        essays = essays[~essays["split"].eq("train") | essays["essay_id"].astype(str).isin(allowed)]
        gold = gold[gold["essay_id"].astype(str).isin(set(essays["essay_id"].astype(str)))]
    staging = out / "_training_input"
    staging.mkdir(parents=True, exist_ok=True)
    essays.to_csv(staging / "essays_split.csv", index=False)
    gold.to_csv(staging / "gold_split.csv", index=False)
    report = train_total_baseline(staging / "essays_split.csv", staging / "gold_split.csv", out, alpha=alpha)
    report.update({
        "schema": "viegrader.it04.model/v2", "created_at": _now(),
        "label_granularity": "total_only", "n_training_eligible": int(essays["split"].eq("train").sum()),
        "data_leakage_check": "PASS: fit chỉ trên split=train",
    })
    _dump(out / "metrics.json", report)
    return report


def _expected_answer(criterion: Mapping[str, Any]) -> str:
    for key in ("accepted_answers", "answers", "answer", "expected_answer"):
        value = criterion.get(key)
        if value:
            return " | ".join(map(str, value)) if isinstance(value, list) else str(value)
    scoring = criterion.get("automated_scoring", criterion.get("scoring", {}))
    if isinstance(scoring, Mapping):
        for key in ("accepted_answers", "answers", "answer", "expected_answer"):
            value = scoring.get(key)
            if value:
                return " | ".join(map(str, value)) if isinstance(value, list) else str(value)
    description = str(criterion.get("description", ""))
    match = re.search(r"Đáp\s*số\s*chuẩn\s*:\s*(.+)", description, re.I)
    return match.group(1).strip() if match else description.strip()


def _math_norm(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).lower()
    value = value.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-")
    value = re.sub(r"\s+", "", value)
    return value.replace("^", "**")


def _answer_atoms(expected: str) -> List[str]:
    # Ưu tiên các con số/công thức đủ dài; tránh coi số thứ tự câu là đáp án.
    atoms = re.findall(r"\d+(?:[.,]\d+)?|\d+!|[a-z]&[a-z]", _math_norm(expected))
    atoms = [a for a in atoms if len(a) >= 2 or "!" in a]
    return list(dict.fromkeys(atoms)) or [_math_norm(expected)]


def split_answer_sections(text: str) -> Dict[str, str]:
    """Tách các đoạn ``cau_1a``/``cau_1`` từ tiêu đề bài làm."""
    body = normalize_text(text)
    pattern = re.compile(
        r"(?im)^\s*(?:câu|cau)\s*([1-9])\s*(?:[.\-:]?\s*([a-d]))?\s*[).:\-]?\s*"
    )
    matches = list(pattern.finditer(body))
    result: Dict[str, str] = {}
    for i, match in enumerate(matches):
        key = f"cau_{match.group(1)}{(match.group(2) or '').lower()}"
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        result[key] = body[match.end():end].strip()
    # Tách a), b)... bên trong một câu lớn.
    for top_key, section in list(result.items()):
        if re.fullmatch(r"cau_\d", top_key):
            sub = list(re.finditer(r"(?im)(?:^|\n)\s*([a-d])\s*[).:\-]\s*", section))
            for i, match in enumerate(sub):
                end = sub[i + 1].start() if i + 1 < len(sub) else len(section)
                result[top_key + match.group(1).lower()] = section[match.end():end].strip()
    return result


@dataclass
class ItemScore:
    criterion: str
    score: float
    max_score: float
    evidence: str
    match_type: str
    location: str
    confidence: float


def score_rubric_items(text: str, rubric_path: str | Path) -> tuple[List[ItemScore], Dict[str, Any]]:
    raw = yaml.safe_load(Path(rubric_path).read_text(encoding="utf-8")) or {}
    sections = split_answer_sections(text)
    items: List[ItemScore] = []
    for criterion in raw.get("criteria", []):
        key = str(criterion.get("key"))
        max_score = float(criterion.get("max_score", 0))
        top = re.match(r"cau_(\d)", key)
        segment = sections.get(key)
        location = key
        if not segment and top:
            location = f"cau_{top.group(1)}"
            segment = sections.get(location)
        if not segment:
            location, segment = "whole_answer_fallback", text
        expected = _expected_answer(criterion)
        atoms = _answer_atoms(expected)
        normalized = _math_norm(segment)
        hits = [atom for atom in atoms if atom and atom in normalized]
        ratio = len(hits) / max(1, len(atoms))
        alternatives = [_math_norm(v) for v in re.split(r"\s*\|\s*", expected) if v.strip()]
        exact_expression = any(len(v) >= 2 and v in normalized for v in alternatives)
        if exact_expression or ratio >= 0.999:
            score, kind = max_score, "exact_answer_atoms"
        elif len(atoms) > 1 and ratio >= 0.5:
            score, kind = max_score * 0.5, "partial_answer_atoms"
        else:
            # Chỉ cấp điểm phương pháp khi rubric khai báo từ khóa rõ ràng.
            scoring = criterion.get("automated_scoring", criterion.get("scoring", {}))
            keywords = criterion.get("keywords", [])
            if isinstance(scoring, Mapping):
                keywords = scoring.get("method_keywords", scoring.get("keywords", keywords))
            keywords = [str(k).lower() for k in (keywords or [])]
            method_hits = [k for k in keywords if k and k in segment.lower()]
            score = max_score * 0.5 if keywords and len(method_hits) >= max(1, math.ceil(len(keywords) / 2)) else 0.0
            kind = "method_keywords" if score else "no_verified_evidence"
            hits = method_hits
        localized = location != "whole_answer_fallback"
        confidence = 0.95 if kind == "exact_answer_atoms" and localized else 0.60 if localized else 0.25
        items.append(ItemScore(key, round(score, 4), max_score, "; ".join(hits), kind, location, confidence))
    coverage = sum(i.location != "whole_answer_fallback" for i in items) / max(1, len(items))
    total = sum(i.score for i in items)
    return items, {"rubric_total": round(total, 4), "section_coverage": round(coverage, 4),
                   "n_sections": len(sections), "scale_max": float(raw.get("scale_max", 10.0))}


def grade_with_rubric(
    model_path: str | Path, input_path: str | Path, rubric_dir: str | Path,
    output_dir: str | Path, *, rubric_weight: float = 0.70,
    min_section_coverage: float = 0.60,
) -> Dict[str, Any]:
    if not 0 <= rubric_weight <= 1:
        raise ValueError("rubric_weight phải thuộc [0, 1].")
    with open(model_path, "rb") as handle:
        artifact = pickle.load(handle)
    frame = pd.read_csv(input_path)
    required = {"essay_id", "text", "exam_id", "prompt_text"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Dữ liệu chấm thiếu cột: {sorted(required - set(frame.columns))}")
    if "viegrader_ready" in frame.columns:
        frame = frame[frame["viegrader_ready"].map(_bool)].copy()
    rubrics = _rubric_files(rubric_dir)
    model_pred = np.clip(artifact["model"].predict(_model_text(frame)), 0, 10)
    score_rows, item_rows, review_rows = [], [], []
    for row, predicted in zip(frame.itertuples(index=False), model_pred):
        exam = normalize_exam(row.exam_id)
        flags: List[str] = []
        if exam not in rubrics:
            flags.append("UNKNOWN_EXAM")
            items, meta = [], {"rubric_total": None, "section_coverage": 0.0}
        else:
            items, meta = score_rubric_items(str(row.text), rubrics[exam])
        empty = len(normalize_text(row.text)) < 20
        if empty:
            final, mode = 0.0, "empty_rule"
            flags.append("EMPTY_OR_TOO_SHORT")
        elif meta["section_coverage"] >= min_section_coverage:
            raw_final = rubric_weight * float(meta["rubric_total"]) + (1 - rubric_weight) * float(predicted)
            final, mode = round(raw_final / 0.25) * 0.25, "hybrid_rubric_model"
        else:
            final, mode = round(float(predicted) / 0.25) * 0.25, "model_fallback"
            flags.append("LOW_SECTION_COVERAGE")
        final = float(np.clip(final, 0, 10))
        disagreement = abs(float(predicted) - float(meta["rubric_total"])) if meta["rubric_total"] is not None else None
        if disagreement is not None and disagreement >= 2.0:
            flags.append("MODEL_RUBRIC_DISAGREEMENT")
        needs_review = bool(flags)
        score_rows.append({
            "essay_id": row.essay_id, "exam_id": exam, "model_total": round(float(predicted) / 0.25) * 0.25,
            "rubric_total": meta["rubric_total"], "final_total": final, "grading_mode": mode,
            "section_coverage": meta["section_coverage"], "needs_human_review": needs_review,
            "flags": "|".join(flags), "label_granularity": "total_only_for_training",
        })
        for item in items:
            item_rows.append({"essay_id": row.essay_id, "exam_id": exam, **item.__dict__,
                              "label_status": "rule_based_estimate"})
        if needs_review:
            review_rows.append(score_rows[-1])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(score_rows)
    scores.to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(item_rows).to_csv(out / "item_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(review_rows, columns=scores.columns).to_csv(out / "human_review_queue.csv", index=False, encoding="utf-8-sig")
    report = {
        "schema": "viegrader.it04.grading/v1", "created_at": _now(), "n_scored": len(scores),
        "n_human_review": len(review_rows), "review_rate": len(review_rows) / max(1, len(scores)),
        "grading_modes": scores["grading_mode"].value_counts().to_dict(),
        "mean_scores": scores[["model_total", "rubric_total", "final_total"]].mean().round(4).to_dict(),
        "rubric_weight": rubric_weight, "min_section_coverage": min_section_coverage,
        "warning": "Điểm câu là rule_based_estimate; cần giảng viên duyệt trước khi công bố.",
    }
    _dump(out / "grading_audit.json", report)
    return report


def calibrate_rubric_weight(
    model_path: str | Path, essays_path: str | Path, gold_path: str | Path,
    rubric_dir: str | Path, output_path: str | Path,
    *, min_section_coverage: float = 0.60,
) -> Dict[str, Any]:
    """Chọn trọng số điểm rubric chỉ trên validation, tuyệt đối không nhìn test."""
    with open(model_path, "rb") as handle:
        artifact = pickle.load(handle)
    essays, gold = pd.read_csv(essays_path), pd.read_csv(gold_path)
    validation = essays[essays["split"].eq("validation")].merge(
        gold[["essay_id", "gold_total"]], on="essay_id", validate="one_to_one"
    )
    if validation.empty:
        report = {"selected_weight": 0.0, "reason": "Không có validation; dùng model-only an toàn.",
                  "uses_test_labels": False, "candidates": []}
        _dump(Path(output_path), report)
        return report
    rubrics = _rubric_files(rubric_dir)
    model_pred = np.clip(artifact["model"].predict(_model_text(validation)), 0, 10)
    rubric_pred, eligible = [], []
    for row in validation.itertuples(index=False):
        exam = normalize_exam(row.exam_id)
        if exam in rubrics:
            _, meta = score_rubric_items(str(row.text), rubrics[exam])
        else:
            meta = {"rubric_total": 0.0, "section_coverage": 0.0}
        rubric_pred.append(float(meta["rubric_total"]))
        eligible.append(float(meta["section_coverage"]) >= min_section_coverage)
    rubric_pred, eligible = np.asarray(rubric_pred), np.asarray(eligible, dtype=bool)
    candidates = []
    for weight in np.linspace(0, 1, 11):
        raw = np.where(eligible, weight * rubric_pred + (1 - weight) * model_pred, model_pred)
        pred = np.round(np.clip(raw, 0, 10) / 0.25) * 0.25
        metrics = _metrics(validation["gold_total"], pred)
        candidates.append({"weight": round(float(weight), 2), **metrics})
    # MAE là tiêu chí chính; nếu bằng nhau ưu tiên trọng số thấp hơn.
    selected = min(candidates, key=lambda x: (float(x["MAE"]), float(x["weight"])))
    report = {
        "schema": "viegrader.it04.rubric-calibration/v1", "created_at": _now(),
        "selected_weight": selected["weight"], "selection_metric": "validation_MAE",
        "n_validation": len(validation), "n_hybrid_eligible": int(eligible.sum()),
        "uses_test_labels": False, "candidates": candidates,
    }
    _dump(Path(output_path), report)
    return report


def evaluate_grading(scores_path: str | Path, gold_path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    scores, gold = pd.read_csv(scores_path), pd.read_csv(gold_path)
    keep = [c for c in ("essay_id", "gold_total", "exam_id", "split") if c in gold]
    joined = gold[keep].merge(scores, on="essay_id", how="inner", validate="one_to_one", suffixes=("_gold", ""))
    report: Dict[str, Any] = {"schema": "viegrader.it04.evaluation/v1", "created_at": _now(), "metrics": {}}
    for split, part in joined.groupby("split") if "split" in joined else [("all", joined)]:
        report["metrics"][str(split)] = {
            "model_total": _metrics(part["gold_total"], part["model_total"]),
            "final_total": _metrics(part["gold_total"], part["final_total"]),
        }
    report["test_is_primary"] = "test" in report["metrics"]
    report["methodology_note"] = "Chỉ metric test đã khóa dùng làm kết quả tổng quát hóa."
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out / "scores_with_gold.csv", index=False, encoding="utf-8-sig")
    _dump(out / "evaluation.json", report)
    return report


def build_workflow_report(root_dir: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    root, out = Path(root_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_files = sorted(root.glob("**/*.json"))
    stages = []
    for path in json_files:
        if out in path.parents:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stages.append({"file": str(path.relative_to(root)), "payload": payload})
    manifest = {"schema": "viegrader.it04.workflow-report/v1", "created_at": _now(),
                "n_stage_reports": len(stages), "stages": stages}
    _dump(out / "workflow_summary.json", manifest)
    lines = ["# Báo cáo pipeline VieGrader IT04", "", f"Thời điểm UTC: `{manifest['created_at']}`", ""]
    for stage in stages:
        lines += [f"## {stage['file']}", "", "```json",
                  json.dumps(stage["payload"], ensure_ascii=False, indent=2), "```", ""]
    (out / "workflow_report.md").write_text("\n".join(lines), encoding="utf-8")
    cards = "".join(
        f"<section><h2>{html.escape(s['file'])}</h2><pre>{html.escape(json.dumps(s['payload'], ensure_ascii=False, indent=2))}</pre></section>"
        for s in stages
    )
    page = ("<!doctype html><meta charset='utf-8'><title>VieGrader IT04 report</title>"
            "<style>body{font:15px system-ui;max-width:1100px;margin:auto;padding:24px;background:#f6f8fa}"
            "section{background:white;padding:18px;margin:16px 0;border-radius:10px}pre{white-space:pre-wrap}</style>"
            f"<h1>Báo cáo pipeline VieGrader IT04</h1>{cards}")
    (out / "workflow_report.html").write_text(page, encoding="utf-8")
    return manifest


def run_all(
    source: str | Path, rubric_dir: str | Path, output_dir: str | Path,
    *, alpha: float = 12.0, rubric_weight: float | str = "auto",
) -> Dict[str, Any]:
    root = Path(output_dir)
    data_report = prepare_standardized_dataset(source, rubric_dir, root / "01_data")
    rubric_report = validate_rubrics(rubric_dir, root / "02_rubric")
    train_report = train_standardized(root / "01_data", root / "03_model", alpha=alpha)
    if str(rubric_weight).lower() == "auto":
        calibration = calibrate_rubric_weight(
            root / "03_model/total_baseline.pkl", root / "01_data/essays_split.csv",
            root / "01_data/gold_split.csv", rubric_dir, root / "03_model/rubric_calibration.json",
        )
        selected_weight = float(calibration["selected_weight"])
    else:
        selected_weight = float(rubric_weight)
        calibration = {"selected_weight": selected_weight, "selection_metric": "user_fixed",
                       "uses_test_labels": False}
        _dump(root / "03_model/rubric_calibration.json", calibration)
    grade_report = grade_with_rubric(
        root / "03_model/total_baseline.pkl", root / "01_data/essays_split.csv",
        rubric_dir, root / "04_grading", rubric_weight=selected_weight,
    )
    eval_report = evaluate_grading(
        root / "04_grading/scores.csv", root / "01_data/gold_split.csv", root / "05_evaluation"
    )
    workflow = build_workflow_report(root, root / "06_reports")
    return {"data": data_report, "rubric": rubric_report, "train": train_report,
            "calibration": calibration,
            "grading": grade_report, "evaluation": eval_report, "workflow": workflow}

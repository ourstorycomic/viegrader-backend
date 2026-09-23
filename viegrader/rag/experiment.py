"""Paired IT04 experiment: the same QLoRA adapter without and with local RAG.

Gold scores are opened only by `report`, after model inference finishes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ..discrete_math import TOTAL_SYSTEM_PROMPT, _total_user_prompt, score_total_qlora
from .index import TfidfRAGIndex


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tokens(tokenizer, row: dict) -> int:
    messages = [{"role": "system", "content": TOTAL_SYSTEM_PROMPT},
                {"role": "user", "content": _total_user_prompt(SimpleNamespace(**row))}]
    return len(tokenizer.apply_chat_template(messages, add_generation_prompt=True))


def prepare(essays_path: Path, index_path: Path, output: Path, model: str,
            *, max_tokens: int = 4608, top_k: int = 3, course_id: str = "IT04",
            expected_test: int = 73) -> dict:
    if max_tokens < 512 or top_k < 1:
        raise ValueError("max_tokens phải >=512 và top_k phải >=1")
    for path, description in ((essays_path, "bài làm"), (index_path, "chỉ mục RAG")):
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp {description}: {path}. Hãy đặt đường dẫn thực tế.")
    frame = pd.read_csv(essays_path).fillna("")
    required = {"essay_id", "text", "prompt_text", "answer_key", "exam_id"}
    if not required.issubset(frame):
        raise ValueError(f"Thiếu cột đầu vào: {sorted(required - set(frame))}")
    if "gold_total" in frame:
        raise ValueError("Tệp bài làm không được chứa gold_total")
    if "split" not in frame:
        raise ValueError("Thiếu cột split; cần tách test trước khi chạy")
    frame = frame[frame["split"].astype(str).eq("test")].copy()
    if frame.empty or frame["essay_id"].astype(str).duplicated().any():
        raise ValueError("Tập test rỗng hoặc trùng essay_id")
    if expected_test > 0 and len(frame) != expected_test:
        raise ValueError(f"Tập test có {len(frame)} bài, cần {expected_test} bài theo cấu hình. "
                         "Kiểm tra đúng bộ dữ liệu; chỉ dùng --expected-test để chạy bộ khác có chủ đích.")
    from transformers import AutoTokenizer

    index = TfidfRAGIndex.from_documents(index_path)
    tok = AutoTokenizer.from_pretrained(model, use_fast=True)
    plain, grounded, retrievals, coverage = [], [], [], []
    for row in frame.to_dict("records"):
        eid = str(row["essay_id"])
        base = {k: str(row[k]) for k in required}
        base["essay_id"] = eid
        base_tokens = _tokens(tok, base)
        hits = index.search(f"{base['prompt_text']}\n{base['text']}", top_k=top_k,
                            course_id=course_id)
        selected, used = [], []
        if base_tokens <= max_tokens:
            for hit in hits:
                passage = (f"[{hit['document_id']}, trang {hit['page']}, {hit['chunk_id']}] "
                           f"{hit['text']}")
                candidate = "\n\n".join(selected + [passage])
                if _tokens(tok, base | {"rag_context": candidate}) > max_tokens:
                    continue
                selected.append(passage)
                used.append(hit)
        condition = "eligible" if selected else ("base_prompt_too_long" if base_tokens > max_tokens
                                                  else "no_retrieval_fit")
        coverage.append({"essay_id": eid, "exam_id": base["exam_id"],
                         "base_tokens": base_tokens, "retrieved": len(hits),
                         "selected": len(selected), "status": condition})
        if not selected:
            continue
        for rank, hit in enumerate(used, 1):
            retrievals.append({"essay_id": eid, "rank": rank, "chunk_id": hit["chunk_id"],
                               "document_id": hit["document_id"], "page": hit["page"],
                               "source_sha256": hit["source_sha256"],
                               "retrieval_score": hit["score"]})
        # Both variants receive exactly the same question, answer and essay.
        plain.append(base)
        grounded.append(base | {"rag_context": "\n\n".join(selected)})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(plain, columns=sorted(required)).to_csv(output / "input_plain.csv", index=False)
    pd.DataFrame(grounded, columns=sorted(required) + ["rag_context"]).to_csv(
        output / "input_rag.csv", index=False)
    pd.DataFrame(retrievals, columns=["essay_id", "rank", "chunk_id", "document_id", "page",
                                      "source_sha256", "retrieval_score"]).to_csv(
        output / "retrieval_audit.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    manifest = {"input_sha256": sha(essays_path), "index_sha256": sha(index_path),
                "plain_sha256": sha(output / "input_plain.csv"),
                "rag_sha256": sha(output / "input_rag.csv"),
                "model": model, "course_id": course_id, "max_input_tokens": max_tokens,
                "top_k": top_k, "n_test": len(frame), "n_paired_eligible": len(plain),
                "retrieval_method": "TF-IDF word (1,2), cosine",
                "status": "inputs_ready; no inference or model metrics measured"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not plain:
        raise ValueError("Không có bài nào đủ điều kiện cho so sánh ghép cặp")
    return manifest


def run(output: Path, *, model: str, adapter: Path, max_tokens: int = 4608,
        max_new_tokens: int = 96) -> None:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Thiếu {manifest_path}. Bước prepare chưa hoàn thành; "
                                "kiểm tra đường dẫn essays_split.csv rồi chạy prepare trước.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["model"] != model or manifest["max_input_tokens"] != max_tokens:
        raise ValueError("Model/token limit khác manifest; hãy prepare lại")
    if not adapter.is_dir():
        raise FileNotFoundError(f"Thiếu adapter: {adapter}")
    plain = pd.read_csv(output / "input_plain.csv").fillna("")
    rag = pd.read_csv(output / "input_rag.csv").fillna("")
    if plain.empty or plain["essay_id"].astype(str).tolist() != rag["essay_id"].astype(str).tolist():
        raise ValueError("Hai đầu vào không còn ghép cặp theo essay_id")
    for name, field in (("input_plain.csv", "plain_sha256"), ("input_rag.csv", "rag_sha256")):
        if sha(output / name) != manifest[field]:
            raise ValueError(f"{name} đã thay đổi; hãy prepare lại")
    combined = pd.concat([plain.assign(variant="plain"), rag.assign(variant="rag")],
                         ignore_index=True).fillna("")
    # One model load for both conditions. Unique IDs retain the original pairing.
    combined["essay_id"] = combined["variant"] + "::" + combined["essay_id"].astype(str)
    combined.to_csv(output / "inference_input.csv", index=False)
    predictions = score_total_qlora(output / "inference_input.csv", output / "inference_raw.csv",
                                    model_name=model, adapter_path=adapter, split=None, runs=1,
                                    temperature=0, max_input_tokens=max_tokens,
                                    max_new_tokens=max_new_tokens)
    predictions["variant"] = predictions["essay_id"].str.split("::", n=1).str[0]
    predictions["essay_id"] = predictions["essay_id"].str.split("::", n=1).str[1]
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    manifest.update({"adapter_path": str(adapter.resolve()), "max_new_tokens": max_new_tokens,
                     "adapter_config_sha256": sha(adapter / "adapter_config.json")
                     if (adapter / "adapter_config.json").is_file() else None,
                     "adapter_weight_sha256": {path.name: sha(path) for path in
                                               sorted(adapter.glob("adapter_model.*")) if path.is_file()},
                     "status": "inference_complete; evaluation pending"})
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_zero_shot(output: Path, *, model: str, max_tokens: int = 4608,
                  max_new_tokens: int = 96) -> None:
    """Score the same eligible essays with the untuned base model, without RAG."""
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Thiếu {manifest_path}; hãy prepare trước.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["model"] != model or manifest["max_input_tokens"] != max_tokens:
        raise ValueError("Model/token limit khác manifest; hãy prepare lại")
    if sha(output / "input_plain.csv") != manifest["plain_sha256"]:
        raise ValueError("input_plain.csv đã thay đổi")
    pred = score_total_qlora(output / "input_plain.csv", output / "pred_zero_shot.csv",
                             model_name=model, adapter_path=None, split=None, runs=1,
                             temperature=0, max_input_tokens=max_tokens,
                             max_new_tokens=max_new_tokens)
    if len(pred) != manifest["n_paired_eligible"]:
        raise ValueError("Số dự đoán zero-shot không khớp đầu vào")


def _metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "MAE": None, "RMSE": None, "QWK_41": None,
                "bias": None, "exact": None, "within_1": None}
    gold = frame["gold_total"].to_numpy(float)
    pred = frame["total"].to_numpy(float)
    if not (np.isfinite(gold).all() and np.isfinite(pred).all()):
        raise ValueError("Điểm không hữu hạn")
    qwk = cohen_kappa_score(np.rint(gold * 4).astype(int), np.rint(pred * 4).astype(int),
                             labels=list(range(41)), weights="quadratic")
    return {"n": len(frame), "MAE": float(np.mean(np.abs(pred - gold))),
            "RMSE": float(np.sqrt(np.mean((pred - gold) ** 2))),
            "QWK_41": float(qwk) if math.isfinite(qwk) else None,
            "bias": float(np.mean(pred - gold)),
            "exact": float(np.mean(np.isclose(pred, gold, atol=1e-9))),
            "within_1": float(np.mean(np.abs(pred - gold) <= 1))}


def report(output: Path, gold_path: Path, *, baseline: Path | None = None,
           hybrid: Path | None = None, bootstrap: int = 1000, seed: int = 42) -> dict:
    pred = pd.read_csv(output / "predictions.csv")
    gold = pd.read_csv(gold_path)
    if not {"essay_id", "gold_total"}.issubset(gold):
        raise ValueError("Gold thiếu essay_id hoặc gold_total")
    if "split" in gold:
        gold = gold[gold["split"].astype(str).eq("test")]
    if gold["essay_id"].astype(str).duplicated().any():
        raise ValueError("Gold trùng essay_id")
    coverage = pd.read_csv(output / "coverage.csv")
    expected = set(coverage.loc[coverage.status.eq("eligible"), "essay_id"].astype(str))
    pred["essay_id"] = pred["essay_id"].astype(str)
    gold["essay_id"] = gold["essay_id"].astype(str)
    if pred.duplicated(["variant", "essay_id"]).any() or set(pred.variant) != {"plain", "rag"}:
        raise ValueError("Thiếu nhánh hoặc có dự đoán trùng")
    if any(set(part.essay_id) != expected for _, part in pred.groupby("variant")):
        raise ValueError("Dự đoán không khớp tập bài đã chuẩn bị")
    if not expected.issubset(set(gold.essay_id)):
        raise ValueError("Gold thiếu bài trong tập ghép cặp")
    pred["valid"] = (pred.parse_ok.astype(str).str.lower().eq("true")
                     & pd.to_numeric(pred.total, errors="coerce").between(0, 10))
    valid = {v: set(part.loc[part.valid, "essay_id"]) for v, part in pred.groupby("variant")}
    common = valid["plain"] & valid["rag"]
    if not common:
        raise ValueError("Không có dự đoán hợp lệ ở cả hai nhánh")
    gold_slice = gold[["essay_id", "gold_total"]]
    joined = {v: pred[(pred.variant.eq(v)) & pred.essay_id.isin(common)].merge(
        gold_slice, on="essay_id", validate="one_to_one").sort_values("essay_id")
        for v in ("plain", "rag")}
    for v in joined:
        joined[v].to_csv(output / f"paired_{v}_with_gold.csv", index=False)
    a, b = joined["plain"], joined["rag"]
    if a.essay_id.tolist() != b.essay_id.tolist():
        raise ValueError("Ghép cặp sai thứ tự")
    changes = a[["essay_id", "gold_total"]].copy()
    changes["plain"] = a.total.to_numpy(float)
    changes["rag"] = b.total.to_numpy(float)
    changes["abs_error_plain"] = abs(changes.plain - changes.gold_total)
    changes["abs_error_rag"] = abs(changes.rag - changes.gold_total)
    changes["delta_abs_error"] = changes.abs_error_rag - changes.abs_error_plain
    changes.to_csv(output / "paired_errors.csv", index=False)
    rng = np.random.default_rng(seed)
    deltas = changes.delta_abs_error.to_numpy(float)
    ci = None
    if bootstrap > 0:
        samples = rng.integers(0, len(deltas), size=(bootstrap, len(deltas)))
        ci = [float(x) for x in np.quantile(deltas[samples].mean(axis=1), [.025, .975])]
    result = {"disclaimer": "Post-hoc IT04 test; duplicate train/test content may affect generalization.",
              "n_test": len(coverage), "n_paired_eligible": len(expected),
              "n_common_valid": len(common), "coverage_reasons": coverage.status.value_counts().to_dict(),
              "parse_rate_on_eligible": {v: len(valid[v]) / len(expected) for v in valid},
              "plain": _metrics(a), "rag": _metrics(b),
              "delta_MAE_rag_minus_plain": float(deltas.mean()),
              "delta_MAE_bootstrap_essay_95": ci,
              "per_exam": {v: {str(exam): _metrics(group) for exam, group in part.groupby("exam_id")}
                           for v, part in joined.items()},
              "retrieval": {"n_selected_passages": len(pd.read_csv(output / "retrieval_audit.csv")),
                            "index_sha256": json.loads((output / "manifest.json").read_text())["index_sha256"]}}
    if baseline is not None:
        base = pd.read_csv(baseline)
        if base.essay_id.astype(str).duplicated().any():
            raise ValueError("Baseline trùng essay_id")
        base["essay_id"] = base.essay_id.astype(str)
        base = base[base.essay_id.isin(common)].merge(gold_slice, on="essay_id", validate="one_to_one")
        if set(base.essay_id) != common:
            raise ValueError("Baseline thiếu bài trong tập ghép cặp")
        result["tfidf_ridge"] = _metrics(base)
    if hybrid is not None:
        raw = pd.read_csv(hybrid)
        if not {"essay_id", "final_total", "grading_mode"}.issubset(raw):
            raise ValueError("Hybrid cần cột essay_id, final_total, grading_mode")
        if raw.essay_id.astype(str).duplicated().any():
            raise ValueError("Hybrid trùng essay_id")
        raw["essay_id"] = raw.essay_id.astype(str)
        raw = raw[raw.essay_id.isin(common)].merge(gold_slice, on="essay_id", validate="one_to_one")
        if set(raw.essay_id) != common:
            raise ValueError("Hybrid thiếu bài trong tập ghép cặp")
        result["hybrid"] = _metrics(raw.rename(columns={"final_total": "total"}))
        result["hybrid_route_counts"] = raw.grading_mode.value_counts().to_dict()
    zero_path = output / "pred_zero_shot.csv"
    if zero_path.is_file():
        zero = pd.read_csv(zero_path)
        zero["essay_id"] = zero.essay_id.astype(str)
        if zero.essay_id.duplicated().any() or set(zero.essay_id) != expected:
            raise ValueError("Zero-shot không khớp tập đã chuẩn bị")
        z_valid = (zero.parse_ok.astype(str).str.lower().eq("true")
                   & pd.to_numeric(zero.total, errors="coerce").between(0, 10))
        common_three = common & set(zero.loc[z_valid, "essay_id"])
        result["zero_shot_parse_rate_on_eligible"] = float(z_valid.mean())
        result["n_common_three_qlora_plain_rag_zero_shot"] = len(common_three)
        if common_three:
            result["three_way_common"] = {
                "zero_shot": _metrics(zero[zero.essay_id.isin(common_three)].merge(
                    gold_slice, on="essay_id", validate="one_to_one")),
                "plain": _metrics(a[a.essay_id.isin(common_three)]),
                "rag": _metrics(b[b.essay_id.isin(common_three)]),
            }
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Đối chứng QLoRA IT04 có/không RAG")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--essays", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-tokens", type=int, default=4608)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--expected-test", type=int, default=73,
                   help="Số bài test mong đợi; mặc định 73, đặt 24 chỉ khi chạy bộ mẫu")
    p = sub.add_parser("run")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=4608)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p = sub.add_parser("run-zero-shot")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-tokens", type=int, default=4608)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p = sub.add_parser("report")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--hybrid", type=Path)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.essays, args.index, args.output, args.model,
                         max_tokens=args.max_tokens, top_k=args.top_k,
                         expected_test=args.expected_test)
    elif args.command == "run":
        run(args.output, model=args.model, adapter=args.adapter,
            max_tokens=args.max_tokens, max_new_tokens=args.max_new_tokens)
        result = {"status": "inference_complete", "output": str(args.output)}
    elif args.command == "run-zero-shot":
        run_zero_shot(args.output, model=args.model, max_tokens=args.max_tokens,
                      max_new_tokens=args.max_new_tokens)
        result = {"status": "zero_shot_complete", "output": str(args.output)}
    else:
        result = report(args.output, args.gold, baseline=args.baseline,
                        hybrid=args.hybrid, bootstrap=args.bootstrap)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

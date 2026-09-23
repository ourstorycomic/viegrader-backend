"""Giao diện dòng lệnh của viegrader.

    viegrader check-rubric  rubrics/bai_kiem_tra_mon_hoc.yaml
    viegrader handbook      -r rubric.yaml -o docs/so_tay_gan_nhan.md
    viegrader clean         -i data/raw/bai_lam.csv -o data/processed/
    viegrader sample        -i data/processed/clean.csv -n 200 -o data/sample.csv
    viegrader form          -r rubric.yaml -i data/sample.csv --raters GK1,GK2 -o forms/
    viegrader agreement     -a data/annotations.csv -r rubric.yaml
    viegrader adjudicate    -a data/annotations.csv -r rubric.yaml -o data/gold.csv
    viegrader train         -i data/clean.csv -g data/gold.csv -r rubric.yaml -o models/m.pkl
    viegrader score         -i data/new.csv -m models/m.pkl -o out/ket_qua.xlsx
    viegrader evaluate      -p out/ket_qua.csv -g data/gold_test.csv -r rubric.yaml
    viegrader demo          -o demo_out/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _p(msg: str = "") -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
def cmd_check_rubric(a: argparse.Namespace) -> int:
    from .config import load_rubric
    from .scoring.rules import check_rules

    r = load_rubric(a.rubric)
    _p(f"Rubric: {r.name} ({r.rubric_id})")
    _p(f"Thang điểm: {r.scale_max}, bước làm tròn {r.scale_step}")
    _p(f"Số tiêu chí: {len(r.criteria)}")
    warns = r.validate()
    _p("")
    if warns:
        _p("CẢNH BÁO CẤU TRÚC:")
        for w in warns:
            _p(f"  - {w}")
    else:
        _p("Cấu trúc rubric hợp lệ.")
    probs = check_rules(r)
    _p("")
    if probs:
        _p("LỖI QUY ĐỊNH CHẤM:")
        for pr in probs:
            _p(f"  - {pr}")
        return 1
    _p(f"Đã kiểm tra {len(r.rules)} quy định chấm: hợp lệ.")
    _p("")
    for c in r.criteria:
        _p(f"  {c.key:<12} {c.name:<28} trọng số {c.weight:>5.0%}  tối đa {c.max_score:>5}"
           f"  mức: {c.level_scores()}")
    return 0


def cmd_handbook(a: argparse.Namespace) -> int:
    from .config import load_rubric
    from .io_utils import write_text
    from .labeling.guidelines import build_handbook

    r = load_rubric(a.rubric)
    md = build_handbook(r)
    write_text(md, a.output)
    _p(f"Đã xuất sổ tay gán nhãn: {a.output}")
    return 0


def cmd_clean(a: argparse.Namespace) -> int:
    from .cleaning.pipeline import CleanConfig, clean_dataframe
    from .io_utils import load_input, write_json, write_table, write_text

    df = load_input(a.input)
    _p(f"Đã nạp {len(df)} bản ghi từ {a.input}")
    cfg = CleanConfig(secret=a.secret, anonymize=not a.no_anonymize,
                      dup_threshold=a.dup_threshold)
    clean, rejected, rep = clean_dataframe(df, cfg)

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    write_table(clean, out / "clean.csv")
    write_table(rejected, out / "rejected.csv")
    write_json(rep.to_dict(), out / "clean_report.json")
    write_text(rep.summary(), out / "clean_report.txt")
    _p("")
    _p(rep.summary())
    _p("")
    _p(f"Đã ghi: {out/'clean.csv'}, {out/'rejected.csv'}, {out/'clean_report.json'}")
    return 0


def cmd_sample(a: argparse.Namespace) -> int:
    from .io_utils import load_input, write_table
    from .labeling.sampling import stratified_pilot

    df = load_input(a.input)
    s = stratified_pilot(df, n=a.n, n_bins=a.bins, seed=a.seed)
    write_table(s, a.output)
    _p(f"Đã chọn {len(s)}/{len(df)} bài theo phân tầng độ dài -> {a.output}")
    return 0


def cmd_form(a: argparse.Namespace) -> int:
    from .config import load_rubric
    from .io_utils import load_input, write_table
    from .labeling.guidelines import build_scoring_form
    from .labeling.sampling import double_scoring_plan

    r = load_rubric(a.rubric)
    df = load_input(a.input)
    raters = [x.strip() for x in a.raters.split(",") if x.strip()]
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    plan = double_scoring_plan(df, raters, double_rate=a.double_rate, seed=a.seed)
    write_table(plan, out / "phan_cong_cham.csv")
    for rid in raters:
        ids = plan[plan["rater_id"] == rid]["essay_id"].tolist()
        form = build_scoring_form(r, ids, rid)
        write_table(form, out / f"phieu_cham_{rid}.xlsx")
        _p(f"  {rid}: {len(ids)} bài -> {out / f'phieu_cham_{rid}.xlsx'}")
    _p(f"Kế hoạch chấm: {out/'phan_cong_cham.csv'} "
       f"({len(plan)} lượt chấm, {plan['essay_id'].nunique()} bài)")
    return 0


def cmd_agreement(a: argparse.Namespace) -> int:
    from .config import load_rubric
    from .io_utils import load_input, write_table
    from .labeling.adjudicate import rater_severity
    from .labeling.agreement import agreement_report, disagreements

    r = load_rubric(a.rubric)
    ann = load_input(a.annotations)
    crits = ["total"] + [c.key for c in r.criteria]
    rep = agreement_report(ann, crits, step=r.scale_step)
    _p("===== ĐỘ ĐỒNG THUẬN GIỮA GIÁM KHẢO =====")
    _p(rep.round(4).to_string(index=False))
    _p("")
    sev = rater_severity(ann)
    _p("===== ĐỘ KHẮT KHE CỦA TỪNG GIÁM KHẢO =====")
    _p(sev.round(3).to_string(index=False))
    _p("")
    dis = disagreements(ann, "total", tol=r.scale_max * 0.1)
    _p(f"Số bài cần phân xử (lệch > {r.scale_max*0.1:.1f} điểm): {len(dis)}")
    if a.output:
        out = Path(a.output)
        out.mkdir(parents=True, exist_ok=True)
        write_table(rep, out / "do_dong_thuan.csv")
        write_table(sev, out / "do_khat_khe.csv")
        write_table(dis, out / "can_phan_xu.csv")
        _p(f"Đã ghi kết quả vào {out}")
    qwk = rep.loc[rep["criterion"] == "total", "QWK_mean"]
    if len(qwk) and qwk.iloc[0] < 0.70:
        _p("")
        _p("⚠ QWK điểm tổng < 0.70: rubric còn mơ hồ. "
           "Cần rà mô tả mức và tập huấn lại TRƯỚC khi gán nhãn diện rộng.")
    return 0


def cmd_adjudicate(a: argparse.Namespace) -> int:
    from .config import load_rubric
    from .io_utils import load_input, write_table
    from .labeling.adjudicate import resolve, snap_to_rubric

    r = load_rubric(a.rubric)
    ann = load_input(a.annotations)
    gold = snap_to_rubric(resolve(ann, r, tol_ratio=a.tol), r)
    write_table(gold, a.output)
    n_adj = int(gold["needs_adjudication"].sum())
    _p(f"Đã sinh nhãn vàng cho {len(gold)} bài -> {a.output}")
    _p(f"Trong đó {n_adj} bài cần giám khảo thứ ba phân xử "
       f"({n_adj/max(1,len(gold)):.1%})")
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    from .config import load_rubric
    from .grader import Grader, GraderConfig
    from .io_utils import load_input, write_text
    from .evaluation.metrics import evaluation_report

    r = load_rubric(a.rubric)
    df = load_input(a.input)
    gold = load_input(a.gold)
    df = df.merge(gold[["essay_id"]], on="essay_id", how="inner")
    gold = gold.set_index("essay_id").loc[df["essay_id"]].reset_index()
    _p(f"Dữ liệu huấn luyện: {len(df)} bài")

    n_test = max(10, int(len(df) * a.test_size))
    tr, te = df.iloc[:-n_test].reset_index(drop=True), df.iloc[-n_test:].reset_index(drop=True)
    gtr = gold.iloc[:-n_test].reset_index(drop=True)
    gte = gold.iloc[-n_test:].reset_index(drop=True)

    cfg = GraderConfig(encoder=a.encoder, use_llm=a.use_llm)
    g = Grader(r, cfg)
    if "keywords" in df.columns:
        kw = {}
        for pid, sub in df.groupby("prompt_id"):
            v = str(sub["keywords"].iloc[0] or "")
            if v:
                kw[str(pid)] = [x.strip() for x in v.split(";") if x.strip()]
        g.set_keywords(kw)
    g.fit(tr, gtr, val_df=te, val_gold=gte)
    g.save(a.output)
    _p("")
    _p(g.scorer.report())
    _p("")
    if g.ensemble.fitted:
        _p("Trọng số tổng hợp nguồn điểm:")
        _p(g.ensemble.explain().to_string(index=False))
        _p("")

    pred = g.score_to_frame(te, with_feedback=False)
    rep = evaluation_report(gte, pred, [c.key for c in r.criteria], r.scale_step)
    _p(rep)
    if a.report:
        write_text(rep, a.report)
    _p("")
    _p(f"Đã lưu mô hình: {a.output}")
    return 0


def cmd_score(a: argparse.Namespace) -> int:
    from .grader import Grader
    from .io_utils import load_input, write_table, write_text

    g = Grader.load(a.model)
    df = load_input(a.input)
    _p(f"Đang chấm {len(df)} bài bằng rubric '{g.rubric.name}'...")
    out_df = g.score_to_frame(df, with_feedback=not a.no_feedback)

    out = Path(a.output)
    write_table(out_df, out)
    _p("")
    _p(f"Điểm trung bình : {out_df['total'].mean():.2f}")
    _p(f"Độ lệch chuẩn   : {out_df['total'].std():.2f}")
    _p(f"Cần phúc tra    : {int(out_df['needs_human_review'].sum())} bài "
       f"({out_df['needs_human_review'].mean():.1%})")
    if out_df["flags"].any():
        from collections import Counter
        c = Counter(f for s in out_df["flags"] for f in str(s).split("|") if f)
        _p("Cờ cảnh báo     : " + ", ".join(f"{k}={v}" for k, v in c.most_common()))
    if a.feedback_dir and not a.no_feedback:
        fd = Path(a.feedback_dir)
        fd.mkdir(parents=True, exist_ok=True)
        for _, row in out_df.iterrows():
            write_text(str(row.get("feedback", "")), fd / f"{row['essay_id']}.md")
        _p(f"Đã xuất phản hồi từng bài vào {fd}")
    _p(f"Đã ghi kết quả: {out}")
    return 0


def cmd_evaluate(a: argparse.Namespace) -> int:
    from .config import load_rubric
    from .evaluation.metrics import evaluation_report, length_bias, review_efficiency
    from .io_utils import load_input, write_text

    r = load_rubric(a.rubric)
    pred = load_input(a.pred)
    gold = load_input(a.gold)
    merged = pred.merge(gold, on="essay_id", suffixes=("", "_gold"))
    rep = evaluation_report(merged, merged, [c.key for c in r.criteria], r.scale_step,
                            qwk_human_human=a.qwk_hh)
    _p(rep)
    if "needs_human_review" in merged.columns:
        eff = review_efficiency(merged["gold_total"], merged["total"],
                                merged["needs_human_review"].astype(bool))
        _p("")
        _p("## Hiệu quả cơ chế phúc tra")
        for k, v in eff.items():
            _p(f"  {k:<26}: {v:.4f}")
    if "n_words" in merged.columns or "text" in merged.columns:
        lb = length_bias(merged)
        if len(lb):
            _p("")
            _p("## Kiểm tra thiên lệch độ dài")
            _p(lb.round(4).to_string(index=False))
    if a.output:
        write_text(rep, a.output)
    return 0


def cmd_demo(a: argparse.Namespace) -> int:
    """Chạy toàn bộ pipeline trên dữ liệu mô phỏng."""
    from .cleaning.pipeline import clean_dataframe
    from .config import default_rubric_path, load_rubric
    from .evaluation.metrics import evaluation_report, review_efficiency
    from .grader import Grader, GraderConfig
    from .io_utils import write_table, write_text
    from .labeling.adjudicate import resolve, snap_to_rubric
    from .labeling.agreement import agreement_report
    from .labeling.guidelines import build_handbook
    from .simulate import make_dataset

    rpath = a.rubric or default_rubric_path()
    r = load_rubric(rpath)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    _p("[1/6] Sinh dữ liệu mô phỏng...")
    essays, gold_true, ann = make_dataset(r, n=a.n, seed=42)
    write_table(essays, out / "01_raw.csv")

    _p("[2/6] Làm sạch dữ liệu...")
    clean, rejected, crep = clean_dataframe(essays)
    write_table(clean, out / "02_clean.csv")
    write_table(rejected, out / "02_rejected.csv")
    write_text(crep.summary(), out / "02_clean_report.txt")
    _p(crep.summary())

    _p("")
    _p("[3/6] Đo độ đồng thuận giám khảo và sinh nhãn vàng...")
    arep = agreement_report(ann, ["total"] + [c.key for c in r.criteria], r.scale_step)
    write_table(arep, out / "03_agreement.csv")
    _p(arep.round(3).to_string(index=False))
    gold = snap_to_rubric(resolve(ann, r), r)
    write_table(gold, out / "03_gold.csv")

    _p("")
    _p("[4/6] Xuất sổ tay gán nhãn...")
    write_text(build_handbook(r), out / "04_so_tay_gan_nhan.md")

    _p("[5/6] Huấn luyện mô hình...")
    df = clean.merge(gold[["essay_id"]], on="essay_id", how="inner")
    g_al = gold.set_index("essay_id").loc[df["essay_id"]].reset_index()
    n_test = max(20, int(len(df) * 0.25))
    tr, te = df.iloc[:-n_test].reset_index(drop=True), df.iloc[-n_test:].reset_index(drop=True)
    gtr, gte = g_al.iloc[:-n_test].reset_index(drop=True), g_al.iloc[-n_test:].reset_index(drop=True)

    g = Grader(r, GraderConfig(encoder="tfidf", use_llm=False))
    kw = {}
    for pid, sub in df.groupby("prompt_id"):
        v = str(sub["keywords"].iloc[0] or "")
        if v:
            kw[str(pid)] = [x.strip() for x in v.split(";") if x.strip()]
    g.set_keywords(kw)
    g.fit(tr, gtr, val_df=te, val_gold=gte)
    g.save(out / "05_model.pkl")
    _p(g.scorer.report())

    _p("")
    _p("[6/6] Chấm tập kiểm tra và đánh giá...")
    pred = g.score_to_frame(te)
    write_table(pred, out / "06_predictions.csv")
    rep = evaluation_report(gte, pred, [c.key for c in r.criteria], r.scale_step,
                            qwk_human_human=float(arep.loc[arep["criterion"] == "total",
                                                           "QWK_mean"].iloc[0]))
    eff = review_efficiency(gte["gold_total"], pred["total"],
                            pred["needs_human_review"].astype(bool))
    rep += "\n\n## Hiệu quả cơ chế phúc tra\n" + "\n".join(
        f"  {k:<26}: {v:.4f}" for k, v in eff.items()
    )
    from .evaluation.metrics import review_threshold_curve, suggest_review_threshold
    curve = review_threshold_curve(gte["gold_total"], pred["total"], pred["confidence"])
    sug = suggest_review_threshold(gte["gold_total"], pred["total"], pred["confidence"])
    write_table(curve, out / "06_review_threshold_curve.csv")
    rep += ("\n\n## Chọn ngưỡng chuyển phúc tra (quy định R07)\n"
            + curve.round(3).to_string(index=False)
            + f"\n\n  => Ngưỡng đề xuất: confidence < {sug['threshold']:.2f} "
              f"(phúc tra {sug['review_rate']:.1%}, bắt được {sug['recall_of_errors']:.1%} "
              f"số bài bị chấm lệch > 1 điểm)")
    write_text(rep, out / "06_evaluation.txt")
    _p(rep)
    _p("")
    _p("--- Ví dụ phản hồi cho 1 bài ---")
    _p(pred["feedback"].iloc[0][:1200])
    _p("")
    _p(f"Toàn bộ kết quả demo nằm trong: {out.resolve()}")
    return 0


def cmd_rag_index(a: argparse.Namespace) -> int:
    from .io_utils import load_input
    from .rag import DocumentChunk, TfidfRAGIndex

    df = load_input(a.input)
    if "text" not in df:
        raise ValueError("Tài liệu RAG cần cột 'text'.")
    chunks = []
    for i, row in df.iterrows():
        chunks.append(DocumentChunk(
            chunk_id=str(row.get("chunk_id", f"chunk_{i:06d}")), text=str(row["text"]),
            document_id=str(row.get("document_id", "")), course_id=str(row.get("course_id", "")),
            prompt_id=str(row.get("prompt_id", "")), section=str(row.get("section", "")),
            version=str(row.get("version", "")), approved=bool(row.get("approved", True)),
        ))
    index = TfidfRAGIndex().fit(chunks)
    index.save_documents(a.output)
    _p(f"Đã lập chỉ mục {len(index.chunks)} đoạn tài liệu -> {a.output}")
    return 0


def cmd_train_qlora(a: argparse.Namespace) -> int:
    from .config import load_rubric
    from .io_utils import load_input
    from .qlora import QLoRAConfig, build_instruction_records, train_qlora

    essays, gold, rubric = load_input(a.input), load_input(a.gold), load_rubric(a.rubric)
    if a.split_column in essays.columns:
        essays = essays[essays[a.split_column].astype(str).eq(a.train_split)].copy()
    elif not a.allow_unsplit:
        raise ValueError(
            f"Dữ liệu QLoRA thiếu cột {a.split_column!r}. Hãy chạy split-dataset trước "
            "hoặc thêm --allow-unsplit nếu đây chắc chắn là file train riêng."
        )
    if a.split_column in gold.columns:
        gold = gold[gold[a.split_column].astype(str).eq(a.train_split)].copy()
    if "essay_id" in essays and "essay_id" in gold:
        essays = essays.merge(gold[["essay_id"]], on="essay_id", how="inner")
        gold = gold.set_index("essay_id").loc[essays["essay_id"]].reset_index()
    records = build_instruction_records(essays, gold, rubric)
    cfg = QLoRAConfig.rtx_5060_ti_16gb(
        model_name=a.model, output_dir=a.output, epochs=a.epochs,
        max_seq_length=a.max_length, batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, seed=a.seed,
    )
    _p(f"Bắt đầu QLoRA với {len(records)} mẫu; adapter -> {a.output}")
    train_qlora(records, cfg)
    return 0


def cmd_score_qlora(a: argparse.Namespace) -> int:
    from .backends.vistral import VistralBackend, VistralConfig
    from .config import load_rubric
    from .io_utils import load_input, write_table

    rubric = load_rubric(a.rubric)
    essays = load_input(a.input)
    backend = VistralBackend(rubric, VistralConfig(
        model_name=a.model, adapter_path=a.adapter,
        temperature=a.temperature, max_new_tokens=a.max_new_tokens,
        max_input_tokens=a.max_input_tokens,
    ))
    output = Path(a.output)
    for run_id in range(1, a.runs + 1):
        prediction = backend.predict(essays)
        frame = prediction.scores.copy()
        for criterion in rubric.criteria:
            if criterion.key not in frame:
                frame[criterion.key] = 0.0
            allowed = criterion.level_scores()
            frame[criterion.key] = pd.to_numeric(frame[criterion.key], errors="coerce").fillna(0.0)
            frame[criterion.key] = frame[criterion.key].map(
                lambda value: min(allowed, key=lambda score: abs(score - value))
            )
        frame["total"] = 0.0
        for criterion in rubric.criteria:
            frame["total"] += (
                frame[criterion.key] / criterion.max_score
                * criterion.weight * rubric.scale_max
            )
        frame["total"] = (frame["total"] / rubric.scale_step).round() * rubric.scale_step
        frame.insert(0, "essay_id", essays["essay_id"].astype(str).values)
        frame["confidence"] = prediction.confidence
        frame["feedback"] = prediction.comments
        frame["flags"] = ["|".join(x) for x in prediction.flags]
        frame["run_id"] = run_id
        target = output if a.runs == 1 else output.with_name(f"{output.stem}_run{run_id:02d}{output.suffix}")
        write_table(frame, target)
        _p(f"Đã chấm QLoRA lượt {run_id}/{a.runs}: {target}")
    backend.close()
    return 0


def cmd_prepare_dataset(a: argparse.Namespace) -> int:
    from .datasets import prepare_vietnamese_it_dataset

    report = prepare_vietnamese_it_dataset(a.source_dir, a.output)
    _p(f"Đã chuẩn hóa {report['n_rows']} bài từ {report['n_files']} tệp.")
    _p(f"Trạng thái nhãn: {report['labels_status']}")
    _p(f"Báo cáo kiểm toán: {Path(a.output) / 'dataset_audit.json'}")
    return 0


def cmd_split_dataset(a: argparse.Namespace) -> int:
    import json
    from .datasets import split_research_dataset
    from .io_utils import load_input

    report = split_research_dataset(
        load_input(a.input), load_input(a.gold), a.output,
        group_column=a.group_column, test_size=a.test_size,
        validation_size=a.validation_size, seed=a.seed,
        holdout_prompt=a.holdout_prompt,
    )
    _p(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_hardware_check(a: argparse.Namespace) -> int:
    import json
    from .hardware import detect_gpu
    info = detect_gpu(strict_blackwell=a.strict_blackwell)
    _p(info.report())
    if a.json:
        output = Path(a.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(info.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if info.ready_for_qlora else 2


def cmd_sus(a: argparse.Namespace) -> int:
    from .io_utils import load_input, write_json
    from .usability import summarize_sus

    result = summarize_sus(load_input(a.input))
    for key, value in result.items():
        _p(f"{key:<20}: {value:.4f}")
    if a.output:
        write_json(result, a.output)
    return 0


def cmd_bertscore(a: argparse.Namespace) -> int:
    from .evaluation.feedback_metrics import bertscore_feedback
    from .io_utils import load_input, write_json
    generated, references = load_input(a.generated), load_input(a.references)
    if a.generated_column not in generated.columns:
        raise ValueError(f"Thiếu cột {a.generated_column!r} trong {a.generated}")
    if a.reference_column not in references.columns:
        raise ValueError(f"Thiếu cột {a.reference_column!r} trong {a.references}")
    if a.id_column and a.id_column in generated.columns and a.id_column in references.columns:
        merged = generated[[a.id_column, a.generated_column]].rename(
            columns={a.generated_column: "__generated"}
        ).merge(
            references[[a.id_column, a.reference_column]].rename(
                columns={a.reference_column: "__reference"}
            ), on=a.id_column, how="inner"
        )
        gen, ref = merged["__generated"], merged["__reference"]
    else:
        if len(generated) != len(references):
            raise ValueError("Hai tệp khác số dòng và không ghép được theo id.")
        gen, ref = generated[a.generated_column], references[a.reference_column]
    result = bertscore_feedback(
        gen.fillna("").astype(str).tolist(), ref.fillna("").astype(str).tolist(),
        model_type=a.model, batch_size=a.batch_size,
    )
    write_json(result, a.output)
    _p("\n".join(f"{key}: {value:.6f}" for key, value in result.items()))
    _p(f"Đã lưu: {a.output}")
    return 0


def cmd_ablation_plan(a: argparse.Namespace) -> int:
    from .experiments.ablation import default_ablation_configs
    from .io_utils import write_table
    frame = pd.DataFrame([cfg.to_dict() for cfg in default_ablation_configs()])
    write_table(frame, a.output)
    _p(frame.to_string(index=False))
    _p(f"Đã lưu ma trận A–H: {a.output}")
    return 0


def cmd_ablation_evaluate(a: argparse.Namespace) -> int:
    from .evaluation.metrics import score_metrics
    from .io_utils import load_input, write_table
    gold = load_input(a.gold)
    if a.gold_column not in gold.columns:
        raise ValueError(f"Thiếu cột điểm vàng {a.gold_column!r}.")
    rows = []
    for path in sorted(Path(a.pred_dir).glob(a.pattern)):
        pred = load_input(path)
        if a.pred_column not in pred.columns:
            raise ValueError(f"{path} thiếu cột {a.pred_column!r}.")
        if a.id_column in gold.columns and a.id_column in pred.columns:
            joined = gold[[a.id_column, a.gold_column]].merge(
                pred[[a.id_column, a.pred_column]], on=a.id_column, how="inner"
            )
            y_true, y_pred = joined[a.gold_column], joined[a.pred_column]
        else:
            if len(gold) != len(pred):
                raise ValueError(f"{path} khác số dòng với gold và không ghép được theo id.")
            y_true, y_pred = gold[a.gold_column], pred[a.pred_column]
        experiment_id = path.stem.removeprefix("pred_")
        rows.append({"experiment_id": experiment_id, "n": len(y_true),
                     **score_metrics(y_true, y_pred)})
    if not rows:
        raise FileNotFoundError(f"Không thấy {a.pattern} trong {a.pred_dir}")
    result = pd.DataFrame(rows).sort_values("experiment_id")
    write_table(result, a.output)
    _p(result.to_string(index=False))
    _p(f"Đã lưu: {a.output}")
    return 0


def cmd_hf_push(a: argparse.Namespace) -> int:
    from .huggingface import HuggingFaceConfig, HuggingFaceService
    service = HuggingFaceService(HuggingFaceConfig(
        repo_id=a.repo, repo_type=a.repo_type, private=not a.public,
        revision=a.revision,
    ))
    _p(f"Tài khoản Hugging Face: {service.whoami().get('name', '')}")
    url = (service.push_adapter(a.path) if a.repo_type == "model"
           else service.upload_folder(a.path, "Upload VieGrader artifact"))
    _p(f"Đã tải lên: {url}")
    return 0


def cmd_moodle(a: argparse.Namespace) -> int:
    import json
    if a.moodle_action == "push-grade" and not a.approved:
        raise PermissionError("Phải thêm --approved sau khi giảng viên phê duyệt.")
    from .integrations import MoodleClient, MoodleConfig
    from .io_utils import write_json
    client = MoodleClient(MoodleConfig.from_env())
    if a.moodle_action == "assignments":
        result = client.get_assignments(a.course_id)
    elif a.moodle_action == "submissions":
        result = client.get_submissions(a.assignment_id)
    else:
        result = client.save_grade(
            a.assignment_id, a.user_id, a.grade, a.feedback, approved=True,
        )
    if a.output:
        write_json(result, a.output)
        _p(f"Đã lưu: {a.output}")
    else:
        _p(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("Cài viegrader[api] trước khi chạy serve.") from exc
    from .api import create_app
    from .grader import Grader
    moodle = None
    if a.enable_moodle:
        from .integrations import MoodleClient, MoodleConfig
        moodle = MoodleClient(MoodleConfig.from_env())
    app = create_app(lambda: Grader.load(a.model), moodle_client=moodle)
    uvicorn.run(app, host=a.host, port=a.port)
    return 0


def cmd_gui(a: argparse.Namespace) -> int:
    from .gui.app import launch
    launch(
        server_name=a.host, server_port=a.port,
        share=a.share, show_error=True,
    )
    return 0


def cmd_ubuntu_gui(a: argparse.Namespace) -> int:
    from .config_ubuntu import UbuntuServerConfig
    from .gui.app import launch
    cfg = UbuntuServerConfig.load(a.config)
    cfg.prepare()
    launch(
        server_name=cfg.host, server_port=cfg.port, share=cfg.share,
        show_error=True, allowed_paths=cfg.allowed_paths,
    )
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    from .reporting import build_report_bundle
    bundle, summary, outputs = build_report_bundle(
        a.input, project_name=a.project, author=a.author,
        formats=[x.strip() for x in a.formats.split(",") if x.strip()],
        output_dir=a.output,
    )
    _p(summary.to_string(index=False))
    _p(f"Đã xuất {len(outputs)} báo cáo; gói ZIP: {bundle}")
    return 0


# --------------------------------------------------------------------------- #
def cmd_dm_prepare(a: argparse.Namespace) -> int:
    import json
    from .discrete_math import prepare_discrete_math_dataset

    report = prepare_discrete_math_dataset(
        a.source_dir, a.gold, a.rubric_dir, a.catalog, a.output,
        secret=a.secret, conflict_policy=a.conflict_policy, id_map_path=a.id_map,
    )
    _p(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_dm_split(a: argparse.Namespace) -> int:
    import json
    from .discrete_math import split_discrete_math_dataset

    report = split_discrete_math_dataset(
        a.input, a.gold, a.output, seed=a.seed,
        train_ratio=a.train_ratio, validation_ratio=a.validation_ratio,
    )
    _p(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_dm_train_baseline(a: argparse.Namespace) -> int:
    import json
    from .discrete_math import train_total_baseline

    report = train_total_baseline(a.input, a.gold, a.output, alpha=a.alpha)
    _p(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_dm_score_baseline(a: argparse.Namespace) -> int:
    from .discrete_math import score_total_baseline

    result = score_total_baseline(a.model, a.input, a.output)
    _p(f"Đã chấm {len(result)} bài -> {a.output}")
    return 0


def cmd_dm_build_qlora(a: argparse.Namespace) -> int:
    from .discrete_math import build_total_instruction_records

    records = build_total_instruction_records(a.input, a.gold, a.output, split=a.split)
    _p(f"Đã tạo {len(records)} instruction records nhãn tổng -> {a.output}")
    return 0


def cmd_dm_train_qlora(a: argparse.Namespace) -> int:
    from .discrete_math import train_total_qlora

    _p(f"Bắt đầu QLoRA nhãn tổng; adapter -> {a.output}")
    train_total_qlora(
        a.records, a.output, model_name=a.model, epochs=a.epochs,
        max_length=a.max_length, batch_size=a.batch_size,
        grad_accum=a.grad_accum, seed=a.seed,
    )
    return 0


def cmd_dm_score_qlora(a: argparse.Namespace) -> int:
    from .discrete_math import score_total_qlora

    result = score_total_qlora(
        a.input, a.output, model_name=a.model, adapter_path=a.adapter,
        split=a.split, runs=a.runs, temperature=a.temperature,
        max_input_tokens=a.max_input_tokens, max_new_tokens=a.max_new_tokens,
    )
    _p(f"Đã sinh {len(result)} kết quả ({a.runs} lượt) -> {a.output}")
    return 0


def cmd_dm_evaluate(a: argparse.Namespace) -> int:
    import json
    from .discrete_math import evaluate_total_predictions

    report = evaluate_total_predictions(a.pred, a.gold, a.output, split=a.split)
    _p(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Pipeline v0.7: đọc trực tiếp dataset chuẩn JSONL/ZIP và chấm lai theo rubric.
def cmd_it04_prepare(a: argparse.Namespace) -> int:
    import json
    from .standardized_pipeline import prepare_standardized_dataset

    report = prepare_standardized_dataset(a.source, a.rubric_dir, a.output, min_chars=a.min_chars)
    _p(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_it04_validate_rubric(a: argparse.Namespace) -> int:
    import json
    from .standardized_pipeline import validate_rubrics

    report = validate_rubrics(a.rubric_dir, a.output)
    _p(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_it04_train(a: argparse.Namespace) -> int:
    import json
    from .standardized_pipeline import train_standardized

    report = train_standardized(a.prepared, a.output, alpha=a.alpha)
    _p(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_it04_grade(a: argparse.Namespace) -> int:
    import json
    from .standardized_pipeline import grade_with_rubric

    report = grade_with_rubric(
        a.model, a.input, a.rubric_dir, a.output,
        rubric_weight=float(a.rubric_weight), min_section_coverage=a.min_section_coverage,
    )
    _p(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_it04_evaluate(a: argparse.Namespace) -> int:
    import json
    from .standardized_pipeline import evaluate_grading

    report = evaluate_grading(a.scores, a.gold, a.output)
    _p(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_it04_report(a: argparse.Namespace) -> int:
    import json
    from .standardized_pipeline import build_workflow_report

    report = build_workflow_report(a.root, a.output)
    _p(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_it04_run_all(a: argparse.Namespace) -> int:
    import json
    from .standardized_pipeline import run_all

    report = run_all(
        a.source, a.rubric_dir, a.output, alpha=a.alpha,
        rubric_weight=a.rubric_weight,
    )
    _p(json.dumps({"status": "completed", "output": a.output,
                   "n_stage_reports": report["workflow"]["n_stage_reports"]},
                  ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="viegrader",
        description="Chấm điểm tự động bài tự luận tiếng Việt theo rubric",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check-rubric", help="Kiểm tra tính hợp lệ của rubric và quy định chấm")
    p.add_argument("rubric")
    p.set_defaults(func=cmd_check_rubric)

    p = sub.add_parser("handbook", help="Xuất sổ tay gán nhãn từ rubric")
    p.add_argument("-r", "--rubric", required=True)
    p.add_argument("-o", "--output", default="so_tay_gan_nhan.md")
    p.set_defaults(func=cmd_handbook)

    p = sub.add_parser("clean", help="Làm sạch và ẩn danh bộ dữ liệu bài làm")
    p.add_argument("-i", "--input", required=True, help="File bảng hoặc thư mục bài làm")
    p.add_argument("-o", "--output", default="data/processed")
    p.add_argument("--secret", default="viegrader-default-key", help="Khoá băm mã sinh viên")
    p.add_argument("--no-anonymize", action="store_true")
    p.add_argument("--dup-threshold", type=float, default=0.5)
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("sample", help="Lấy mẫu phân tầng để gán nhãn")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", default="data/sample.csv")
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("form", help="Sinh kế hoạch chấm và phiếu chấm cho giám khảo")
    p.add_argument("-r", "--rubric", required=True)
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", default="forms")
    p.add_argument("--raters", required=True, help="Danh sách giám khảo, vd: GK1,GK2,GK3")
    p.add_argument("--double-rate", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_form)

    p = sub.add_parser("agreement", help="Đo độ đồng thuận giữa giám khảo")
    p.add_argument("-a", "--annotations", required=True)
    p.add_argument("-r", "--rubric", required=True)
    p.add_argument("-o", "--output", default=None)
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser("adjudicate", help="Hợp nhất nhãn nhiều giám khảo thành nhãn vàng")
    p.add_argument("-a", "--annotations", required=True)
    p.add_argument("-r", "--rubric", required=True)
    p.add_argument("-o", "--output", default="data/gold.csv")
    p.add_argument("--tol", type=float, default=0.10, help="Ngưỡng lệch chấp nhận (tỉ lệ thang điểm)")
    p.set_defaults(func=cmd_adjudicate)

    p = sub.add_parser("train", help="Huấn luyện mô hình chấm")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-g", "--gold", required=True)
    p.add_argument("-r", "--rubric", required=True)
    p.add_argument("-o", "--output", default="models/model.pkl")
    p.add_argument("--encoder", default="auto", choices=["auto", "phobert", "tfidf", "none"])
    p.add_argument("--use-llm", action="store_true")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--report", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("score", help="Chấm bài bằng mô hình đã huấn luyện")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("-o", "--output", default="ket_qua.xlsx")
    p.add_argument("--no-feedback", action="store_true")
    p.add_argument("--feedback-dir", default=None)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("evaluate", help="Đánh giá hiệu năng so với điểm giám khảo")
    p.add_argument("-p", "--pred", required=True)
    p.add_argument("-g", "--gold", required=True)
    p.add_argument("-r", "--rubric", required=True)
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--qwk-hh", type=float, default=None, help="QWK giữa 2 giám khảo (để so chuẩn)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("demo", help="Chạy toàn bộ pipeline trên dữ liệu mô phỏng")
    p.add_argument("-o", "--output", default="demo_out")
    p.add_argument("-r", "--rubric", default=None)
    p.add_argument("-n", type=int, default=300)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("rag-index", help="Lập chỉ mục tài liệu RAG đã phê duyệt")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", default="artifacts/rag_documents.json")
    p.set_defaults(func=cmd_rag_index)

    p = sub.add_parser("train-qlora", help="Fine-tune Vistral-7B bằng QLoRA")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-g", "--gold", required=True)
    p.add_argument("-r", "--rubric", required=True)
    p.add_argument("-o", "--output", default="artifacts/vistral_qlora")
    p.add_argument("--model", default="Viet-Mistral/Vistral-7B-Chat")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-column", default="split")
    p.add_argument("--train-split", default="train")
    p.add_argument("--allow-unsplit", action="store_true",
                   help="Chỉ dùng khi -i và -g đã là các file train riêng")
    p.set_defaults(func=cmd_train_qlora)

    p = sub.add_parser("score-qlora", help="Chấm test set bằng adapter QLoRA")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-r", "--rubric", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--model", default="Viet-Mistral/Vistral-7B-Chat")
    p.add_argument("-o", "--output", default="reports/pred_qlora.csv")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-input-tokens", type=int, default=3072)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.set_defaults(func=cmd_score_qlora)

    p = sub.add_parser("prepare-dataset", help="Chuẩn hóa dataset Vietnamese IT Essays từ Kaggle")
    p.add_argument("--source-dir", required=True, help="Thư mục dataset Kaggle đã giải nén")
    p.add_argument("-o", "--output", default="data/imported/vietnamese_it_aes")
    p.set_defaults(func=cmd_prepare_dataset)

    p = sub.add_parser("dm-prepare", help="Ghép, kiểm tra đề và ẩn danh dữ liệu Toán Rời Rạc")
    p.add_argument("--source-dir", required=True, help="Thư mục chứa 5 clean_de_*.csv.zip")
    p.add_argument("--gold", required=True, help="Nhan_Vang_Hoan_Chinh.csv hoặc .xlsx")
    p.add_argument("--rubric-dir", default="data/toan_roi_rac/rubrics")
    p.add_argument("--catalog", default="data/toan_roi_rac/exam_catalog.yaml")
    p.add_argument("-o", "--output", default="data/toan_roi_rac/prepared")
    p.add_argument("--secret", required=True, help="Khóa HMAC riêng để ẩn danh")
    p.add_argument("--conflict-policy", choices=["quarantine", "correct", "trust-source"],
                   default="quarantine")
    p.add_argument("--id-map", default=None,
                   help="Tùy chọn: lưu bảng ánh xạ chứa PII ở vị trí bảo mật, ngoài repo")
    p.set_defaults(func=cmd_dm_prepare)

    p = sub.add_parser("dm-split", help="Chia train/validation/test theo sinh viên")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-g", "--gold", required=True)
    p.add_argument("-o", "--output", default="data/toan_roi_rac/splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.set_defaults(func=cmd_dm_split)

    p = sub.add_parser("dm-train-baseline", help="Huấn luyện baseline điểm tổng nhiều đề")
    p.add_argument("-i", "--input", required=True, help="essays_split.csv")
    p.add_argument("-g", "--gold", required=True, help="gold_split.csv")
    p.add_argument("-o", "--output", default="artifacts/toan_roi_rac/baseline")
    p.add_argument("--alpha", type=float, default=12.0)
    p.set_defaults(func=cmd_dm_train_baseline)

    p = sub.add_parser("dm-score-baseline", help="Chấm điểm tổng bằng baseline đã huấn luyện")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("-o", "--output", default="reports/toan_roi_rac/pred_baseline.csv")
    p.set_defaults(func=cmd_dm_score_baseline)

    p = sub.add_parser("dm-build-qlora", help="Tạo JSONL SFT từ nhãn điểm tổng")
    p.add_argument("-i", "--input", required=True, help="essays_split.csv")
    p.add_argument("-g", "--gold", required=True, help="gold_split.csv")
    p.add_argument("-o", "--output", default="data/toan_roi_rac/splits/qlora_train.jsonl")
    p.add_argument("--split", default="train")
    p.set_defaults(func=cmd_dm_build_qlora)

    p = sub.add_parser("dm-train-qlora", help="Fine-tune QLoRA điểm tổng trên RTX 5060 Ti")
    p.add_argument("--records", required=True, help="JSONL do dm-build-qlora tạo")
    p.add_argument("-o", "--output", default="artifacts/toan_roi_rac/qlora_total")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_dm_train_qlora)

    p = sub.add_parser("dm-score-qlora", help="Chấm test set bằng adapter QLoRA nhãn tổng")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("-o", "--output", default="reports/toan_roi_rac/pred_qlora_total.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-input-tokens", type=int, default=3072)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.set_defaults(func=cmd_dm_score_qlora)

    p = sub.add_parser("dm-evaluate", help="Đánh giá MAE/RMSE/QWK và độ ổn định nhiều lượt")
    p.add_argument("-p", "--pred", required=True)
    p.add_argument("-g", "--gold", required=True)
    p.add_argument("-o", "--output", default="reports/toan_roi_rac/evaluation")
    p.add_argument("--split", default="test")
    p.set_defaults(func=cmd_dm_evaluate)

    p = sub.add_parser("it04-prepare", help="Kiểm tra/làm sạch dataset chuẩn JSONL hoặc ZIP")
    p.add_argument("--source", required=True, help="Thư mục, JSONL hoặc ZIP dataset chuẩn")
    p.add_argument("--rubric-dir", default="data/toan_roi_rac/rubrics")
    p.add_argument("-o", "--output", default="runs/it04/01_data")
    p.add_argument("--min-chars", type=int, default=20)
    p.set_defaults(func=cmd_it04_prepare)

    p = sub.add_parser("it04-validate-rubric", help="Kiểm tra 5 rubric và xuất bảng tiêu chí")
    p.add_argument("--rubric-dir", default="data/toan_roi_rac/rubrics")
    p.add_argument("-o", "--output", default="runs/it04/02_rubric")
    p.set_defaults(func=cmd_it04_validate_rubric)

    p = sub.add_parser("it04-train", help="Train điểm tổng từ split đã khóa")
    p.add_argument("--prepared", default="runs/it04/01_data")
    p.add_argument("-o", "--output", default="runs/it04/03_model")
    p.add_argument("--alpha", type=float, default=12.0)
    p.set_defaults(func=cmd_it04_train)

    p = sub.add_parser("it04-grade", help="Chấm lai: mô hình điểm tổng + bằng chứng rubric")
    p.add_argument("-i", "--input", required=True, help="CSV bài cần chấm")
    p.add_argument("-m", "--model", required=True, help="total_baseline.pkl")
    p.add_argument("--rubric-dir", default="data/toan_roi_rac/rubrics")
    p.add_argument("-o", "--output", default="runs/it04/04_grading")
    p.add_argument("--rubric-weight", default="0.70", help="Số từ 0 đến 1")
    p.add_argument("--min-section-coverage", type=float, default=0.60)
    p.set_defaults(func=cmd_it04_grade)

    p = sub.add_parser("it04-evaluate", help="Đánh giá kết quả chấm trên các split")
    p.add_argument("--scores", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("-o", "--output", default="runs/it04/05_evaluation")
    p.set_defaults(func=cmd_it04_evaluate)

    p = sub.add_parser("it04-report", help="Tổng hợp báo cáo JSON/Markdown/HTML theo bước")
    p.add_argument("--root", default="runs/it04")
    p.add_argument("-o", "--output", default="runs/it04/06_reports")
    p.set_defaults(func=cmd_it04_report)

    p = sub.add_parser("it04-run-all", help="Chạy chuẩn hóa → rubric → train → chấm → báo cáo")
    p.add_argument("--source", required=True, help="Thư mục, JSONL hoặc ZIP dataset chuẩn")
    p.add_argument("--rubric-dir", default="data/toan_roi_rac/rubrics")
    p.add_argument("-o", "--output", default="runs/it04")
    p.add_argument("--alpha", type=float, default=12.0)
    p.add_argument("--rubric-weight", default="auto",
                   help="auto: chọn bằng validation MAE; hoặc số từ 0 đến 1")
    p.set_defaults(func=cmd_it04_run_all)

    p = sub.add_parser("split-dataset", help="Chia train/validation/test không rò rỉ")
    p.add_argument("-i", "--input", required=True, help="Dữ liệu bài luận đã làm sạch")
    p.add_argument("-g", "--gold", required=True, help="Nhãn vàng đã phân xử")
    p.add_argument("-o", "--output", default="data/splits")
    p.add_argument("--group-column", default="student_hash")
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--validation-size", type=float, default=0.10)
    p.add_argument("--holdout-prompt", default=None,
                   help="Khóa toàn bộ prompt_id này làm test ngoài đề")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_split_dataset)

    p = sub.add_parser("hardware-check", help="Kiểm tra RTX 50, CUDA 12.8 và QLoRA 16GB")
    p.add_argument("--strict-blackwell", action="store_true")
    p.add_argument("--json", default=None, help="Lưu kết quả kiểm tra JSON")
    p.set_defaults(func=cmd_hardware_check)

    p = sub.add_parser("sus", help="Tính điểm và độ tin cậy khảo sát SUS")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", default=None)
    p.set_defaults(func=cmd_sus)

    p = sub.add_parser("bertscore", help="Đánh giá phản hồi sinh bằng BERTScore")
    p.add_argument("--generated", required=True)
    p.add_argument("--references", required=True)
    p.add_argument("--generated-column", default="feedback")
    p.add_argument("--reference-column", default="feedback")
    p.add_argument("--id-column", default="essay_id")
    p.add_argument("--model", default="bert-base-multilingual-cased")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("-o", "--output", default="reports/bertscore.json")
    p.set_defaults(func=cmd_bertscore)

    p = sub.add_parser("ablation-plan", help="Xuất ma trận thí nghiệm A–H")
    p.add_argument("-o", "--output", default="reports/ablation/ablation_plan.csv")
    p.set_defaults(func=cmd_ablation_plan)

    p = sub.add_parser("ablation-evaluate", help="Tổng hợp metric các dự đoán A–H")
    p.add_argument("-g", "--gold", required=True)
    p.add_argument("--pred-dir", default="reports/ablation")
    p.add_argument("--pattern", default="pred_*.csv")
    p.add_argument("--id-column", default="essay_id")
    p.add_argument("--gold-column", default="gold_total")
    p.add_argument("--pred-column", default="total")
    p.add_argument("-o", "--output", default="reports/ablation/ablation_summary.csv")
    p.set_defaults(func=cmd_ablation_evaluate)

    p = sub.add_parser("hf-push", help="Đẩy adapter/artifact lên Hugging Face Hub")
    p.add_argument("--path", required=True)
    p.add_argument("--repo", required=True, help="username/repository")
    p.add_argument("--repo-type", choices=["model", "dataset", "space"], default="model")
    p.add_argument("--revision", default="main")
    p.add_argument("--public", action="store_true")
    p.set_defaults(func=cmd_hf_push)

    p = sub.add_parser("moodle", help="Đọc bài tập/bài nộp hoặc gửi điểm Moodle")
    moodle_sub = p.add_subparsers(dest="moodle_action", required=True)
    m = moodle_sub.add_parser("assignments")
    m.add_argument("--course-id", type=int, required=True)
    m.add_argument("-o", "--output", default=None)
    m.set_defaults(func=cmd_moodle)
    m = moodle_sub.add_parser("submissions")
    m.add_argument("--assignment-id", type=int, required=True)
    m.add_argument("-o", "--output", default=None)
    m.set_defaults(func=cmd_moodle)
    m = moodle_sub.add_parser("push-grade")
    m.add_argument("--assignment-id", type=int, required=True)
    m.add_argument("--user-id", type=int, required=True)
    m.add_argument("--grade", type=float, required=True)
    m.add_argument("--feedback", default="")
    m.add_argument("--approved", action="store_true")
    m.add_argument("-o", "--output", default=None)
    m.set_defaults(func=cmd_moodle)

    p = sub.add_parser("serve", help="Chạy VieGrader FastAPI")
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--enable-moodle", action="store_true",
                   help="Đọc MOODLE_BASE_URL và MOODLE_TOKEN từ biến môi trường")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("gui", help="Chạy giao diện Gradio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Tạo liên kết Gradio tạm thời")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("ubuntu-gui", help="Chạy GUI theo config Ubuntu Server")
    p.add_argument("--config", default="config.ubuntu.example.json")
    p.set_defaults(func=cmd_ubuntu_gui)

    p = sub.add_parser("report", help="Tổng hợp và xuất báo cáo kết quả các giai đoạn")
    p.add_argument("-i", "--input", nargs="+", required=True, help="Danh sách tệp kết quả")
    p.add_argument("-o", "--output", default="reports/stage_report")
    p.add_argument("--project", default="VieGrader – NCKH 2026")
    p.add_argument("--author", default="")
    p.add_argument("--formats", default="xlsx,docx,html,json",
                   help="xlsx,docx,pdf,html,json")
    p.set_defaults(func=cmd_report)

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        return a.func(a)
    except KeyboardInterrupt:
        _p("\nĐã dừng theo yêu cầu.")
        return 130
    except Exception as exc:
        _p(f"LỖI: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

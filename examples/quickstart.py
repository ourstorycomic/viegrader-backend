"""Ví dụ dùng viegrader như thư viện — chạy: python examples/quickstart.py"""

from pathlib import Path

import pandas as pd

from viegrader import Grader, GraderConfig, default_rubric_path, load_rubric
from viegrader.cleaning import clean_dataframe
from viegrader.evaluation import evaluation_report, suggest_review_threshold
from viegrader.labeling import agreement_report, resolve, snap_to_rubric
from viegrader.simulate import make_dataset

OUT = Path("quickstart_out")
OUT.mkdir(exist_ok=True)

rubric = load_rubric(default_rubric_path())

# --- 1. Dữ liệu (thay bằng pd.read_csv("bai_lam_that.csv") khi có dữ liệu thật)
essays, _, annotations = make_dataset(rubric, n=300, seed=1)

# --- 2. Làm sạch
clean, rejected, report = clean_dataframe(essays)
print(report.summary())
print()

# --- 3. Đo đồng thuận giám khảo -> nhãn vàng
print(agreement_report(annotations, ["total"], rubric.scale_step).round(3).to_string(index=False))
gold = snap_to_rubric(resolve(annotations, rubric), rubric)

# --- 4. Ghép và chia dữ liệu
df = clean.merge(gold[["essay_id"]], on="essay_id", how="inner")
gold = gold.set_index("essay_id").loc[df["essay_id"]].reset_index()
n_test = 60
tr, te = df.iloc[:-n_test].reset_index(drop=True), df.iloc[-n_test:].reset_index(drop=True)
gtr, gte = gold.iloc[:-n_test].reset_index(drop=True), gold.iloc[-n_test:].reset_index(drop=True)

# --- 5. Huấn luyện
#   encoder="phobert" nếu đã cài viegrader[deep]; use_llm=True nếu có ANTHROPIC_API_KEY
g = Grader(rubric, GraderConfig(encoder="tfidf", use_llm=False))
g.set_keywords({
    pid: str(sub["keywords"].iloc[0]).split(";")
    for pid, sub in df.groupby("prompt_id") if sub["keywords"].iloc[0]
})
g.fit(tr, gtr, val_df=te, val_gold=gte)
g.save(OUT / "model.pkl")
print()
print(g.scorer.report())

# --- 6. Chấm và đánh giá
pred = g.score_to_frame(te)
pred.to_csv(OUT / "ket_qua.csv", index=False, encoding="utf-8-sig")
print()
print(evaluation_report(gte, pred, [c.key for c in rubric.criteria], rubric.scale_step))

sug = suggest_review_threshold(gte["gold_total"], pred["total"], pred["confidence"])
print(f"\nNgưỡng phúc tra đề xuất: confidence < {sug['threshold']:.2f} "
      f"→ phúc tra {sug['review_rate']:.1%}, bắt {sug['recall_of_errors']:.1%} lỗi")

print("\n--- Phản hồi mẫu ---")
print(pred["feedback"].iloc[0])

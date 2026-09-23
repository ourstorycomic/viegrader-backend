"""Bộ kiểm thử tối thiểu cho viegrader.  Chạy: pytest -q"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from viegrader import load_rubric, default_rubric_path, Grader, GraderConfig
from viegrader.cleaning import clean_dataframe, clean_text, anonymize, assess
from viegrader.cleaning.dedup import find_duplicates, prompt_copy_ratio
from viegrader.features import extract_all, is_valid_syllable
from viegrader.labeling import quadratic_weighted_kappa, resolve, agreement_report
from viegrader.scoring.rules import UnsafeExpression, check_rules, safe_eval
from viegrader.simulate import make_dataset


@pytest.fixture(scope="module")
def rubric():
    return load_rubric(default_rubric_path())


# --------------------------- Chuẩn hoá ------------------------------------- #
def test_unicode_and_tone_style():
    a, _ = clean_text("hòa bình thủy lợi")
    assert "hoà" in a and "thuỷ" in a


def test_teencode_expansion():
    out, rep = clean_text("Sv ko lm dc bài này vs lý do sk.")
    assert "không" in out and rep.as_dict().get("teencode_expanded", 0) > 0


def test_repeat_and_emoji():
    out, _ = clean_text("hayyyy quá 😀😀 !!!!")
    assert "hayyy" not in out and "😀" not in out and "!!!" not in out


def test_pii_anonymize():
    t = "Họ và tên: Nguyễn Văn A\nMSSV: 21A4010123\nEmail: a@hou.edu.vn\nNội dung bài làm."
    out, rep = anonymize(t)
    assert "Nguyễn Văn A" not in out
    assert "21A4010123" not in out
    assert "a@hou.edu.vn" not in out
    assert rep.student_id_found == "21A4010123"


# --------------------------- Chính tả -------------------------------------- #
@pytest.mark.parametrize("w", ["nghiêng", "khuyến", "trường", "quyền", "nguyễn", "giáo"])
def test_valid_syllables(w):
    assert is_valid_syllable(w)


@pytest.mark.parametrize("w", ["vaf", "cuar", "nhg", "xzq", "kbc"])
def test_invalid_syllables(w):
    assert not is_valid_syllable(w)


# --------------------------- Chất lượng ------------------------------------ #
def test_quality_reject_empty():
    assert assess("").decision == "reject"


def test_quality_reject_no_diacritics():
    t = "toi di hoc moi ngay va rat vui ve cung ban be trong lop hoc " * 6
    assert assess(t).decision == "reject"


def test_quality_keep_normal():
    t = ("Chuyển đổi số là quá trình ứng dụng công nghệ vào hoạt động quản trị. "
         "Doanh nghiệp cần xây dựng cơ sở dữ liệu và chuẩn hoá quy trình. ") * 6
    assert assess(t).decision in ("keep", "flag")


# --------------------------- Trùng lặp ------------------------------------- #
def test_duplicate_detection():
    base = "Chuyển đổi số giúp doanh nghiệp nâng cao hiệu quả quản trị và ra quyết định nhanh hơn. " * 4
    docs = {"a": base, "b": base + " Ngoài ra còn nhiều lợi ích khác.", "c": "Kế toán dồn tích ghi nhận doanh thu khi phát sinh giao dịch. " * 4}
    pairs = find_duplicates(docs, threshold=0.5)
    ids = {tuple(sorted((p.id_a, p.id_b))) for p in pairs}
    assert ("a", "b") in ids
    assert all("c" not in p for p in ids)


def test_prompt_copy_ratio():
    prompt = "Trình bày khái niệm chuyển đổi số và phân tích tác động của nó."
    assert prompt_copy_ratio(prompt * 3, prompt) > 0.7
    assert prompt_copy_ratio("Nội dung hoàn toàn khác biệt về kế toán dồn tích.", prompt) < 0.2


# --------------------------- Đặc trưng ------------------------------------- #
def test_features_basic():
    t = ("Trước hết, chuyển đổi số là một xu hướng tất yếu. Tuy nhiên, doanh nghiệp "
         "cần chuẩn bị nguồn lực. Tóm lại, đây là vấn đề quan trọng.")
    f = extract_all(t, keywords=["chuyển đổi số", "nguồn lực", "khách hàng"])
    assert f["n_words"] > 10
    assert f["keyword_coverage"] == pytest.approx(2 / 3, abs=0.01)
    assert f["connective_variety"] >= 2


# --------------------------- Rule engine ----------------------------------- #
def test_safe_eval_blocks_unsafe():
    with pytest.raises(UnsafeExpression):
        safe_eval("__import__('os').system('ls')", {})
    with pytest.raises(UnsafeExpression):
        safe_eval("open('/etc/passwd').read()", {})


def test_safe_eval_works():
    assert safe_eval("n_words < 0.5 * min_words", {"n_words": 100, "min_words": 300}) is True


def test_rules_valid(rubric):
    assert check_rules(rubric) == []


# --------------------------- Đồng thuận ------------------------------------ #
def test_qwk_perfect():
    y = [1, 2, 3, 4, 5, 6, 7, 8]
    assert quadratic_weighted_kappa(y, y, step=1.0) == pytest.approx(1.0)


def test_qwk_lower_when_noisy():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 10, 200).astype(float)
    noisy = y + rng.normal(0, 2, 200)
    assert quadratic_weighted_kappa(y, noisy, step=1.0) < 0.95


# --------------------------- End-to-end ------------------------------------ #
def test_end_to_end(rubric):
    essays, gold_true, ann = make_dataset(rubric, n=120, seed=7)
    clean, rejected, rep = clean_dataframe(essays)
    assert rep.n_output > 100
    assert rep.n_rejected >= 2                    # các trường hợp biên bị loại

    ar = agreement_report(ann, ["total"], rubric.scale_step)
    assert ar.loc[0, "QWK_mean"] > 0.5

    gold = resolve(ann, rubric)
    df = clean.merge(gold[["essay_id"]], on="essay_id", how="inner")
    g_al = gold.set_index("essay_id").loc[df["essay_id"]].reset_index()

    tr, te = df.iloc[:-25], df.iloc[-25:].reset_index(drop=True)
    gtr, gte = g_al.iloc[:-25], g_al.iloc[-25:].reset_index(drop=True)

    g = Grader(rubric, GraderConfig(encoder="tfidf", use_llm=False))
    g.fit(tr.reset_index(drop=True), gtr.reset_index(drop=True))
    out = g.score_to_frame(te)

    assert len(out) == len(te)
    assert out["total"].between(0, rubric.scale_max).all()
    # Điểm phải là bội của bước làm tròn
    assert np.allclose((out["total"] / rubric.scale_step) % 1, 0)
    assert "feedback" in out.columns and out["feedback"].str.len().min() > 50


def test_rule_zero_for_empty_essay(rubric):
    essays, gold, ann = make_dataset(rubric, n=60, seed=3)
    g = Grader(rubric, GraderConfig(encoder="tfidf"))
    g.fit(essays.iloc[:50].reset_index(drop=True), gold.iloc[:50].reset_index(drop=True))
    res = g.score(pd.DataFrame([{
        "essay_id": "X1", "text": "", "prompt_id": "P01", "prompt_text": "Đề bài."
    }]))
    assert res[0].total == 0.0
    assert "BAI_TRONG" in res[0].flags


def test_save_load_roundtrip(rubric, tmp_path):
    essays, gold, _ = make_dataset(rubric, n=60, seed=11)
    g = Grader(rubric, GraderConfig(encoder="tfidf"))
    g.fit(essays.iloc[:50].reset_index(drop=True), gold.iloc[:50].reset_index(drop=True))
    p = tmp_path / "m.pkl"
    g.save(p)
    g2 = Grader.load(p)
    a = g.score_to_frame(essays.iloc[50:55].reset_index(drop=True), with_feedback=False)
    b = g2.score_to_frame(essays.iloc[50:55].reset_index(drop=True), with_feedback=False)
    pd.testing.assert_series_equal(a["total"], b["total"])

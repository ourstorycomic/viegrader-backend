from __future__ import annotations

from pathlib import Path


REPORT_STAGES = {
    "data": "1. Thu thập, làm sạch và kiểm định dữ liệu",
    "labeling": "2. Gán nhãn, đồng thuận và phân xử",
    "baseline": "3. Baseline TF-IDF và PhoBERT",
    "qlora": "4. Vistral-7B QLoRA và Hugging Face",
    "rag": "5. Kho tri thức và truy xuất RAG",
    "scoring": "6. Chấm điểm và phản hồi",
    "ablation": "7. Thực nghiệm ablation A–H",
    "evaluation": "8. Đánh giá hiệu năng và độ tin cậy",
    "sus": "9. Đánh giá khả dụng SUS",
    "moodle": "10. Tích hợp và vận hành Moodle",
    "other": "Phụ lục và kết quả khác",
}


KEYWORDS = {
    "data": ("clean", "rejected", "raw", "processed", "quality", "dedup", "lam_sach"),
    "labeling": ("annotation", "agreement", "adjud", "gold", "rater", "dong_thuan", "phan_xu"),
    "baseline": ("baseline", "tfidf", "phobert", "trait", "feature_importance"),
    "qlora": ("qlora", "adapter", "training_config", "manifest", "vistral", "huggingface"),
    "rag": ("rag", "retrieval", "chunk", "document_index", "context_audit"),
    "scoring": ("prediction", "score", "ket_qua", "feedback", "phan_hoi"),
    "ablation": ("ablation", "experiment_a", "experiment_b", "experiment_h"),
    "evaluation": ("evaluation", "metric", "qwk", "mae", "bootstrap", "bertscore", "fairness"),
    "sus": ("sus", "usability", "kha_dung"),
    "moodle": ("moodle", "deployment", "api_audit", "grade_push"),
}


def detect_stage(path: str | Path) -> str:
    name = Path(path).name.lower()
    for stage, words in KEYWORDS.items():
        if any(word in name for word in words):
            return stage
    return "other"

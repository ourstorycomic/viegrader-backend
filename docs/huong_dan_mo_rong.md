# VieGrader 0.2 — QLoRA, RAG, ablation, BERTScore/SUS và Moodle

## 1. Cài đặt theo nhu cầu

```bash
pip install -e .
pip install -e ".[deep]"       # PhoBERT baseline
pip install -e ".[qlora]"      # Vistral-7B + QLoRA
pip install -e ".[evaluation]" # BERTScore
pip install -e ".[api,moodle]" # FastAPI + Moodle
```

Trên Hugging Face cần chấp thuận điều kiện truy cập
`Viet-Mistral/Vistral-7B-Chat` và khai báo token bằng biến môi trường. Không ghi
token vào notebook hoặc mã nguồn.

## 2. Baseline

```bash
viegrader train -i data/clean.csv -g data/gold.csv -r rubric.yaml \
  -o artifacts/tfidf.pkl --encoder tfidf

viegrader train -i data/clean.csv -g data/gold.csv -r rubric.yaml \
  -o artifacts/phobert.pkl --encoder phobert
```

## 3. Kho RAG

Chuẩn bị `rag_documents.csv` với các cột `chunk_id,text,document_id,course_id,
prompt_id,section,version,approved` rồi chạy:

```bash
viegrader rag-index -i rag_documents.csv -o artifacts/rag_documents.json
```

Chỉ tài liệu có `approved=true` được lập chỉ mục. Hàm `attach_rag_context` trả
đồng thời dữ liệu đã gắn ngữ cảnh và bảng audit chứa `chunk_id`, điểm truy xuất,
phiên bản tài liệu.

## 4. QLoRA Vistral-7B

```bash
viegrader train-qlora -i data/train.csv -g data/gold_train.csv -r rubric.yaml \
  -o artifacts/vistral_qlora --epochs 3 --max-length 1536 \
  --batch-size 1 --grad-accum 8
```

Đầu ra là adapter PEFT, tokenizer và `training_config.json`. Không lưu LLM bằng
pickle. Tách validation/test theo sinh viên hoặc đề bài trước khi gọi lệnh này.

## 5. Ablation A–H

Ma trận mặc định:

| ID | Cấu hình |
|---|---|
| A | TF-IDF + mô hình đặc trưng |
| B | PhoBERT + mô hình đặc trưng |
| C | Vistral zero-shot |
| D | Vistral + bài neo |
| E | Vistral + bài neo + RAG |
| F | Vistral QLoRA |
| G | Vistral QLoRA + RAG |
| H | PhoBERT + QLoRA + RAG + bài neo + self-consistency + rule engine |

`AblationRunner` bắt buộc mọi cấu hình dùng chung test set và lưu dự đoán, cấu
hình, seed cùng báo cáo. Có thể thay ma trận bằng cấu hình đúng trong đề cương.

## 6. BERTScore và SUS

```python
from viegrader.evaluation import bertscore_feedback
metrics = bertscore_feedback(machine_feedback, teacher_feedback)
```

BERTScore chỉ dùng cho phản hồi có bản tham chiếu; QWK/ICC/MAE vẫn là chỉ số
đánh giá điểm.

Tệp SUS cần các cột `sus_1` đến `sus_10`, thang Likert 1–5:

```bash
viegrader sus -i survey_sus.xlsx -o reports/sus.json
```

## 7. API và Moodle

```bash
viegrader serve -m artifacts/tfidf.pkl --host 0.0.0.0 --port 8000
```

Các endpoint chính: `/health`, `/v1/score`, `/v1/batch-score`,
`/v1/moodle/push-grade`. `MoodleClient.save_grade()` từ chối ghi điểm nếu
`approved=False`. Khi vận hành cần HTTPS, token quyền tối thiểu, nhật ký kiểm
toán và không gửi PII vào mô hình.

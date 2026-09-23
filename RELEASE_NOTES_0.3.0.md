# VieGrader 0.3.0

## Giao diện

- Gradio GUI gồm 10 tab, chạy local, Kaggle hoặc Hugging Face Spaces.
- Hỗ trợ xem trước dữ liệu, train TF-IDF/PhoBERT, train QLoRA, RAG, chấm điểm,
  đánh giá, ablation, SUS và Moodle.
- Tác vụ GPU chạy qua hàng đợi với concurrency bằng 1.

## Hugging Face

- Xác thực bằng `HF_TOKEN` từ Secrets/biến môi trường.
- Tải base model gated từ Hub.
- Tạo model repository private/public.
- Đẩy adapter, tokenizer, manifest, training config và model card lên Hub.
- Trả adapter ZIP sau huấn luyện.
- Có `app.py`, metadata Space và `requirements.txt` để triển khai Gradio Space.

## An toàn

- Không nhận hoặc hiển thị token trong GUI.
- Xác thực Hub trước khi bắt đầu tải/huấn luyện model lớn.
- Moodle vẫn yêu cầu giảng viên phê duyệt trước khi ghi điểm.
- Khuyến nghị repository private và dữ liệu đã ẩn danh.

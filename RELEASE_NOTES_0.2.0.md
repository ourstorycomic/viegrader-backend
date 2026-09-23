# VieGrader 0.2.0

## Nội dung mới

- Giao diện `ScoringBackend` và `BackendPrediction`; `Grader.add_backend()` cho
  phép hợp nhất backend tùy biến với TF-IDF/PhoBERT và rule engine hiện có.
- `VistralBackend` hỗ trợ base model hoặc adapter QLoRA 4-bit.
- Pipeline tạo instruction records và huấn luyện QLoRA bằng PEFT/TRL.
- RAG TF-IDF nhẹ, lọc tài liệu được phê duyệt, ràng buộc course/prompt và bảng audit.
- Cấu hình ablation A–H cùng runner lưu dự đoán/cấu hình/seed.
- Bootstrap confidence interval và BERTScore cho phản hồi.
- Tính SUS 10 câu và Cronbach's alpha.
- FastAPI chấm đơn/lô và Moodle REST client.
- Bắt buộc `approved=True` trước khi gửi điểm Moodle.
- Artifact manifest cho adapter QLoRA.

## Xác minh

- `python -m compileall -q viegrader`: đạt.
- `python -m pytest -q`: 37 kiểm thử đạt.
- Demo end-to-end 100 bài mô phỏng với TF-IDF: chạy thành công.

QLoRA đầy đủ cần GPU CUDA, quyền truy cập Vistral-7B và chưa được chạy huấn luyện
trong môi trường kiểm thử CPU này. Các module GPU được thiết kế theo cơ chế
optional dependency và lazy loading.

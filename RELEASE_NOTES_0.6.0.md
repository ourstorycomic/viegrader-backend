# VieGrader 0.6.0 — Toán Rời Rạc

- Thêm pipeline nhiều đề `dm-*` cho năm đề IT04.
- Ghép 166 nhãn vàng với năm clean set, ẩn danh HMAC và loại PII khỏi artifact.
- Phát hiện xung đột nội dung–mã đề; mặc định cách ly thay vì tự sửa.
- Chia train/validation/test không rò rỉ sinh viên.
- Baseline điểm tổng TF-IDF + Ridge và QLoRA nhãn tổng.
- Đánh giá nhiều lượt: MAE, RMSE, QWK, Pearson, bias và test–retest.
- Chặn diễn giải sai: dữ liệu hiện chưa có nhãn vàng từng câu.


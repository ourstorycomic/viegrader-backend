# VieGrader 0.4.0 — Báo cáo kết quả các giai đoạn

## Chức năng mới

- Tự nhận diện kết quả theo 10 giai đoạn nghiên cứu.
- Bảng kiểm tra giai đoạn đã có kết quả hoặc còn thiếu.
- Xuất Excel nhiều sheet, Word, PDF Unicode tiếng Việt, HTML và JSON manifest.
- Đóng gói mọi báo cáo vào một tệp ZIP.
- GUI tab **Xuất báo cáo** nhận nhiều tệp kết quả cùng lúc.
- CLI `viegrader report` cho pipeline tự động trên Kaggle/Hugging Face.
- Giới hạn số dòng đưa vào Word/PDF/HTML để báo cáo vẫn đọc được; Excel giữ bảng đầy đủ.
- Không tự động sao chép dữ liệu thô hoặc định danh sinh viên vào gói báo cáo.

## Các giai đoạn

1. Dữ liệu và làm sạch.
2. Gán nhãn và đồng thuận.
3. Baseline TF-IDF/PhoBERT.
4. QLoRA và Hugging Face.
5. RAG.
6. Chấm điểm và phản hồi.
7. Ablation A–H.
8. Đánh giá hiệu năng.
9. SUS.
10. Moodle.

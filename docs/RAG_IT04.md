# RAG cục bộ cho chấm Toán Rời Rạc IT04

Chỉ mục `data/toan_roi_rac/rag_it04.json` được lập từ hai PDF người dùng cung cấp:
`BaiGiang_ToanRR.pdf` và `TRR_NguyenDucNghia.pdf`. Mỗi đoạn lưu tên tài liệu,
trang PDF (đếm từ 1), SHA-256 của tệp nguồn và nội dung trích xuất. Hệ thống
dùng TF-IDF trên CPU để truy xuất, không gửi bài làm hoặc giáo trình ra ngoài.

## Tạo lại chỉ mục sau khi kiểm tra tài liệu

```bash
python -m pip install -e '.[rag,teacher-portal,qlora]'
python -m viegrader.rag.pdf_index \
  data/toan_roi_rac/references/BaiGiang_ToanRR.pdf \
  data/toan_roi_rac/references/TRR_NguyenDucNghia.pdf \
  --course-id IT04 --output data/toan_roi_rac/rag_it04.json
```

Chỉ nạp tài liệu được phép sử dụng và đã kiểm tra nội dung. Trang ảnh quét
không có đủ chữ bị bỏ qua; cần OCR và đối chiếu ký hiệu toán trước khi tạo lại
chỉ mục. Thay PDF sẽ đổi mã băm và đòi hỏi tạo lại chỉ mục.

## Bật trên cổng giảng viên

```bash
export VIEGRADER_RAG_INDEX="$(pwd)/data/toan_roi_rac/rag_it04.json"
uvicorn viegrader.teacher_portal:app --host 127.0.0.1 --port 8088
```

RAG chỉ áp dụng cho lô `qlora` khi biến trên được đặt. Mỗi bài truy xuất tối
đa ba đoạn; hệ thống thử thêm từng đoạn theo ngân sách token của chat template.
Nếu không còn chỗ cho một đoạn hoặc không tìm được tài liệu, lô báo lỗi để
giảng viên xem xét. Mỗi lô ghi `jobs/<job_id>/rag_audit.json` gồm mã băm chỉ
mục và danh sách đoạn thực sự dùng: mã đoạn, tên tệp, trang, mã băm PDF và
điểm truy xuất. `input.csv` cũng lưu `rag_context` dùng để chấm.

Đoạn truy xuất là nguồn đối chiếu, không phải đáp án chuẩn cho từng mã đề.
Giảng viên vẫn phải duyệt điểm cuối cùng; cần đánh giá đối chứng trên cùng
tập bài trước khi kết luận RAG cải thiện MAE hoặc QWK. Nhánh hybrid hiện không
dùng các đoạn truy xuất này.

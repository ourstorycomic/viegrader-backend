# Hướng dẫn xuất báo cáo kết quả các giai đoạn

## 1. Giai đoạn được hỗ trợ

Hệ thống tổng hợp 10 nhóm kết quả: dữ liệu; gán nhãn; baseline; QLoRA/Hugging
Face; RAG; chấm điểm; ablation; đánh giá; SUS; Moodle. Tên giai đoạn được nhận
diện từ tên tệp, ví dụ `clean_report.json`, `agreement.csv`,
`baseline_evaluation.txt`, `training_config_qlora.json`,
`rag_context_audit.csv`, `ablation_summary.csv`, `evaluation_metrics.csv`,
`sus_report.json`, `moodle_grade_push_audit.json`.

## 2. Xuất từ GUI

1. Mở tab **Xuất báo cáo**.
2. Tải đồng thời các tệp CSV, XLSX, JSON, TXT, Markdown hoặc YAML.
3. Nhập tên dự án và tác giả.
4. Chọn Excel, Word, PDF, HTML hoặc JSON.
5. Chọn **Tạo báo cáo các giai đoạn**.
6. Kiểm tra bảng trạng thái và tải gói ZIP.

Excel giữ toàn bộ bảng dữ liệu đầu vào. Word giới hạn 30 dòng mỗi bảng, PDF 25
dòng và HTML 100 dòng để báo cáo dễ đọc. JSON lưu manifest và thống kê tóm tắt.

## 3. Xuất bằng CLI

```bash
viegrader report \
  -i reports/clean_report.json \
     reports/agreement.csv \
     reports/baseline_evaluation.txt \
     reports/ablation_summary.csv \
     reports/evaluation_metrics.csv \
     reports/sus_report.json \
  -o reports/final \
  --project "VieGrader – NCKH 2026" \
  --author "Nhóm nghiên cứu" \
  --formats xlsx,docx,pdf,html,json
```

Kết quả gồm:

```text
bao_cao_cac_giai_doan.xlsx
bao_cao_cac_giai_doan.docx
bao_cao_cac_giai_doan.pdf
bao_cao_cac_giai_doan.html
bao_cao_cac_giai_doan.json
final_BaoCao_VieGrader.zip
```

## 4. Bảo vệ dữ liệu

Gói báo cáo không tự động sao chép các tệp nguồn. Không tải bài làm thô hoặc PII
lên báo cáo công khai. Nhật ký Moodle dùng hash rút gọn của user ID và không lưu
token hay nội dung phản hồi. Khi công bố, cần kiểm tra lại bảng Excel vì bảng này
có thể giữ toàn bộ dòng từ tệp kết quả đã chọn.

# VieGrader 0.7.0 – Train và chấm tự động Toán Rời Rạc IT04

Phiên bản này chạy trên Ubuntu Server, đọc trực tiếp dataset chuẩn VieGrader
JSONL/ZIP, train mô hình điểm tổng, chấm theo rubric của từng mã đề, tạo hàng
đợi phúc tra và xuất báo cáo cho từng giai đoạn.

## 1. Phạm vi và nguyên tắc dữ liệu

Bộ dữ liệu IT04 đã chuẩn hóa có 735 bài nộp, trong đó 712 bài có văn bản dùng
được, 628 bản ghi đủ điều kiện huấn luyện và 623 bản ghi sẵn sàng thực thi
VieGrader. Dataset đã khóa split và xử lý bản trùng/xung đột nhãn; pipeline này
không tự đưa các bản bị loại trở lại tập train.

Nhãn gốc chỉ có điểm tổng. Vì vậy:

- mô hình chỉ học `gold_total` trên thang 0–10;
- không suy diễn điểm câu thành nhãn huấn luyện;
- `item_scores.csv` là ước lượng có bằng chứng của rule engine và mang nhãn
  `rule_based_estimate`;
- bài không tách đủ câu, mã đề không rõ hoặc mô hình/rubric lệch từ 2 điểm được
  chuyển vào `human_review_queue.csv`;
- giảng viên phải duyệt trước khi công bố điểm chính thức.

## 2. Luồng xử lý

1. Đọc JSONL/ZIP và ánh xạ schema về `essay_id`, `text`, `exam_id`, `split`,
   `gold_total`, `training_eligible`.
2. Chuẩn hóa Unicode NFC, xuống dòng, khoảng trắng; kiểm tra văn bản, mã đề,
   điểm, split, ID trùng và cờ sẵn sàng.
3. Giữ nguyên split đã khóa; chỉ `split=train AND training_eligible=true` được
   dùng để fit.
4. Kiểm tra đủ 5 rubric, mỗi rubric 10 tiêu chí, tổng điểm 10 và tổng trọng số 1.
5. Train baseline `TF-IDF word+char + Ridge`, đánh giá riêng train/validation/test.
6. Tách câu trả lời, đối chiếu đáp án/rubric, lưu điểm câu và bằng chứng.
7. Khi độ bao phủ câu đạt ngưỡng, tính điểm lai:

   `final_total = w × rubric_total + (1 − w) × model_total`

   với `w` được chọn tự động trong 0.0…1.0 theo MAE của validation, tuyệt đối
   không nhìn nhãn test, rồi làm tròn 0.25. Khi không đủ độ bao phủ, dùng điểm
   mô hình và bắt buộc phúc tra. Bài trống/không đủ độ dài nhận 0 theo rule.
8. Xuất CSV, JSON, Markdown và HTML; metric trên test là kết quả chính.

## 3. Cấu trúc đầu vào

Có thể truyền ZIP dataset chuẩn, thư mục đã giải nén chứa `dataset/*.jsonl`,
hoặc đường dẫn trực tiếp tới tệp `.jsonl`. Schema khuyến nghị:

```json
{
  "essay_id": "IT04-...",
  "exam_code": "DE01",
  "answer_text": "Câu 1a: ...",
  "split": "train",
  "training_eligible": true,
  "viegrader_ready": true,
  "labels": {"score_10": 7.5, "grade_band": "Kha"},
  "flags": []
}
```

Loader cũng chấp nhận alias phổ biến. Rubric nằm tại
`data/toan_roi_rac/rubrics/`; tên file có thể là `rubric_de_1.yaml` hoặc
`rubric_de_01.yaml`.

## 4. Cài đặt trên Ubuntu Server

### Baseline CPU

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip unzip
sudo mkdir -p /opt/viegrader
sudo unzip VieGrader_0.7.0_IT04_Ubuntu.zip -d /opt/viegrader
sudo chown -R "$USER":"$USER" /opt/viegrader
cd /opt/viegrader

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[reports,docs]'
pytest -q
```

### QLoRA GPU tùy chọn

Máy phải có driver NVIDIA phù hợp. Với máy RTX 50 và CUDA 12.8:

```bash
source /opt/viegrader/.venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio
python -m pip install -e '.[qlora,reports,docs]'
viegrader hardware-check --strict-blackwell
```

Không cần cài QLoRA để chạy pipeline baseline/rubric.

## 5. Cấu hình

```bash
cd /opt/viegrader
cp config.it04.example.env config.it04.env
nano config.it04.env
chmod +x scripts/run_it04_pipeline.sh
```

| Biến | Ý nghĩa |
|---|---|
| `IT04_DATASET` | ZIP, thư mục hoặc JSONL dataset chuẩn |
| `IT04_RUBRICS` | Thư mục chứa 5 rubric |
| `IT04_RUN` | Thư mục output của một lần chạy |
| `IT04_ALPHA` | Hệ số Ridge, mặc định 12 |
| `IT04_RUBRIC_WEIGHT` | `auto` (khuyến nghị) hoặc trọng số cố định 0…1 |

## 6. Chạy nhanh một lệnh

```bash
source .venv/bin/activate
./scripts/run_it04_pipeline.sh all
```

Lệnh trên chạy chuẩn hóa → kiểm tra rubric → train → chấm → đánh giá → báo cáo.
Nếu một bước lỗi, shell dừng ngay và không giả lập bước thành công.

## 7. Chạy và kiểm tra từng bước

### Bước 1 – Kiểm tra rubric

```bash
./scripts/run_it04_pipeline.sh check
cat "$IT04_RUN/02_rubric/rubric_audit.json"
```

Output: `rubric_audit.json`, `rubric_items.csv`.

### Bước 2 – Chuẩn hóa và làm sạch

```bash
./scripts/run_it04_pipeline.sh prepare
cat "$IT04_RUN/01_data/data_audit.json"
```

Output:

- `essays_split.csv`: văn bản, mã đề, prompt/rubric, split;
- `gold_split.csv`: điểm tổng, nguồn nhãn, split;
- `clean_dataset.jsonl`: bản sạch trung gian;
- `rejected_or_quarantined.csv`: bản loại và lý do;
- `data_audit.json`: số lượng theo split/đề, thống kê điểm và lỗi schema.

Điều kiện PASS trước khi train: có bản train đủ điều kiện; mã đề chỉ thuộc
`De_1`…`De_5`; điểm thuộc [0, 10]; không có ID trùng; split đã khóa không đổi.

### Bước 3 – Train baseline

```bash
./scripts/run_it04_pipeline.sh train
cat "$IT04_RUN/03_model/metrics.json"
```

Output: `total_baseline.pkl`, `predictions.csv`, `metrics.json` và
`rubric_calibration.json`. Báo cáo gồm
MAE, RMSE, QWK, Pearson, bias, exact agreement và adjacent agreement ±1 điểm.
Chọn siêu tham số bằng validation; kết quả chính lấy trên test đã khóa.

### Bước 4 – Chấm tự động theo rubric

```bash
./scripts/run_it04_pipeline.sh grade
```

Output:

- `scores.csv`: điểm mô hình, điểm rubric, điểm cuối, chế độ chấm, cờ phúc tra;
- `item_scores.csv`: điểm tiêu chí, bằng chứng, loại match, vị trí, confidence;
- `human_review_queue.csv`: danh sách cần giảng viên duyệt;
- `grading_audit.json`: tỷ lệ phúc tra và thống kê chế độ chấm.

Chấm một CSV mới có `essay_id,text,exam_id,prompt_text`:

```bash
viegrader it04-grade \
  -i /data/bai_moi.csv \
  -m "$IT04_RUN/03_model/total_baseline.pkl" \
  --rubric-dir "$IT04_RUBRICS" \
  -o "$IT04_RUN/new_batch"
```

### Bước 5 – Đánh giá

```bash
./scripts/run_it04_pipeline.sh evaluate
cat "$IT04_RUN/05_evaluation/evaluation.json"
```

Output: `scores_with_gold.csv`, `evaluation.json`. Metric được tách theo split;
`test_is_primary=true` xác nhận có test set để báo cáo.

### Bước 6 – Tổng hợp báo cáo

```bash
./scripts/run_it04_pipeline.sh report
```

Output: `workflow_summary.json`, `workflow_report.md`, `workflow_report.html`.

## 8. QLoRA tùy chọn

QLoRA vẫn chỉ học điểm tổng, không học điểm câu giả lập:

```bash
./scripts/run_it04_pipeline.sh build-qlora
wc -l "$IT04_RUN/03_model/qlora_train.jsonl"
./scripts/run_it04_pipeline.sh train-qlora
```

Preset: 4-bit NF4, batch 1, gradient accumulation 16. Nếu thiếu VRAM, đặt
`IT04_MAX_LENGTH=1536` hoặc `1024`. Model base phải tải được từ server.

## 9. Cây output

```text
runs/it04/
├── 01_data/       dữ liệu sạch, nhãn tổng, kiểm toán dữ liệu
├── 02_rubric/     kiểm toán rubric và bảng 50 tiêu chí
├── 03_model/      model, dự đoán split, metric train
├── 04_grading/    điểm tổng, điểm câu, hàng đợi phúc tra
├── 05_evaluation/ metric theo split và bảng ghép nhãn vàng
└── 06_reports/    báo cáo tổng hợp JSON, Markdown, HTML
```

## 10. Tiêu chí nghiệm thu

- `pytest -q` không lỗi.
- `rubric_audit.json` có `valid: true`, 5 rubric, 50 tiêu chí.
- Model chỉ fit các bản `train` đủ điều kiện; validation/test không vào fit.
- `scores.csv` không có điểm ngoài [0, 10].
- Mỗi dòng điểm câu có `label_status=rule_based_estimate`.
- Mọi bài có cờ đều nằm trong `human_review_queue.csv`.
- Điểm chỉ được công bố sau khi người có thẩm quyền duyệt phúc tra.

## 11. Lệnh CLI mới

```text
it04-prepare
it04-validate-rubric
it04-train
it04-grade
it04-evaluate
it04-report
it04-run-all
```

Xem tham số đầy đủ bằng `viegrader <lệnh> --help`.

## 12. Giới hạn phải ghi trong báo cáo nghiên cứu

- Nhãn hiện tại là holistic total; chưa có nhãn vàng từng câu từ hai giám khảo.
- Rule exact-answer không chứng minh tính đúng của toàn bộ lập luận toán học.
- Trích xuất DOCX/OCR và cách ghi số câu ảnh hưởng section coverage.
- Bài bị flag phải phúc tra, không được tự động đẩy thẳng lên Moodle.
- Muốn đánh giá độ tin cậy từng tiêu chí cần chấm kép điểm câu, đo QWK/ICC trước
  hòa giải, rồi mới train/evaluate mô hình đa tiêu chí.

Hướng dẫn thao tác cô đọng nằm tại
`docs/HUONG_DAN_THUC_THI_IT04_UBUNTU.md`.

Báo cáo chạy thử đã xác minh mã nguồn nằm tại
`reports/verification_v0_7/VERIFICATION_REPORT.md`.

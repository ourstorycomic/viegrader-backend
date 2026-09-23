# Thực thi VieGrader hoàn toàn bằng CMD trên Ubuntu

Không cần mở GUI. Mọi chức năng được gọi bằng lệnh `viegrader` hoặc script
`scripts/viegrader_cmd.sh`.

## 1. Chuẩn bị

```bash
cd /opt/viegrader/app
source .venv/bin/activate
cp config.cmd.example.env config.cmd.env
chmod 600 config.cmd.env
chmod +x scripts/viegrader_cmd.sh
nano config.cmd.env
```

Khai báo khóa ẩn danh ngoài file cấu hình:

```bash
export VIEGRADER_HMAC_SECRET='thay-bang-khoa-dai-ngau-nhien'
```

## 2. Chạy tuần tự

```bash
./scripts/viegrader_cmd.sh check
./scripts/viegrader_cmd.sh clean
./scripts/viegrader_cmd.sh label
```

Hai giảng viên điền các phiếu trong `forms/`, sau đó gộp thành đường dẫn
`ANNOTATIONS` đã khai báo và chạy:

```bash
./scripts/viegrader_cmd.sh gold
./scripts/viegrader_cmd.sh baseline-tfidf
./scripts/viegrader_cmd.sh baseline-phobert
```

Chỉ huấn luyện QLoRA sau khi kiểm tra GPU đạt và QWK người–người đạt ngưỡng:

```bash
./scripts/viegrader_cmd.sh qlora
```

Đẩy adapter lên Hugging Face:

```bash
export HF_TOKEN='hf_...'
./scripts/viegrader_cmd.sh hf-push
```

## 3. RAG, chấm và đánh giá

```bash
./scripts/viegrader_cmd.sh rag
./scripts/viegrader_cmd.sh score-tfidf
./scripts/viegrader_cmd.sh evaluate-tfidf
```

## 4. Ablation A–H

Sinh ma trận cấu hình:

```bash
./scripts/viegrader_cmd.sh ablation-plan
```

Chạy từng cấu hình trên cùng test split và lưu kết quả lần lượt thành
`reports/ablation/pred_A.csv` … `pred_H.csv`. Mỗi file cần có `essay_id,total`.
Sau đó tổng hợp QWK, MAE, RMSE và các metric khác:

```bash
./scripts/viegrader_cmd.sh ablation-evaluate
```

## 5. BERTScore, SUS và báo cáo

```bash
./scripts/viegrader_cmd.sh bertscore
./scripts/viegrader_cmd.sh sus
./scripts/viegrader_cmd.sh report
```

BERTScore cần tệp phản hồi chuẩn có `essay_id,feedback`. SUS cần các cột
`sus_1` đến `sus_10`.

## 6. Moodle có phê duyệt

```bash
export MOODLE_BASE_URL='https://moodle.example.edu'
export MOODLE_TOKEN='...'
./scripts/viegrader_cmd.sh moodle-assignments 12

viegrader moodle submissions --assignment-id 37 -o reports/submissions.json
viegrader moodle push-grade --assignment-id 37 --user-id 105 \
  --grade 8.5 --feedback 'Đạt yêu cầu rubric' --approved \
  -o reports/moodle_push.json
```

Không có `--approved`, hệ thống từ chối gửi điểm.

## 7. Chạy API không GUI

```bash
./scripts/viegrader_cmd.sh api models/tfidf.pkl
```

API nghe tại `http://server:8000`. Có thể quản lý bằng systemd hoặc đặt sau
Nginx. Xem trợ giúp chính xác của từng lệnh bằng `viegrader <lệnh> --help`.

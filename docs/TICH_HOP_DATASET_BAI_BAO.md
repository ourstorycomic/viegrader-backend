# TÍCH HỢP DATASET VÀ HUẤN LUYỆN CHO BÀI BÁO

## 1. Nguồn dữ liệu được lựa chọn

Nguồn ngoài được chọn là **Vietnamese IT Essays for AES Research** trên Kaggle:

- slug: `vokhoa/vietnamese-it-essays-for-aes-research`;
- vai trò: corpus tiếng Việt đúng miền bài luận CNTT, dùng để xây dựng tập bài
  và gán nhãn theo rubric của nghiên cứu;
- không coi điểm có sẵn là nhãn vàng cho đến khi xác minh được người chấm,
  rubric, tính độc lập và quy trình phân xử;
- để trả lời câu hỏi về human–AI agreement, test set phải có điểm độc lập của
  hai giám khảo. Nếu nguồn Kaggle không có dữ liệu này, phải tổ chức chấm lại.

ELLIPSE và DREsS không được chọn làm dữ liệu chính vì là bài tiếng Anh. Chúng chỉ
phù hợp cho kiểm tra kỹ thuật hoặc thí nghiệm chuyển giao, không thay thế được
test set tiếng Việt trong bài báo.

## 2. Tải và kiểm toán dữ liệu

```bash
source .venv/bin/activate
python -m pip install kaggle
mkdir -p ~/.kaggle
# Chép kaggle.json của tài khoản vào ~/.kaggle rồi:
chmod 600 ~/.kaggle/kaggle.json

chmod +x scripts/download_selected_dataset.sh scripts/viegrader_cmd.sh
./scripts/viegrader_cmd.sh dataset-download
```

Kết quả quan trọng:

- `data/imported/vietnamese_it_aes/essays_raw.csv`;
- `data/imported/vietnamese_it_aes/dataset_audit.json`;
- `labels_unverified.csv` chỉ xuất hiện nếu bộ nhập nhận diện đủ sáu cột điểm.

Đọc `dataset_audit.json`. Không tiếp tục nếu chưa xác minh quyền sử dụng, nguồn
bài làm, cách ẩn danh và nguồn gốc điểm.

## 3. Làm sạch và ẩn danh

Sửa `RAW_INPUT` trong `config.cmd.env` thành đường dẫn tới `essays_raw.csv`, sau
đó đặt khóa HMAC chỉ trong phiên shell:

```bash
export VIEGRADER_HMAC_SECRET='mot-khoa-ngau-nhien-khong-dua-vao-bai-bao'
./scripts/viegrader_cmd.sh check
./scripts/viegrader_cmd.sh clean
```

Rubric nghiên cứu mặc định là `rubrics/bai_luan_6_tieu_chi.yaml`, gồm: nội dung,
lập luận, tổ chức, từ vựng, ngữ pháp và quy ước trình bày; mỗi tiêu chí 0–5,
tổng 0–30.

## 4. Gán nhãn người và tạo gold

```bash
viegrader handbook -r rubrics/bai_luan_6_tieu_chi.yaml \
  -o docs/so_tay_6_tieu_chi.md
viegrader sample -i data/processed/clean.csv -n 200 --seed 42 \
  -o data/pilot.csv
viegrader form -r rubrics/bai_luan_6_tieu_chi.yaml -i data/pilot.csv \
  --raters GK1,GK2 --double-rate 1.0 --seed 42 -o forms/pilot
```

Sau vòng hiệu chỉnh rubric, gán nhãn phần còn lại. Test set dùng cho bài báo phải
được hai giám khảo chấm độc lập. Ghép phiếu chấm thành `data/annotations.csv` rồi:

```bash
viegrader agreement -a data/annotations.csv \
  -r rubrics/bai_luan_6_tieu_chi.yaml -o reports/agreement
viegrader adjudicate -a data/annotations.csv \
  -r rubrics/bai_luan_6_tieu_chi.yaml -o data/gold.csv
```

Không train nếu QWK tổng giữa hai giám khảo dưới 0,70 hoặc nếu còn bài
`needs_adjudication=true` chưa được xử lý.

## 5. Chia dữ liệu không rò rỉ

```bash
./scripts/viegrader_cmd.sh split
```

Mặc định chia 70% train, 10% validation và 20% test theo `student_hash`. Báo cáo
`data/splits/split_report.json` phải có `group_leakage_count = 0`.

Để kiểm tra ngoài đề, chọn một `prompt_id` chưa xuất hiện khi huấn luyện:

```bash
viegrader split-dataset -i data/processed/clean.csv -g data/gold.csv \
  -o data/splits_prompt_holdout --group-column student_hash \
  --validation-size 0.10 --test-size 0.20 --holdout-prompt P05 --seed 42
```

## 6. Huấn luyện QLoRA trên RTX 5060 Ti 16 GB

```bash
./scripts/viegrader_cmd.sh qlora
```

Lệnh đã khóa đầu vào ở `essays_train.csv` và `gold_train.csv`; test set không đi
vào QLoRA. Cấu hình mặc định: Vistral-7B-Chat, NF4 4-bit, LoRA rank 16,
`max_length=1024`, batch size 1, gradient accumulation 16, ba epoch và seed 42.

Để báo cáo mean ± SD, chạy độc lập ba seed, dùng thư mục adapter khác nhau:

```bash
for seed in 42 123 2026; do
  viegrader train-qlora \
    -i data/splits/essays_train.csv -g data/splits/gold_train.csv \
    -r rubrics/bai_luan_6_tieu_chi.yaml \
    -o "artifacts/vistral_qlora_seed_${seed}" \
    --model Viet-Mistral/Vistral-7B-Chat --epochs 3 --max-length 1024 \
    --batch-size 1 --grad-accum 16 --seed "$seed" --allow-unsplit
done
```

## 7. Chấm test set và đánh giá

```bash
viegrader score-qlora -i data/splits/essays_test.csv \
  -r rubrics/bai_luan_6_tieu_chi.yaml \
  --adapter artifacts/vistral_qlora_seed_42 \
  --model Viet-Mistral/Vistral-7B-Chat \
  -o reports/pred_qlora_seed42.csv --runs 3 --temperature 0.1

viegrader evaluate -p reports/pred_qlora_seed42_run01.csv \
  -g data/splits/gold_test.csv -r rubrics/bai_luan_6_tieu_chi.yaml \
  --qwk-hh 0.78 -o reports/evaluate_qlora_seed42_run01.txt
```

Thay `0.78` bằng QWK human–human thực tế từ báo cáo agreement. Ba lượt suy luận
phục vụ phân tích test–retest; ba seed huấn luyện phục vụ mean ± SD. Không gộp
hai loại biến thiên này thành một đại lượng.

## 8. Tệp cần giữ để viết Methodology và Results

- `dataset_audit.json`, `clean_report.json`, `split_report.json`;
- rubric YAML và sổ tay gán nhãn;
- điểm riêng của GK1/GK2 trước hòa giải và `gold.csv` sau phân xử;
- `training_config.json`, `train_metrics.json`, `manifest.json` của từng seed;
- dự đoán từng lượt, QWK/ICC/MAE theo từng tiêu chí và điểm tổng;
- phiên bản driver, CUDA, PyTorch, Transformers, PEFT, TRL và ngày chạy.

Không đưa bài luận nguyên văn, mã sinh viên, khóa HMAC hoặc token truy cập vào
phụ lục hay kho mã công khai.

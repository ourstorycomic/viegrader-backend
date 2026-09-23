# VieGrader 0.6.0 — Toán Rời Rạc trên Ubuntu Server

## 1. Phạm vi dữ liệu

Bộ đầu vào có 166 bài và năm rubric. Nhãn vàng hiện tại gồm `Diem` (điểm tổng)
và `Nhan_Xet`; chưa có điểm vàng `cau_1a ...` do hai giám khảo chấm độc lập.
Vì vậy phiên bản này huấn luyện **điểm tổng** và không tự suy diễn nhãn từng câu.

Kiểm toán tự động phát hiện 16 bài có nội dung xung đột mạnh với `De_Thi`. Mặc
định chúng được đưa vào vùng cách ly, còn 150 bài để chia dữ liệu. Không tự sửa
đề nếu chưa đối chiếu lại bài gốc.

## 2. Cài vào `/opt/viegrader`

```bash
sudo mkdir -p /opt/viegrader
sudo unzip VieGrader_0.6.0_ToanRoiRac_Ubuntu_RTX5060Ti.zip -d /opt/viegrader
sudo chown -R "$USER":"$USER" /opt/viegrader
cd /opt/viegrader

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
python -m pip install -e '.[qlora,reports,docs]'
```

Nếu môi trường `.venv` hiện tại đã cho kết quả `QLoRA ready: True`, chỉ cần:

```bash
cd /opt/viegrader
source .venv/bin/activate
python -m pip install -e '.[qlora,reports,docs]'
```

## 3. Đặt dữ liệu nguồn

```bash
mkdir -p /opt/viegrader/data/toan_roi_rac/source
```

Chép vào thư mục trên đúng sáu tệp:

- `clean_de_1.csv.zip` ... `clean_de_5.csv.zip`
- `Nhan_Vang_Hoan_Chinh.csv`

Rubric và catalog đã nằm trong `data/toan_roi_rac/` của gói.

## 4. Tạo cấu hình

```bash
cd /opt/viegrader
cp config.dm.example.env config.dm.env
chmod +x scripts/viegrader_dm.sh
export VIEGRADER_DM_HMAC_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

Lưu khóa HMAC ở nơi bảo mật nếu cần tái tạo cùng mã ẩn danh. Không ghi khóa vào
bài báo, Git hoặc tệp dữ liệu chia sẻ.

## 5. Kiểm tra máy và rubric

```bash
./scripts/viegrader_dm.sh check
```

## 6. Chuẩn hóa, cách ly xung đột và ẩn danh

```bash
./scripts/viegrader_dm.sh prepare
```

Xem trước khi train:

```bash
cat data/toan_roi_rac/prepared/audit.json
column -s, -t < data/toan_roi_rac/prepared/exam_assignment_audit.csv | less -S
```

Không có tên sinh viên, `source_path` hoặc `essay_id` gốc trong `essays.csv`.

## 7. Chia dữ liệu và chạy baseline

```bash
./scripts/viegrader_dm.sh split
cat data/toan_roi_rac/splits/split_audit.json

./scripts/viegrader_dm.sh baseline
cat artifacts/toan_roi_rac/baseline/metrics.json
```

Split mặc định là 70/15/15, theo `student_hash`, seed 42. Với bộ hiện tại:
train 102, validation 24, test 24; không có sinh viên trùng giữa các tập.

## 8. Tạo dữ liệu và huấn luyện QLoRA

```bash
./scripts/viegrader_dm.sh build-qlora
wc -l data/toan_roi_rac/splits/qlora_train.jsonl

./scripts/viegrader_dm.sh qlora
```

Cấu hình mặc định cho RTX 5060 Ti 16 GB: 4-bit NF4, BF16, batch size 1,
gradient accumulation 16, sequence length 2048. Nếu hết VRAM, sửa
`DM_MAX_LENGTH=1536` hoặc `1024` trong `config.dm.env` rồi chạy lại.

## 9. Chấm test ba lượt và đánh giá

```bash
./scripts/viegrader_dm.sh score-qlora
./scripts/viegrader_dm.sh evaluate-qlora
cat reports/toan_roi_rac/qlora/evaluation.json
```

Báo cáo chứa MAE, RMSE, QWK, Pearson, bias và độ lệch chuẩn giữa các lượt chấm.

## 10. Diễn giải cho bài báo

- Chỉ báo cáo metric tổng trên test set đã khóa trước.
- Đề 2–5 có dưới 20 bài sau kiểm toán; chỉ mô tả, không kết luận metric riêng đề.
- Baseline đi kèm chỉ là kiểm tra pipeline. Kết quả thử hiện tại trên test:
  MAE khoảng 1.61 và QWK khoảng 0.25, chưa đạt mức dùng để chấm thật.
- Chưa được gọi đây là human benchmark theo từng tiêu chí. Muốn trả lời RQ về
  rubric, cần hai giám khảo chấm độc lập `cau_*`, tính agreement trước hòa giải,
  sau đó mới fine-tune/evaluate per-criterion.


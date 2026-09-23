# Hướng dẫn thực thi VieGrader IT04 trên Ubuntu

## Chuẩn bị

```bash
cd /opt/viegrader
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e '.[reports,docs]'
cp config.it04.example.env config.it04.env
nano config.it04.env
chmod +x scripts/run_it04_pipeline.sh
```

## Chạy tuần tự

```bash
./scripts/run_it04_pipeline.sh check
./scripts/run_it04_pipeline.sh prepare
./scripts/run_it04_pipeline.sh train
./scripts/run_it04_pipeline.sh grade
./scripts/run_it04_pipeline.sh evaluate
./scripts/run_it04_pipeline.sh report
```

Hoặc chạy một lệnh:

```bash
./scripts/run_it04_pipeline.sh all
```

## Kiểm tra kết quả

```bash
python -m json.tool "$IT04_RUN/01_data/data_audit.json"
python -m json.tool "$IT04_RUN/02_rubric/rubric_audit.json"
python -m json.tool "$IT04_RUN/03_model/metrics.json"
python -m json.tool "$IT04_RUN/04_grading/grading_audit.json"
python -m json.tool "$IT04_RUN/05_evaluation/evaluation.json"
```

Mở `06_reports/workflow_report.html` để xem toàn bộ kết quả. Kiểm tra
`04_grading/human_review_queue.csv` và phúc tra trước khi phát hành điểm.

## Chạy QLoRA (tùy chọn)

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio
python -m pip install -e '.[qlora]'
./scripts/run_it04_pipeline.sh build-qlora
./scripts/run_it04_pipeline.sh train-qlora
```

Nếu hết VRAM, giảm `IT04_MAX_LENGTH` trong `config.it04.env`. README ở thư mục
dự án mô tả đầy đủ schema, logic chấm, output, metric và giới hạn nghiên cứu.

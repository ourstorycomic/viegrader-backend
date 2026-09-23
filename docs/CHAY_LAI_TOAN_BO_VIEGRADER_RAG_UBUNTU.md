# Chạy lại các nhánh VieGrader và RAG trên server Ubuntu

Hướng dẫn này áp dụng cho bộ IT04 **623 bài** (472 train, 78 validation, 73
test), adapter QLoRA v2 đã huấn luyện và một GPU RTX 5060 Ti. Năm điều kiện
được chạy trên cùng tập test đủ chỗ cho RAG là: TF-IDF/Ridge, hybrid dựa trên
rubric, Qwen2.5-7B-Instruct zero-shot, QLoRA không RAG và QLoRA có RAG.
Thực nghiệm chỉ tạo **điểm đề xuất**, không tự ghi lên Moodle.

## 1. Cập nhật gói mã và vào môi trường Python

Chép gói ZIP mới nhất vào `~/viegrader-rag`, giải nén đè mã dự án, giữ nguyên
thư mục kết quả ở `/opt/viegrader`:

```bash
cd ~/viegrader-rag
unzip -o VieGrader_IT04_RAG_Ubuntu.zip
cd VieGrader_0.7.0_IT04_Ubuntu
source .venv/bin/activate
python -m pip install -e '.[rag,teacher-portal]'
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Kết quả CUDA phải là `True`. Kiểm tra mã mới có lệnh zero-shot:

```bash
python -m viegrader.rag.experiment --help
```

## 2. Gán đường dẫn thật và khóa đầu vào

```bash
export VG_ESSAYS=/opt/viegrader/runs/it04/01_data/essays_split.csv
export VG_GOLD=/opt/viegrader/runs/it04/01_data/gold_split.csv
export VG_ADAPTER=/opt/viegrader/runs/it04/03_model/qlora_adapter_v2_2048
export VG_MODEL=Qwen/Qwen2.5-7B-Instruct
export VG_RUBRICS=/opt/viegrader/data/toan_roi_rac/rubrics
export VG_INDEX="$(pwd)/data/toan_roi_rac/rag_it04.json"
export VG_RUN="$(pwd)/runs/it04_full_rag_$(date +%Y%m%d_%H%M%S)"
```

Nếu rubric đang dùng nằm ở chỗ khác, thay `VG_RUBRICS` bằng đúng thư mục chứa
năm tệp `rubric_de_1.yaml` ... `rubric_de_5.yaml`. Kiểm tra định danh và chia
tập trước khi tính toán:

```bash
python - <<'PY'
import os, pandas as pd
e = pd.read_csv(os.environ['VG_ESSAYS'])
g = pd.read_csv(os.environ['VG_GOLD'])
for name, d in [('essays', e), ('gold', g)]:
    counts = d.split.value_counts().to_dict()
    print(name, len(d), counts)
    assert len(d) == 623 and counts.get('train') == 472
    assert counts.get('validation') == 78 and counts.get('test') == 73
assert set(e.essay_id.astype(str)) == set(g.essay_id.astype(str))
PY
test -f "$VG_ADAPTER/adapter_config.json"
test -f "$VG_INDEX"
ls "$VG_RUBRICS"/rubric_de_{1,2,3,4,5}.yaml
```

Không dùng cặp `data/toan_roi_rac/splits/*.csv` của gói mẫu: nó chỉ có
150 bản ghi và 24 bài test. Lưu dấu vết đầu vào trong thư mục chạy mới:

```bash
mkdir -p "$VG_RUN"
sha256sum "$VG_ESSAYS" "$VG_GOLD" "$VG_INDEX" \
  "$VG_RUBRICS"/rubric_de_{1,2,3,4,5}.yaml \
  "$VG_ADAPTER"/adapter_config.json > "$VG_RUN/input_hashes.sha256"
```

## 3. Chuẩn bị đúng tập bài chung cho đối chứng

```bash
python -m viegrader.rag.experiment prepare \
  --essays "$VG_ESSAYS" --index "$VG_INDEX" --model "$VG_MODEL" \
  --max-tokens 4608 --top-k 3 --expected-test 73 --output "$VG_RUN"
test -s "$VG_RUN/manifest.json"
python - <<'PY'
import os, pandas as pd
p = os.environ['VG_RUN']
print(pd.read_csv(p + '/coverage.csv').status.value_counts().to_string())
PY
```

Số bài `eligible` có thể thấp hơn 73. Tất cả nhánh dưới đây chấm đúng tập
`input_plain.csv` của bước này. Trong `input_rag.csv`, chỉ phần trích đoạn
giáo trình khác đi. `coverage.csv` cho biết bài bị loại vì độ dài hoặc không
đủ chỗ cho một đoạn RAG; không cắt bỏ đầu prompt một cách im lặng.

## 4. Huấn luyện lại mốc TF-IDF/Ridge trên train và chấm tập chung

Mốc này chạy CPU; chỉ train trên cột `split=train`. Không ghi đè baseline cũ:

```bash
python -m viegrader.cli dm-train-baseline \
  -i "$VG_ESSAYS" -g "$VG_GOLD" -o "$VG_RUN/baseline"
python -m viegrader.cli dm-score-baseline \
  -m "$VG_RUN/baseline/total_baseline.pkl" \
  -i "$VG_RUN/input_plain.csv" -o "$VG_RUN/pred_tfidf_ridge.csv"
```

## 5. Hiệu chỉnh rubric trên validation rồi chạy hybrid

Tệp `rubric_calibration.json` chọn trọng số theo MAE trên **78 bài
validation**, không theo test. Số bài đủ điều kiện hybrid được ghi riêng:

```bash
python - <<'PY'
import os
from pathlib import Path
from viegrader.standardized_pipeline import calibrate_rubric_weight
p = Path(os.environ['VG_RUN'])
report = calibrate_rubric_weight(
    p/'baseline/total_baseline.pkl', os.environ['VG_ESSAYS'], os.environ['VG_GOLD'],
    os.environ['VG_RUBRICS'], p/'rubric_calibration.json')
print('validation đủ điều kiện:', report['n_hybrid_eligible'])
print('trọng số rubric:', report['selected_weight'])
PY
export VG_WEIGHT="$(python - "$VG_RUN/rubric_calibration.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['selected_weight'])
PY
)"
python -m viegrader.cli it04-grade \
  -i "$VG_RUN/input_plain.csv" -m "$VG_RUN/baseline/total_baseline.pkl" \
  --rubric-dir "$VG_RUBRICS" --rubric-weight "$VG_WEIGHT" \
  -o "$VG_RUN/hybrid"
```

Xem `hybrid/grading_audit.json`: số bài theo `hybrid_rubric_model` và
`model_fallback` phải được báo cáo riêng. Điểm từng câu trong
`hybrid/item_scores.csv` là **ước lượng quy tắc**, không phải nhãn giảng viên.

## 6. Chạy zero-shot và hai nhánh QLoRA trên GPU

Trong thời gian suy luận, tạm dừng tác vụ VieGrader khác đang dùng cùng GPU.
Nên chạy trong `tmux`; khai báo lại các biến `VG_*` nếu tạo shell mới.
Chạy lần lượt, không khởi chạy song song các lệnh GPU:

```bash
set -o pipefail
python -m viegrader.rag.experiment run-zero-shot \
  --output "$VG_RUN" --model "$VG_MODEL" \
  --max-tokens 4608 --max-new-tokens 96 2>&1 | tee "$VG_RUN/zero_shot.log"
test -s "$VG_RUN/pred_zero_shot.csv"

python -m viegrader.rag.experiment run \
  --output "$VG_RUN" --model "$VG_MODEL" --adapter "$VG_ADAPTER" \
  --max-tokens 4608 --max-new-tokens 96 2>&1 | tee "$VG_RUN/qlora_rag.log"
test -s "$VG_RUN/predictions.csv"
```

Lệnh `run` nạp cùng adapter một lần và chấm tuần tự QLoRA không RAG rồi
QLoRA + RAG ở nhiệt độ 0. Zero-shot dùng **mô hình nền chưa gắn adapter**,
trên đúng prompt và tập bài chung của lần chạy này. Không so sánh QWK của
thí nghiệm này trực tiếp với QWK cũ ở giao thức khác.

## 7. Ghép gold sau suy luận và xuất báo cáo

```bash
python -m viegrader.rag.experiment report \
  --output "$VG_RUN" --gold "$VG_GOLD" \
  --baseline "$VG_RUN/pred_tfidf_ridge.csv" \
  --hybrid "$VG_RUN/hybrid/scores.csv" \
  --bootstrap 1000 | tee "$VG_RUN/report_printed.json"
```

Trong `report.json`, so sánh QLoRA không RAG và QLoRA + RAG trên cùng tập
`n_common_valid`. Ba nhánh zero-shot, QLoRA không RAG, QLoRA + RAG còn có
chỉ số `three_way_common` trên bài hợp lệ của cả ba. Mốc TF-IDF/Ridge và
hybrid được tính trên `n_common_valid` của cặp QLoRA. QWK dùng 41 mức cố định
từ 0 đến 10 bước 0,25. Xem thêm:

| Tệp | Ý nghĩa |
| --- | --- |
| `manifest.json`, `input_hashes.sha256` | Nguồn dữ liệu, chỉ mục, mô hình và cấu hình |
| `coverage.csv` | Số bài đủ điều kiện; lý do loại bài |
| `retrieval_audit.csv` | Đoạn, tài liệu, trang PDF được đưa vào prompt |
| `hybrid/grading_audit.json` | Số bài dùng hybrid thực sự và fallback |
| `pred_zero_shot.csv`, `predictions.csv` | Kết quả mô hình trước khi ghép gold |
| `report.json`, `paired_errors.csv` | Chỉ số đối chứng và sai số từng bài |

Kiểm tra những bài có `delta_abs_error > 2` hoặc sai số RAG lớn, đối chiếu
bài gốc, đáp án và đúng trang truy xuất. Khoảng tin cậy bootstrap ở đây lấy
theo bài đơn lẻ; 13 bài test đã được phát hiện trùng nội dung train, nên kết
quả chưa chứng minh khả năng khái quát hóa sang khóa học mới.

## 8. Lựa chọn chạy một lệnh có thể tiếp tục sau gián đoạn

Sau khi xuất các biến ở Mục 2 và kích hoạt `.venv`, có thể dùng:

```bash
mkdir -p "$VG_RUN"
tmux new -s vg-all
set -o pipefail
bash scripts/run_it04_all_models_rag.sh 2>&1 | tee "$VG_RUN/all_models.log"
```

Script kiểm tra đúng bộ 623/73, chuẩn bị tập chung, chạy CPU trước GPU rồi
tạo báo cáo. Nếu lệnh đã tạo một tệp đầu ra thành công, chạy lại script sẽ
bỏ qua tệp đó; dùng **VG_RUN mới** khi đổi dữ liệu, PDF, rubric hoặc mô hình.
Không chạy script và các lệnh từng bước đồng thời.

## 9. Huấn luyện lại QLoRA (chỉ khi muốn thử adapter mới)

Các bước trên **chạy lại suy luận** với adapter v2 hiện có; baseline được
huấn luyện lại. Nếu muốn huấn luyện adapter QLoRA mới, tạo thư mục khác,
kiểm toán token hóa rồi chấm lại tất cả nhánh dùng adapter mới:

```bash
python -m viegrader.cli dm-build-qlora -i "$VG_ESSAYS" -g "$VG_GOLD" \
  --split train -o "$VG_RUN/qlora_train.jsonl"
python -m viegrader.cli dm-train-qlora \
  --records "$VG_RUN/qlora_train.jsonl" \
  --model "$VG_MODEL" --max-length 2048 --batch-size 1 \
  --grad-accum 16 --epochs 3 --seed 42 \
  -o "$VG_RUN/adapter_new"
```

`max-length=2048` vẫn có thể rút gọn phần ngữ cảnh; kiểm tra
`tokenization_audit.json` trong thư mục adapter mới trước khi diễn giải kết
quả. Không thay `VG_ADAPTER` bên trong một `VG_RUN` đã có `predictions.csv`.
Muốn so sánh adapter mới, tạo một `VG_RUN` mới, đặt `VG_ADAPTER` đến
`adapter_new` và chạy lại từ Mục 3. Bộ test hiện vẫn có trùng nội dung với
train; muốn kết luận trên tập độc lập phải khử trùng, chia lại và huấn luyện
lại trên phiên bản dữ liệu mới.

## 10. Kiểm tra luồng cổng giảng viên và Moodle

Sau khi xem báo cáo mô hình, thử riêng kiểm thử cổng trong chế độ mô phỏng:

```bash
export VIEGRADER_MOODLE_MODE=simulate
export VIEGRADER_RAG_INDEX="$VG_INDEX"
python -m pytest tests/test_teacher_portal.py -q
```

Kiểm thử này dùng điểm chấm giả, chỉ xác nhận quy trình đề → lô → duyệt →
mô phỏng gửi. Không phải bằng chứng gửi điểm đến Moodle thật. Quy trình
Moodle sandbox và màn hình duyệt toàn văn cần được kiểm thử riêng trước khi
bật live.

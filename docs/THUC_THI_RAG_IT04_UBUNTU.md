# Thực thi RAG trên Ubuntu và xuất báo cáo IT04

Để chạy lại **toàn bộ năm nhánh** (TF-IDF/Ridge, hybrid, zero-shot, QLoRA và
QLoRA + RAG) trong một giao thức, dùng hướng dẫn
`docs/CHAY_LAI_TOAN_BO_VIEGRADER_RAG_UBUNTU.md`.

Quy trình này dùng **adapter QLoRA Qwen2.5-7B-Instruct đã huấn luyện** trên
server RTX 5060 Ti và so sánh hai điều kiện `QLoRA` với `QLoRA + RAG` trên
cùng các bài test đủ điều kiện. Có thể thêm mốc `TF-IDF/Ridge` đã huấn luyện.
Không cần huấn luyện lại adapter cho phép đối chứng đầu tiên. Chưa có điểm RAG
cho đến khi chạy đủ suy luận và lệnh `report` trên server.

## 1. Chép gói và kiểm tra máy

```bash
mkdir -p ~/viegrader-rag
cd ~/viegrader-rag
unzip ~/Downloads/VieGrader_IT04_RAG_Ubuntu.zip
cd VieGrader_0.7.0_IT04_Ubuntu
nvidia-smi
python3 --version
```

Nếu máy chủ nhận gói qua `scp`, thay đường dẫn `~/Downloads/...` bằng nơi
đã chép ZIP. Kiểm tra phiên bản PyTorch CUDA đang dùng trong môi trường
VieGrader hiện có:

```bash
python3 - <<'PY'
import torch
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'không có')
assert torch.cuda.is_available(), 'Cần môi trường PyTorch có CUDA trước khi chạy QLoRA'
PY
```

Nếu PyTorch chưa dùng được GPU, chọn lệnh cài tương ứng với hệ thống từ
https://pytorch.org/get-started/locally/ rồi chạy lại kiểm tra. Để tránh thay
PyTorch CUDA đã dùng ổn định, cài các phần phụ thuộc vào chính môi trường đó:

```bash
python3 -m pip install -e '.[rag,teacher-portal]'
python3 -m pip install 'transformers>=4.48,<5' 'peft>=0.14' \
  'accelerate>=1.2' 'bitsandbytes>=0.48' 'safetensors>=0.5'
python3 - <<'PY'
import fitz, torch, transformers, peft, bitsandbytes
print('Môi trường sẵn sàng:', torch.cuda.get_device_name(0))
PY
```

## 2. Xác định dữ liệu và adapter đã có

Tệp `essays_split.csv` cần các cột `essay_id,text,exam_id,prompt_text,answer_key,split`.
Tệp `gold_split.csv` cần `essay_id,gold_total,split`. Chỉ chuyển tệp gold cho
bước báo cáo **sau khi** suy luận. Adapter phải cùng mô hình nền đã huấn luyện.

**Chú ý:** Gói ví dụ có `data/toan_roi_rac/splits/essays_split.csv` và
`gold_split.csv` gồm **150 bản ghi, chỉ 24 bài test**. Chúng không phải bộ IT04
đầy đủ 623 bản ghi, 73 bài test dùng trong báo cáo. Hãy tìm bộ đầy đủ đã chạy
trên server:

```bash
find /home/giangvien /opt -type f \( -name essays_split.csv -o -name gold_split.csv -o -name adapter_config.json \) -print 2>/dev/null
```

Chọn cặp tệp cùng một phiên bản dữ liệu. Gán đường dẫn **thực tế tìm được**;
những đường dẫn dưới đây chỉ là ví dụ, không chạy nguyên văn.

```bash
export VG_ESSAYS=/duong/dan/thuc/essays_split.csv
export VG_GOLD=/duong/dan/thuc/gold_split.csv
export VG_ADAPTER=/duong/dan/thuc/qlora_adapter_v2_2048
export VG_MODEL=Qwen/Qwen2.5-7B-Instruct
export VG_RUN="$(pwd)/runs/rag_it04_$(date +%Y%m%d_%H%M%S)"
test -f "$VG_ESSAYS" && test -f "$VG_GOLD" && test -f "$VG_ADAPTER/adapter_config.json"
python3 - <<'PY'
import os, pandas as pd
essays = pd.read_csv(os.environ['VG_ESSAYS'])
gold = pd.read_csv(os.environ['VG_GOLD'])
print('Essays:', len(essays), essays.split.value_counts().to_dict())
print('Gold:', len(gold), gold.split.value_counts().to_dict())
assert len(essays) == len(gold) == 623
assert (essays.split == 'test').sum() == (gold.split == 'test').sum() == 73
PY
```

Điều chỉnh ba đường dẫn theo vị trí thực tế. Adapter chỉ có `qlora_adapter`
thay vì `qlora_adapter_v2_2048` cần xác nhận phiên bản trước khi dùng. Nếu
đã có mô hình nền trong cache Hugging Face, có thể thay `VG_MODEL` bằng đường
dẫn cục bộ của **đúng** Qwen2.5-7B-Instruct để chạy không cần mạng.
Nếu chưa tìm thấy bộ 623 bài, dừng bước chuẩn bị và chép bộ dữ liệu thực nghiệm
lên server. Chỉ dùng `--expected-test 24` để chạy thử bộ mẫu, không ghi các
chỉ số đó thành kết quả thực nghiệm 73 bài.

## 3. Kiểm tra giáo trình và tạo chỉ mục

Tài liệu trong `data/toan_roi_rac/references/` là hai PDF người dùng cung cấp.
Rà các trang chứa công thức và bài giải trước khi dùng làm nguồn chấm.

```bash
sha256sum data/toan_roi_rac/references/*.pdf
python3 -m viegrader.rag.pdf_index \
  data/toan_roi_rac/references/BaiGiang_ToanRR.pdf \
  data/toan_roi_rac/references/TRR_NguyenDucNghia.pdf \
  --course-id IT04 --output data/toan_roi_rac/rag_it04.json
```

Lệnh này lập chỉ mục TF-IDF cục bộ, ghi tên tệp, số trang và SHA-256 PDF cho
từng đoạn. Tệp PDF ảnh quét hoặc trang trích chữ sai ký hiệu cần OCR và rà soát
thủ công rồi lập chỉ mục lại. Không dùng điểm sinh viên làm tài liệu truy xuất.

```bash
python3 -m viegrader.rag.audit --index data/toan_roi_rac/rag_it04.json \
  --pdf data/toan_roi_rac/references/BaiGiang_ToanRR.pdf \
        data/toan_roi_rac/references/TRR_NguyenDucNghia.pdf \
  --output reports/rag_it04/index_audit.json
```

Bản kiểm toán đã đóng gói báo cáo **770 đoạn**, gồm 238 đoạn từ bài giảng
(137/137 trang có chữ đủ ngưỡng) và 532 đoạn từ giáo trình Nguyễn Đức Nghĩa
(290/298 trang có chữ đủ ngưỡng). Tám trang còn lại cần xem trực tiếp nếu
chứa kiến thức phải truy xuất. Đây là kết quả lập chỉ mục, chưa phải kết quả
chấm bài. Tệp `reports/rag_it04/index_audit.json` lưu ba truy vấn kiểm tra,
các trang truy xuất và mã băm nguồn.

## 4. Chuẩn bị đối chứng và kiểm tra độ bao phủ

```bash
python3 -m viegrader.rag.experiment prepare \
  --essays "$VG_ESSAYS" --index data/toan_roi_rac/rag_it04.json \
  --model "$VG_MODEL" --max-tokens 4608 --top-k 3 --output "$VG_RUN"
python3 - <<'PY'
import os, pandas as pd
p = os.environ['VG_RUN']
print(pd.read_csv(p + '/coverage.csv').status.value_counts().to_string())
print(pd.read_csv(p + '/retrieval_audit.csv').head(10).to_string(index=False))
PY
```

`coverage.csv` ghi lý do loại bài nếu prompt gốc quá dài hoặc không vừa một
đoạn RAG. `input_plain.csv` và `input_rag.csv` cùng bài, đề, đáp án và adapter;
chỉ khác phần ngữ cảnh truy xuất. `retrieval_audit.csv` ghi đoạn tài liệu được
chọn. Không được xem trước gold để chọn đoạn hay điều chỉnh cấu hình.

## 5. Chạy QLoRA trên một GPU

Tắt lô chấm web hoặc dịch vụ đang giữ GPU trước khi chạy. Trong phiên SSH,
dùng `tmux` để lệnh tiếp tục chạy khi mất kết nối:

```bash
tmux new -s viegrader-rag
cd ~/viegrader-rag/VieGrader_0.7.0_IT04_Ubuntu
set -o pipefail
python3 -m viegrader.rag.experiment run \
  --output "$VG_RUN" --model "$VG_MODEL" --adapter "$VG_ADAPTER" \
  --max-tokens 4608 --max-new-tokens 96 2>&1 | tee "$VG_RUN/inference.log"
```

Nếu mở `tmux` bằng một shell mới, khai báo lại `VG_RUN`, `VG_MODEL` và
`VG_ADAPTER` trong shell đó. Có thể tách khỏi tmux bằng `Ctrl+B`, rồi `D`;
quay lại bằng `tmux attach -t viegrader-rag`. Lệnh này tải mô hình một lần,
chấm lần lượt hai nhánh nhiệt độ 0 và tạo `predictions.csv`. Xác nhận kết quả:

```bash
test -s "$VG_RUN/predictions.csv"
python3 - <<'PY'
import os, pandas as pd
p = pd.read_csv(os.environ['VG_RUN'] + '/predictions.csv')
print(p.groupby('variant').agg(n=('essay_id','size'), valid=('parse_ok','sum')))
PY
```

Nếu hết VRAM, bảo đảm chỉ có một tiến trình mô hình dùng GPU và kiểm tra
`nvidia-smi`. Không tự giảm `max-tokens` sau khi đã `prepare`; hãy chạy lại
`prepare` với ngưỡng mới và ghi nhận rằng độ bao phủ đã thay đổi.

## 6. Mốc TF-IDF/Ridge và báo cáo

Nếu có mô hình baseline đúng phiên bản, chấm trên cùng tệp bài test:

```bash
export VG_BASELINE=/opt/viegrader/runs/it04/03_model/total_baseline.pkl
python3 -m viegrader.cli dm-score-baseline \
  -m "$VG_BASELINE" -i "$VG_RUN/input_plain.csv" \
  -o "$VG_RUN/pred_tfidf_ridge.csv"
```

Nếu không có baseline tương thích, bỏ hẳn tham số `--baseline` ở lệnh tiếp
theo. Ví dụ khi đã chấm baseline:

```bash
python3 -m viegrader.rag.experiment report \
  --output "$VG_RUN" --gold "$VG_GOLD" \
  --baseline "$VG_RUN/pred_tfidf_ridge.csv" --bootstrap 1000 \
  | tee "$VG_RUN/report_printed.json"
```

Kết quả ở `report.json` gồm: n test, n đủ điều kiện ghép cặp, tỷ lệ JSON hợp
lệ của hai nhánh, MAE, RMSE, bias, tỷ lệ đúng tuyệt đối, tỷ lệ sai số ≤1,
QWK theo 41 mức 0–10 bước 0,25, chênh lệch MAE và khoảng bootstrap 95%.
`paired_errors.csv` chứa sai số của từng bài; `paired_plain_with_gold.csv` và
`paired_rag_with_gold.csv` phục vụ đối chiếu; `retrieval_audit.csv` phục vụ
kiểm toán nguồn. Các tệp có bài làm hoặc nhãn cần lưu theo quyền hạn của đơn vị.

Khoảng bootstrap hiện lấy bài làm làm đơn vị; vì có 13/73 bài test trùng nội
dung train, không dùng khoảng này để khẳng định khả năng khái quát hóa. Chỉ số
theo từng mã đề có nhóm rất ít bài; dùng để phát hiện vấn đề, không xếp hạng
độ khó. Có thể soát riêng tám bài từng sai trên hai điểm trong kết quả cũ.

## 7. Bật RAG cho cổng giảng viên sau khi kiểm tra báo cáo

```bash
export VIEGRADER_RAG_INDEX="$(pwd)/data/toan_roi_rac/rag_it04.json"
export VIEGRADER_PORTAL_MAX_INPUT_TOKENS=3584
uvicorn viegrader.teacher_portal:app --host 127.0.0.1 --port 8088
```

Lô QLoRA của cổng sẽ ghi `rag_audit.json` trong thư mục lô. Ngưỡng token
3584 của cổng khác mức 4608 của thí nghiệm; theo dõi tỷ lệ bài bị từ chối
riêng. Nhánh hybrid không dùng RAG. Moodle vẫn ở chế độ mô phỏng mặc định.

## 8. Cách ghi vào báo cáo nghiên cứu

Chỉ điền chỉ số từ `report.json` sau khi kiểm tra tệp dự đoán, `coverage.csv`
và các bài sai lớn. Bảng đề xuất:

| Cấu hình | n đầu ra hợp lệ trên tập ghép cặp | MAE | RMSE | QWK 41 mức | Bias | |e| ≤ 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF/Ridge (nếu có) | đọc từ báo cáo | đọc từ báo cáo | đọc từ báo cáo | đọc từ báo cáo | đọc từ báo cáo | đọc từ báo cáo |
| QLoRA không RAG | đọc từ báo cáo | đọc từ báo cáo | đọc từ báo cáo | đọc từ báo cáo | đọc từ báo cáo | đọc từ báo cáo |
| QLoRA + RAG | đọc từ báo cáo | đọc từ báo cáo | đọc từ báo cáo | đọc từ báo cáo | đọc từ báo cáo | đọc từ báo cáo |

Diễn giải chênh lệch trên tập bài hợp lệ chung; nêu song song tỷ lệ bị loại
và đầu ra sai JSON. Kết luận về giá trị của RAG cần kèm kiểm toán nguồn truy
xuất và kiểm tra bài sai lớn; chưa suy ra hiệu quả ở khóa học mới.

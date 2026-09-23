# Hướng dẫn VieGrader GUI và Hugging Face

## 1. Chạy GUI trên máy cá nhân

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[gui,docs]"
viegrader gui --host 127.0.0.1 --port 7860
```

Mở `http://127.0.0.1:7860`. Nếu dùng Windows PowerShell, kích hoạt môi trường
bằng `.venv\Scripts\Activate.ps1`.

## 2. Chạy GUI trên Kaggle

```python
!pip install -e "/kaggle/working/viegrader[gui,qlora,evaluation,docs]"
```

Trong notebook:

```python
from viegrader.gui.app import launch
launch(server_name="0.0.0.0", share=True)
```

Chọn GPU trong Notebook settings trước khi train QLoRA. Lưu `HF_TOKEN` bằng
Kaggle Secrets, không viết token trong cell.

## 3. Tạo Hugging Face Space

1. Tạo Space mới, chọn SDK **Gradio**.
2. Tải toàn bộ thư mục dự án lên Space. `app.py`, `README.md` và
   `requirements.txt` phải nằm ở thư mục gốc.
3. Trong Settings → Secrets, thêm `HF_TOKEN`.
4. Nếu dùng Moodle, thêm `MOODLE_BASE_URL` và `MOODLE_TOKEN`.
5. Để train Vistral‑7B, chọn GPU hardware. CPU Space chỉ phù hợp với GUI,
   TF‑IDF, RAG và SUS.

`app.py` mở GUI tại cổng chuẩn 7860. Hàng đợi được giới hạn một tác vụ huấn luyện
đồng thời để tránh hai tiến trình chiếm cùng GPU.

## 4. Token Hugging Face

- Token đọc: đủ để tải model gated đã được chấp thuận.
- Token ghi hoặc fine-grained: cần để tạo/cập nhật model repository.
- Lưu dưới tên `HF_TOKEN` trong Secrets hoặc biến môi trường.
- Không đưa token vào CSV, notebook công khai, mã nguồn hoặc ô nhập GUI.

Để dùng `Viet-Mistral/Vistral-7B-Chat`, tài khoản phải chấp thuận điều kiện truy
cập trên trang model trước khi chạy.

## 5. Huấn luyện trên giao diện

Tab **QLoRA & Hugging Face** nhận:

- dữ liệu sạch có `essay_id`, `text`, `prompt_text`, `prompt_id`;
- nhãn vàng có `essay_id`, điểm tổng và điểm từng tiêu chí;
- rubric YAML;
- base model trên Hub;
- `repo_id` dạng `username/model-name`;
- epoch, độ dài chuỗi, batch và gradient accumulation.

Quy trình:

1. xác thực `HF_TOKEN`;
2. ghép dữ liệu và nhãn theo `essay_id`;
3. sinh hội thoại instruction/output JSON;
4. tải Vistral‑7B ở chế độ NF4 4-bit;
5. huấn luyện adapter QLoRA;
6. lưu tokenizer, adapter, manifest, training config và model card;
7. tạo repository riêng tư/công khai;
8. đẩy thư mục adapter lên Hub;
9. trả adapter ZIP và commit URL cho người dùng.

Huấn luyện diễn ra trên GPU của Kaggle, máy chủ hoặc Hugging Face Space đang chạy;
Hub là nơi cung cấp base model và lưu phiên bản adapter.

## 6. Chuẩn bị dữ liệu

Không train trực tiếp trên toàn bộ dữ liệu. Phải giữ riêng test set chưa nhìn
thấy, chia theo sinh viên hoặc đề bài và chỉ dùng train split trong tab QLoRA.
Không đưa họ tên, email, mã sinh viên gốc hoặc dữ liệu nhạy cảm lên Hub. Dataset
và model repository nên để private trong giai đoạn nghiên cứu.

## 7. Các tab khác

- **Baseline**: TF-IDF hoặc PhoBERT, xuất `.pkl` và báo cáo.
- **RAG**: lập chỉ mục tài liệu đã phê duyệt.
- **Chấm điểm**: chấm lô và xuất CSV.
- **Đánh giá**: QWK, MAE, SMD và đối sánh người–người.
- **Ablation**: xuất ma trận A–H.
- **SUS**: điểm SUS và Cronbach's alpha.
- **Xuất báo cáo**: tổng hợp kết quả 10 giai đoạn thành Excel, Word, PDF, HTML và JSON.
- **Moodle**: chỉ gửi điểm khi giảng viên đánh dấu phê duyệt.

## 8. Khuyến nghị tài nguyên

Thử nghiệm đầu tiên nên dùng `max_length=1024`, `batch_size=1`,
`gradient_accumulation=8`, 1 epoch để kiểm tra pipeline. Sau khi không còn lỗi
mới tăng lên 3 epoch và 1536 token. Nếu hết VRAM, giảm độ dài chuỗi trước, sau
đó giảm batch hoặc tăng gradient accumulation.

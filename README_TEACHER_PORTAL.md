# Cổng giảng viên VieGrader trên Ubuntu

Ứng dụng FastAPI cung cấp giao diện tại `/`, API tại `/api/docs`, tạo bộ đề IT04, tải bài làm, gọi pipeline chấm hiện có, xem toàn văn bài cạnh đề–đáp án–rubric, lưu nhận xét và người duyệt, xuất CSV, ghi nhật ký kiểm toán và mô phỏng/gửi điểm đến Moodle. **Điểm và nhận xét của mô hình là đề xuất; mọi điểm phải được giảng viên duyệt trước khi gửi LMS.** Hướng dẫn công bố qua HTTPS nằm trong `DEPLOY_PORTAL_UBUNTU.md`.

## 1. Cài đặt

Giải nén gói dự án và chép các tệp mới vào cây mã nguồn VieGrader tại `/opt/viegrader` nếu trên máy đã có dữ liệu và adapter đã huấn luyện. Không ghi đè thư mục `runs/it04` hiện hữu. Ví dụ, để cài bản mã nguồn mới tại một thư mục riêng:

```bash
cd /opt
unzip VieGrader_Teacher_Portal_Ubuntu.zip -d viegrader_teacher_release
cd /opt/viegrader_teacher_release/VieGrader_0.7.0_IT04_Ubuntu
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[teacher-portal]'
```

Nếu dùng QLoRA, cài các gói GPU tương thích với môi trường VieGrader Ubuntu hiện tại (bao gồm `torch`, `transformers`, `peft`, `bitsandbytes`); cách cài là `python -m pip install -e '.[teacher-portal,qlora]'` sau khi đã thiết lập PyTorch/CUDA đúng driver RTX 5060 Ti. Chạy ứng dụng từ thư mục gốc của dự án để đường dẫn tương đối của rubric và mô hình được giải đúng. Bản triển khai trên máy đã train nên dùng chính môi trường `.venv` và thư mục `/opt/viegrader` đang hoạt động.

## 2. Khai báo đường dẫn và tài khoản

```bash
cd /opt/viegrader
source .venv/bin/activate
export VIEGRADER_PORTAL_USER='giangvien'
export VIEGRADER_PORTAL_PASSWORD='THAY_BANG_MAT_KHAU_DU_MANH'
export VIEGRADER_PORTAL_DATA='/opt/viegrader/runs/teacher_portal'
export VIEGRADER_RUBRIC_DIR='/opt/viegrader/data/toan_roi_rac/rubrics'
export VIEGRADER_BASELINE_MODEL='/opt/viegrader/runs/it04/03_model/total_baseline.pkl'
export VIEGRADER_QLORA_ADAPTER='/opt/viegrader/runs/it04/03_model/qlora_adapter_v2_2048'
export VIEGRADER_RUBRIC_CALIBRATION='/opt/viegrader/runs/it04/03_model/rubric_calibration.json'
export VIEGRADER_QLORA_MODEL='Qwen/Qwen2.5-7B-Instruct'
export VIEGRADER_MOODLE_MODE='simulate'
viegrader-teacher-portal
```

Kiểm tra đúng tệp mô hình bằng `ls -lh "$VIEGRADER_BASELINE_MODEL"` và `ls -ld "$VIEGRADER_QLORA_ADAPTER"`. Nếu không có baseline, chỉ chọn QLoRA; nếu chưa cài các gói GPU hoặc chưa có adapter, chỉ chọn hybrid. Truy cập qua SSH tunnel `ssh -L 8088:127.0.0.1:8088 giangvien@server`, rồi mở `http://127.0.0.1:8088`. Có thể kiểm tra `curl -u giangvien:'MAT_KHAU' http://127.0.0.1:8088/api/v1/health`.

Mặc định máy chủ chỉ nghe `127.0.0.1:8088`. Khi cung cấp cho nhiều giảng viên, cần đặt sau reverse proxy HTTPS và cơ chế xác thực tổ chức. Basic Auth hiện tại là **một tài khoản quản trị chung**, không phân quyền từng giảng viên; không phơi trực tiếp cổng 8088 ra Internet. Giữ riêng và sao lưu `VIEGRADER_PORTAL_DATA`, nơi lưu bài làm, đề và SQLite; người dùng Unix chạy dịch vụ phải có quyền với thư mục đó. Không gửi dữ liệu sinh viên vào hệ thống ngoài phạm vi được phép.

## 3. Quy trình trên website

1. Chọn mã đề `De_1`–`De_5`, tải đề và đáp án chuẩn TXT, DOCX hoặc PDF văn bản. Ứng dụng kiểm tra rubric YAML đúng mã đề có tồn tại; PDF scan phải OCR trước.
2. Chọn bộ đề đã tải, phương pháp `hybrid` hoặc `qlora`, tải một hoặc nhiều bài làm. Có thể dùng CSV UTF-8 với cột `essay_id,text`, một dòng mỗi bài; tối đa 100 bài/lô và 20 MB/tệp.
3. Bấm **Bắt đầu chấm**, xem trạng thái và cập nhật khi hoàn thành. Điểm `proposed` và cờ `flags` là dữ liệu hỗ trợ xem lại. Job chạy lần lượt trong một worker; với QLoRA, dự toán prompt quá dài sẽ làm lô thất bại có giải thích, không cắt mất system/rubric một cách âm thầm.
4. Kiểm tra bài và nhập điểm duyệt `[0,10]` theo bước `0,25`. Nếu muốn gửi LMS, nhập `Moodle user ID` đúng sinh viên, duyệt rồi nhập `Assignment ID` đúng học phần/bài tập. Thử **Gửi / mô phỏng** trước và tải CSV kết quả để lưu hồ sơ.

Nhánh `hybrid` gọi `viegrader.standardized_pipeline.grade_with_rubric` dùng mô hình TF-IDF/Ridge cùng YAML theo mã đề, dùng trọng số trong `rubric_calibration.json` nếu có, mặc định 0 nếu không. **Văn bản đáp án vừa tải không được nhánh này sử dụng trực tiếp**. Nhánh `qlora` gọi `viegrader.discrete_math.score_total_qlora`, ghép đề + đáp án vừa tải + rubric YAML để suy luận JSON điểm tổng. Cả hai nhánh chỉ có nhãn tổng `total_only`; điểm mục/chất lượng lời nhận xét chưa được giảng viên xác nhận độc lập. Nên kiểm tra đề và đáp án có khớp đúng bộ đề IT04 mà mô hình được huấn luyện trước khi chấm.

## 4. API chính

Các yêu cầu dùng HTTP Basic Auth. Có thể thử trong `/api/docs` hoặc dùng curl:

```bash
curl -u giangvien:'MAT_KHAU' -F exam_id=De_1 \
  -F question=@de_1.docx -F answer=@dap_an_de_1.docx \
  http://127.0.0.1:8088/api/v1/exams

curl -u giangvien:'MAT_KHAU' -F exam_ref=EXAM_REF -F mode=hybrid \
  -F files=@bai_1.pdf -F files=@bai_2.docx \
  http://127.0.0.1:8088/api/v1/jobs

curl -u giangvien:'MAT_KHAU' http://127.0.0.1:8088/api/v1/jobs/JOB_ID
curl -u giangvien:'MAT_KHAU' http://127.0.0.1:8088/api/v1/jobs/JOB_ID/review-queue
curl -u giangvien:'MAT_KHAU' http://127.0.0.1:8088/api/v1/essays/ESSAY_ID
curl -u giangvien:'MAT_KHAU' -H 'Content-Type: application/json' \
  -d '{"score":6.75,"comment":"Đúng hướng giải; cần làm rõ bước quy nạp.","moodle_user_id":123}' \
  http://127.0.0.1:8088/api/v1/essays/ESSAY_ID/approve
curl -u giangvien:'MAT_KHAU' -H 'Content-Type: application/json' \
  -d '{"assignment_id":456}' \
  http://127.0.0.1:8088/api/v1/essays/ESSAY_ID/moodle-push
curl -u giangvien:'MAT_KHAU' -o ket_qua.csv \
  http://127.0.0.1:8088/api/v1/jobs/JOB_ID/results.csv
curl -u giangvien:'MAT_KHAU' http://127.0.0.1:8088/api/v1/reports/summary
curl -u giangvien:'MAT_KHAU' 'http://127.0.0.1:8088/api/v1/audit?limit=100'
```

`GET /api/v1/moodle/assignments/{course_id}` và `/api/v1/moodle/submissions/{assignment_id}` trả dữ liệu mô phỏng rỗng trong chế độ thử. Trong chế độ live, hai endpoint gọi trực tiếp `MoodleClient` của VieGrader.

## 5. Cấu hình Moodle thật

Sau khi kiểm thử trên Moodle thử nghiệm và đã kiểm tra đúng user ID/assignment ID, người quản trị cấu hình môi trường:

```bash
export MOODLE_BASE_URL='https://moodle.example.edu'
export MOODLE_TOKEN='TOKEN_WEB_SERVICE_DUOC_CAP_CHO_MOD_ASSIGN'
export VIEGRADER_MOODLE_MODE='live'
```

Token Moodle cần quyền các hàm `mod_assign_get_assignments`, `mod_assign_get_submissions`, `mod_assign_save_grade` và quyền phù hợp trong khóa học. Để gửi **điểm thật**, bài phải được duyệt, phải có Moodle user ID, và riêng request phải kèm header `X-VieGrader-Confirm-Live: yes`. Website yêu cầu xác nhận riêng. Sau khi có kết quả `sent`/`simulated`, API không gửi lại chính bài đó. Khi mạng báo lỗi không rõ đã ghi hay chưa, trạng thái `needs_reconciliation`: đối soát với Moodle và xử lý thủ công trước khi tạo lượt chấm khác; không tự thử lại.

## 6. Chạy liên tục và kiểm thử

Tạo tệp môi trường riêng với quyền `chmod 600 /etc/viegrader/portal.env` và áp dụng mẫu `deploy/viegrader-teacher-portal.service` (sửa `User`, `WorkingDirectory`, `ExecStart`).

```bash
sudo cp deploy/viegrader-teacher-portal.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now viegrader-teacher-portal
sudo systemctl status viegrader-teacher-portal
sudo journalctl -u viegrader-teacher-portal -f
```

Chạy thử luồng API với môi trường phát triển có `pytest` và `httpx`:

```bash
python -m pip install pytest httpx
python -m pytest -q tests/test_teacher_portal.py
```

Kiểm thử này dùng mô phỏng chấm và Moodle nên không phải phép đánh giá chất lượng điểm QLoRA, và không ghi vào Moodle thật. Hệ thống chưa có hàng đợi phân tán/khôi phục job đang chạy nếu server khởi động lại; khi khởi động lại cần kiểm tra và chấm lại các job còn `queued`/`running`. Không chạy nhiều Uvicorn workers cho cùng một SQLite và GPU.

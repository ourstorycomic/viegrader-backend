# Triển khai cổng giảng viên VieGrader trên Ubuntu

## Phạm vi phiên bản

Phiên bản này cung cấp API và giao diện để tải đề, đáp án, bài làm; tạo lô chấm; xem toàn văn bài cạnh tài nguyên chấm; lưu điểm, nhận xét và người duyệt; xuất CSV; ghi nhật ký phê duyệt; và mô phỏng hoặc gửi điểm Moodle. Điểm và nhận xét của mô hình là đề xuất. Giảng viên phải đọc bài và phê duyệt trước khi gửi LMS.

Hệ thống hiện dùng SQLite, một worker và một tiến trình Uvicorn. Cấu hình này phù hợp với thí điểm có giám sát trên một máy chủ GPU. Không chạy nhiều Uvicorn worker trên cùng cơ sở dữ liệu và GPU. Trước khi mở rộng nhiều lớp đồng thời, cần chuyển sang PostgreSQL, hàng đợi bền vững và phân quyền theo lớp.

## 1. Cài mã nguồn

```bash
sudo mkdir -p /opt/viegrader /var/lib/viegrader /etc/viegrader
sudo chown -R giangvien:giangvien /opt/viegrader /var/lib/viegrader
cd /opt/viegrader
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[teacher-portal]'
```

Nếu chạy QLoRA, cài PyTorch đúng phiên bản CUDA của máy chủ trước, sau đó cài `.[qlora,teacher-portal]`. Kiểm tra bằng:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_available())"
```

## 2. Cấu hình

```bash
sudo cp deploy/portal.env.example /etc/viegrader/portal.env
sudo chmod 600 /etc/viegrader/portal.env
sudo editor /etc/viegrader/portal.env
```

Thay toàn bộ mật khẩu mẫu, kiểm tra đường dẫn adapter, baseline và rubric. Giữ `VIEGRADER_MOODLE_MODE=simulate` cho đến khi hoàn thành giao dịch ghi–đọc lại trên Moodle sandbox. Tệp môi trường có thể khai báo nhiều tài khoản giảng viên bằng `VIEGRADER_PORTAL_USERS_JSON`; phiên bản này ghi được người thực hiện duyệt nhưng chưa giới hạn dữ liệu theo từng lớp.

## 3. Kiểm thử trước triển khai

```bash
source /opt/viegrader/.venv/bin/activate
python -m pip install pytest httpx
python -m pytest -q tests/test_teacher_portal.py
```

Kiểm thử phải đạt luồng tải tệp, tạo lô, xem toàn văn, lưu nhận xét, duyệt điểm, nhật ký và Moodle simulate. Đây không phải kiểm thử chất lượng mô hình hay Moodle live.

## 4. Chạy bằng systemd

```bash
sudo cp deploy/viegrader-teacher-portal.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now viegrader-teacher-portal
sudo systemctl status viegrader-teacher-portal
curl -u 'giangvien_1:MAT_KHAU' http://127.0.0.1:8088/api/v1/health
```

## 5. Công bố qua HTTPS

Cổng Uvicorn vẫn chỉ nghe ở `127.0.0.1:8088`. Cài Nginx và chứng thư TLS, sao chép cấu hình mẫu rồi thay `viegrader.example.edu` bằng tên miền thật:

```bash
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx
sudo cp deploy/nginx-viegrader.conf /etc/nginx/sites-available/viegrader
sudo ln -s /etc/nginx/sites-available/viegrader /etc/nginx/sites-enabled/viegrader
sudo nginx -t
sudo certbot --nginx -d viegrader.example.edu
sudo systemctl reload nginx
```

Chỉ mở cổng 80/443 trên tường lửa; không mở 8088 ra Internet. Khi nhà trường có SSO, nên thay Basic Auth bằng xác thực tổ chức tại reverse proxy hoặc dịch vụ định danh.

## 6. Quy trình giảng viên

1. Tạo bộ đề bằng mã đề, tệp đề và đáp án; hệ thống kiểm tra rubric tương ứng.
2. Tải bài TXT, DOCX, PDF văn bản hoặc CSV `essay_id,text`, rồi tạo lô chấm.
3. Mở **Xem toàn văn**, đối chiếu các thẻ Bài làm, Đề thi, Đáp án và Rubric.
4. Xem điểm và nhận xét đề xuất, nhập điểm cùng nhận xét của giảng viên; hoặc chuyển bài sang chấm thủ công.
5. Xuất CSV để lưu hồ sơ. Chỉ gửi Moodle sau khi gắn đúng user ID và duyệt điểm.

## 7. Sao lưu và giám sát

Sao lưu hằng ngày `/var/lib/viegrader`, giữ tệp sao lưu ngoài máy chủ và thử phục hồi định kỳ. Theo dõi dịch vụ bằng:

```bash
sudo journalctl -u viegrader-teacher-portal -f
curl -u 'giangvien_1:MAT_KHAU' https://viegrader.example.edu/api/v1/reports/summary
curl -u 'giangvien_1:MAT_KHAU' https://viegrader.example.edu/api/v1/audit?limit=100
```

Không đưa bài làm, đáp án hoặc đầu ra chi tiết vào báo cáo chia sẻ trước khi rà soát thông tin định danh và thời hạn lưu dữ liệu.

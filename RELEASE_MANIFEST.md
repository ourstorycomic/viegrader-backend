# Phạm vi gói phát hành cổng giảng viên

Gói phát hành chứa mã nguồn VieGrader, API FastAPI, giao diện cổng giảng viên, kiểm thử, cấu hình systemd/Nginx, rubric YAML và tài liệu triển khai.

Để bảo vệ dữ liệu nghiên cứu, tệp ZIP bàn giao không chứa:

- bài làm và bảng điểm trong `data/toan_roi_rac/prepared` hoặc `data/toan_roi_rac/splits`;
- hai PDF nguồn RAG và chỉ mục RAG đã tạo;
- thư mục `runs`, cơ sở dữ liệu cổng, tệp upload hoặc log vận hành;
- adapter QLoRA và bí mật kết nối Moodle.

Khi triển khai, người quản trị đặt dữ liệu, mô hình và chỉ mục đã được phê duyệt trên máy chủ rồi khai báo đường dẫn trong `/etc/viegrader/portal.env`. Không đưa token, mật khẩu hoặc dữ liệu sinh viên vào tệp ZIP.

Phiên bản hiện tại hỗ trợ nhiều tài khoản để ghi nhận người duyệt, nhưng chưa phân quyền dữ liệu theo lớp. Trước khi mở cho nhiều đơn vị, cần tích hợp xác thực tổ chức, quyền theo học phần/lớp, PostgreSQL và hàng đợi bền vững.

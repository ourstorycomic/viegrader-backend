# Dataset integration update

- Chọn nguồn Kaggle `vokhoa/vietnamese-it-essays-for-aes-research` làm corpus
  tiếng Việt đúng miền; nhãn ngoài được đánh dấu chưa xác minh, không tự tạo gold.
- Thêm `prepare-dataset` để nhận diện schema, hợp nhất bảng và xuất dấu vân tay
  SHA-256 cùng báo cáo provenance.
- Thêm `split-dataset` để chia 70/10/20 theo người học, hỗ trợ giữ riêng một đề
  và kiểm tra rò rỉ nhóm.
- Thêm rubric bài luận sáu tiêu chí, thang 0–5 mỗi tiêu chí.
- Khóa `train-qlora` không cho dùng dữ liệu chưa chia nếu người chạy không xác
  nhận rõ bằng `--allow-unsplit`.
- Thêm `score-qlora` để suy luận bằng adapter, xuất nhiều lượt phục vụ phân tích
  test–retest.
- Thêm lệnh tải dataset và tài liệu CMD từ dữ liệu thô đến đánh giá bài báo.

Dataset thật và adapter không được đóng gói trong bản phát hành này. Người dùng
cần tải dữ liệu bằng tài khoản Kaggle, xác minh quyền sử dụng và hoàn tất điểm
giám khảo trước khi huấn luyện.

# Báo cáo kiểm chứng kỹ thuật VieGrader 0.7.0

Ngày kiểm chứng: 2026-09-11 (UTC).

## Kết quả kiểm thử

- `python -m compileall`: PASS.
- `python -m pytest -q`: **60 passed** trong 19,59 giây.
- Pipeline `it04-run-all`: PASS từ chuẩn hóa đến báo cáo.
- 5 rubric hợp lệ, tổng cộng 50 tiêu chí.
- Điểm cuối nằm trong thang 0–10; điểm câu đều mang nhãn
  `rule_based_estimate`.

## Lượt chạy end-to-end

Lượt kiểm chứng dùng 150 bài legacy đã có trong gói nguồn 0.6.0, gồm 102 train,
24 validation và 24 test. Đây không phải bộ chuẩn 735 bài, nên các con số dưới
đây chỉ chứng minh pipeline chạy đúng, không phải kết quả nghiên cứu cuối cùng.

Baseline trên test: MAE 1,59375; RMSE 1,80782; QWK 0,28378; Pearson 0,50345;
adjacent agreement ±1 điểm 0,41667.

Calibration chỉ dùng validation và chọn trọng số rubric `w=0.0` vì bản legacy
có độ bao phủ tiêu đề câu thấp. Do đó điểm cuối test bằng baseline, tránh làm
xấu metric bằng rule chưa đủ bằng chứng. Hệ thống vẫn xuất điểm câu và chuyển
bài thiếu độ bao phủ vào hàng đợi phúc tra.

Các JSON trong thư mục này là output gốc của từng bước kiểm chứng. Khi chạy bộ
735 bài, hệ thống sẽ tạo báo cáo mới tại `runs/it04/`.

## Chưa thực hiện trong môi trường kiểm chứng

QLoRA chưa được chạy vì môi trường kiểm chứng không có GPU/model weight. Mã,
config và lệnh Ubuntu đã được kiểm tra cú pháp; cần chạy `hardware-check` trên
server đích trước khi fine-tune.

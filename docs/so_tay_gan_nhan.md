# SỔ TAY GÁN NHÃN — Rubric chấm bài kiểm tra tự luận môn học

- **Mã rubric:** `bkt_mon_hoc_v1`
- **Thang điểm tổng:** 10.0 (bước làm tròn 0.25)
- **Phiên bản:** 1.0

## 1. Quy trình chấm của giám khảo

1. Đọc trọn bài **một lượt** để có ấn tượng tổng thể, chưa cho điểm.
2. Đọc lượt hai, chấm **từng tiêu chí độc lập** theo thứ tự trong bảng dưới.
3. Với mỗi tiêu chí, chọn mức mô tả **sát nhất**; nếu phân vân giữa hai mức, chọn mức thấp hơn rồi ghi chú lí do.
4. Ghi ít nhất **một bằng chứng trích từ bài làm** cho mỗi tiêu chí bị trừ điểm.
5. Không xem điểm của giám khảo khác trước khi nộp (chấm mù đôi).
6. Không suy đoán về người viết; bài đã được ẩn danh.

### Nguyên tắc bắt buộc

- **Chấm theo mô tả, không theo cảm tính.** Nếu mô tả mức không khớp bài thực tế, báo lại để chỉnh rubric, không tự diễn giải riêng.
- **Không phạt hai lần cùng một lỗi** ở hai tiêu chí khác nhau.
- **Không thưởng độ dài.** Bài dài mà loãng không được điểm cao hơn bài ngắn mà chặt.
- **Bỏ qua chữ viết/trình bày** ngoài các yêu cầu đã nêu ở tiêu chí hình thức.
- Nghi vấn gian lận: **không tự trừ điểm**, gắn cờ và chuyển bộ phận xử lí.

## 2. Bảng tiêu chí

### Nội dung kiến thức (`noi_dung`) — trọng số 35%, tối đa 4.0 điểm

> Mức độ đầy đủ, chính xác của kiến thức môn học; bao phủ các ý cốt lõi mà đề bài yêu cầu; sử dụng đúng thuật ngữ chuyên ngành.

| Mức | Điểm | Mô tả biểu hiện | Dấu hiệu định lượng |
|---|---|---|---|
| Tốt | 4.0 | Bao phủ ≥85% ý cốt lõi, kiến thức chính xác, thuật ngữ dùng đúng và nhất quán. | keyword_coverage >= 0.85; không có lỗi kiến thức nghiêm trọng |
| Khá | 3.0 | Bao phủ 65–85% ý cốt lõi, còn thiếu ý phụ hoặc diễn giải chưa sâu. | keyword_coverage 0.65–0.85 |
| Đạt | 2.0 | Bao phủ 40–65% ý cốt lõi, có sai sót nhỏ về kiến thức. | keyword_coverage 0.40–0.65 |
| Chưa đạt | 1.0 | Bao phủ <40% ý cốt lõi, nhiều sai sót kiến thức. | keyword_coverage < 0.40 |
| Không tính điểm | 0.0 | Bỏ trống, lạc đề hoàn toàn, hoặc chép nguyên văn đề bài. | — |

### Lập luận và tổ chức ý (`lap_luan`) — trọng số 25%, tối đa 3.0 điểm

> Bài có luận điểm rõ ràng; lí lẽ và dẫn chứng hỗ trợ luận điểm; bố cục mở – thân – kết mạch lạc; các đoạn liên kết logic.

| Mức | Điểm | Mô tả biểu hiện | Dấu hiệu định lượng |
|---|---|---|---|
| Tốt | 3.0 | Luận điểm rõ, mỗi đoạn một ý, có dẫn chứng/lí lẽ thuyết phục, kết luận chốt được vấn đề. | — |
| Khá | 2.25 | Có bố cục và luận điểm nhưng một vài đoạn còn lan man hoặc thiếu dẫn chứng. | — |
| Đạt | 1.5 | Ý rời rạc, bố cục mờ nhạt, chủ yếu liệt kê chưa phân tích. | — |
| Chưa đạt | 0.75 | Không có bố cục, không phân biệt được luận điểm và dẫn chứng. | — |
| Không tính điểm | 0.0 | Bài quá ngắn hoặc không thành đoạn văn. | — |

### Diễn đạt và ngôn ngữ (`dien_dat`) — trọng số 20%, tối đa 2.0 điểm

> Dùng từ chính xác, câu đúng ngữ pháp, văn phong học thuật, hạn chế lỗi chính tả và lỗi dấu câu.

| Mức | Điểm | Mô tả biểu hiện | Dấu hiệu định lượng |
|---|---|---|---|
| Tốt | 2.0 | Hầu như không lỗi chính tả/ngữ pháp (<1%), từ vựng phong phú, văn phong học thuật. | spell_error_rate < 0.01; informal_rate < 0.005 |
| Khá | 1.5 | Một vài lỗi nhỏ (1–3%), không cản trở việc hiểu. | spell_error_rate 0.01–0.03 |
| Đạt | 1.0 | Lỗi xuất hiện thường xuyên (3–6%), câu dài lủng củng, còn dùng văn nói. | — |
| Chưa đạt | 0.5 | Lỗi dày đặc (>6%), nhiều câu tối nghĩa. | — |
| Không tính điểm | 0.0 | Không đọc hiểu được. | — |

### Hình thức và tính liêm chính (`hinh_thuc`) — trọng số 20%, tối đa 1.0 điểm

> Trình bày đúng yêu cầu về độ dài, định dạng; có trích dẫn nguồn khi cần; bài làm là sản phẩm độc lập của người học.

| Mức | Điểm | Mô tả biểu hiện | Dấu hiệu định lượng |
|---|---|---|---|
| Tốt | 1.0 | Đúng độ dài quy định, trình bày sạch, có trích dẫn nguồn hợp lệ, không trùng lặp. | — |
| Khá | 0.75 | Đạt yêu cầu cơ bản, trích dẫn chưa đầy đủ hoặc trình bày còn cẩu thả. | — |
| Đạt | 0.5 | Thiếu/thừa độ dài đáng kể hoặc không trích dẫn khi có dùng nguồn. | — |
| Chưa đạt | 0.25 | Vi phạm nhiều yêu cầu hình thức. | — |
| Không tính điểm | 0.0 | Nghi vấn sao chép/gian lận, cần chuyển giám khảo xử lí. | — |

## 3. Quy định chấm tự động đang áp dụng

Các quy định này do hệ thống thi hành sau khi mô hình dự đoán điểm. Giám khảo cần nắm để hiểu vì sao điểm máy khác điểm mình.

| Mã | Điều kiện | Hành động | Mô tả |
|---|---|---|---|
| R01_bai_trong | `n_words < 20` | `cap_total` = 0.0 | Bài trống hoặc dưới 20 từ -> 0 điểm toàn bài. |
| R02_lac_de | `sim_prompt < 0.15 and n_words >= 20` | `review` | Độ tương đồng ngữ nghĩa với đề bài quá thấp -> nghi ngờ lạc đề, chuyển giám khảo. |
| R03_chep_de | `prompt_copy_ratio > 0.60` | `set_criterion` = 0.0 | Chép lại đề bài chiếm >60% nội dung -> nội dung = 0. |
| R04_qua_ngan | `n_words < 0.5 * min_words` | `cap_total` = 0.5 * scale_max | Bài dưới 50% độ dài tối thiểu -> trần điểm tổng 50%. |
| R05_trung_lap | `dup_ratio >= 0.80` | `review` | Trùng lặp cao với bài khác trong cùng lớp -> chuyển xử lí liêm chính học thuật. |
| R06_loi_chinh_ta_nang | `spell_error_rate > 0.15` | `subtract` = 0.5 | Tỉ lệ lỗi chính tả trên 15% -> trừ tối đa 0.5 điểm tổng. |
| R07_do_tin_cay_thap | `confidence < 0.50` | `review` | Mô hình thiếu tự tin -> phúc tra thủ công. |
| R07b_nam_ranh_gioi_muc | `max_snap_gap > 0.90` | `review` | Điểm thô nằm đúng ranh giới giữa hai mức rubric -> phúc tra. |
| R08_diem_ranh_gioi | `4.75 <= total <= 5.25` | `review` | Điểm rơi sát ngưỡng đạt (4.75–5.25) -> phúc tra thủ công. |

## 4. Quy tắc phân xử bất đồng

- Hai giám khảo lệch **≤ 1.0 điểm**: lấy trung bình.
- Lệch **> 1.0 điểm**: giám khảo thứ ba chấm mù; điểm cuối là trung bình của hai điểm gần nhau nhất.
- Lệch **> 30% thang điểm**: đưa ra họp hội đồng, ghi biên bản, cân nhắc sửa mô tả mức trong rubric.
- Mọi ca phân xử phải được lưu để tính lại QWK sau khi hợp nhất nhãn.

## 5. Ngưỡng chất lượng gán nhãn cần đạt

| Chỉ số | Ngưỡng tối thiểu | Ý nghĩa |
|---|---|---|
| QWK giữa 2 giám khảo (điểm tổng) | ≥ 0.70 | Rubric đủ rõ để người khác nhau hiểu giống nhau |
| QWK từng tiêu chí | ≥ 0.60 | Tiêu chí không mơ hồ |
| Adjacent agreement (lệch ≤ 1 điểm) | ≥ 0.90 | Không có bất đồng lớn |
| Krippendorff's α | ≥ 0.667 | Ngưỡng dùng được cho kết luận sơ bộ |
| ICC(2,k) | ≥ 0.75 | Điểm trung bình nhiều giám khảo đủ tin cậy |

Nếu chưa đạt: **sửa rubric và tập huấn lại**, không ép mô hình học nhãn nhiễu.

## 7. Nhật ký phiên chấm

Giám khảo ghi lại: ngày chấm, số bài, thời gian trung bình mỗi bài, các trường hợp khó. Hệ thống dùng nhật ký này để phát hiện mệt mỏi và trôi chuẩn (drift) theo thời gian.
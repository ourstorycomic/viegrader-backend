# QUY TRÌNH XÂY DỰNG BỘ DỮ LIỆU VÀ QUY ĐỊNH CHẤM ĐIỂM TỰ ĐỘNG

Tài liệu phương pháp kèm theo công cụ `viegrader`, phục vụ đề tài NCKH 2026 về
chấm điểm tự động bài kiểm tra tự luận tiếng Việt.

---

## PHẦN A — THU THẬP DỮ LIỆU

### A.1. Xác định phạm vi trước khi thu thập

Chốt bốn tham số này trước, vì thay đổi giữa chừng sẽ làm hỏng tính so sánh của
bộ dữ liệu:

| Tham số | Khuyến nghị cho đề tài |
|---|---|
| Loại bài | Câu hỏi tự luận của bài kiểm tra giữa/cuối kỳ, 300–1500 từ |
| Số đề bài | ≥ 5 đề khác nhau (để kiểm tra khả năng khái quát sang đề mới) |
| Số bài/đề | ≥ 150 bài/đề; tổng ≥ 800 bài |
| Nguồn | Bài làm thật của sinh viên, có sự đồng ý sử dụng cho nghiên cứu |

**Vì sao cần nhiều đề bài:** mô hình chấm tự luận rất dễ "học thuộc" đặc điểm của
một đề. Nếu chỉ có một đề, kết quả QWK cao không chứng minh được gì. Thiết kế
thực nghiệm bắt buộc phải có kịch bản **prompt-independent** (huấn luyện trên đề
1–4, kiểm tra trên đề 5).

### A.2. Kích thước mẫu tối thiểu

| Mục đích | Số bài cần gán nhãn |
|---|---|
| Vòng thử, chốt rubric | 150–200 (toàn bộ chấm kép) |
| Huấn luyện được | ≥ 400 |
| Đủ để công bố | 800–1500, phủ đều dải điểm |
| Kiểm tra ngoài đề | ≥ 150 của đề chưa từng huấn luyện |

Phân bố điểm phải **phủ đều**. Nếu 80% bài rơi vào 6–8 điểm, mô hình sẽ chỉ học
được cách đoán "khoảng 7" và mọi chỉ số tương quan đều ảo. Khi lớp học thực tế
lệch như vậy, cần chủ động lấy thêm bài ở hai đầu dải điểm.

### A.3. Hồ sơ pháp lý và đạo đức

Bắt buộc có trước khi thu thập:

1. **Phiếu đồng ý** của sinh viên (hoặc phê duyệt của hội đồng khoa học nhà
   trường cho việc sử dụng dữ liệu học tập vào mục đích nghiên cứu).
2. **Cam kết ẩn danh**: mọi định danh trực tiếp bị thay bằng thẻ giữ chỗ; mã sinh
   viên chỉ tồn tại dưới dạng băm HMAC. Khoá băm lưu riêng, không kèm bộ dữ liệu.
3. **Giới hạn mục đích**: dữ liệu không dùng để đánh giá cá nhân giảng viên hay
   sinh viên ngoài phạm vi nghiên cứu.
4. **Kế hoạch huỷ dữ liệu** sau khi đề tài nghiệm thu.

### A.4. Định dạng đầu vào chuẩn

`viegrader` nhận cả thư mục bài làm lẫn file bảng. Bảng chuẩn:

| Cột | Bắt buộc | Ý nghĩa |
|---|---|---|
| `essay_id` | Không (tự sinh) | Mã bài |
| `text` | **Có** | Nội dung bài làm |
| `prompt_id` | Nên có | Mã đề bài |
| `prompt_text` | Nên có | Nội dung đề bài (dùng để phát hiện lạc đề) |
| `course_id` | Không | Mã học phần |
| `student_id` | Không | Mã SV — sẽ bị băm và xoá khỏi bảng đầu ra |
| `keywords` | Nên có | Ý cốt lõi trong đáp án, ngăn bằng `;`, đồng nghĩa ngăn bằng `\|` |

---

## PHẦN B — LÀM SẠCH DỮ LIỆU

Bảy bước, chạy tuần tự bằng `viegrader clean`. Mỗi bước đều ghi vết số lần can
thiệp để đưa vào phần mô tả tiền xử lí của báo cáo.

### B.1. Chuẩn hoá cấu trúc bảng

- Nhận diện cột nội dung theo nhiều tên gọi (`text`, `essay`, `bai_lam`, `noi_dung`).
- Sinh `essay_id` nếu thiếu; ép kiểu chuỗi cho mọi cột định danh.
- Giữ lại `raw_text` — bản gốc trước mọi can thiệp. **Không bao giờ ghi đè bản gốc**,
  vì cần đối chiếu khi thẩm định lại kết quả.

### B.2. Giả danh hoá (pseudonymisation)

| Đối tượng | Cách xử lí |
|---|---|
| Họ tên có nhãn ("Họ và tên: …") | → `<TEN>` |
| Mã sinh viên / SBD | → `<MSSV>`, đồng thời băm HMAC-SHA256 vào `student_hash` |
| Email, số điện thoại | → `<EMAIL>`, `<SDT>` |
| CCCD, ngày sinh | → `<CCCD>`, `<NGAYSINH>` |
| Lớp, khoá | → `<LOP>` |
| Phần đầu bài (≤6 dòng) chứa nhãn định danh | cắt bỏ |

Băm HMAC (không phải hash trần) để không thể dò ngược bằng cách thử toàn bộ dải
mã sinh viên — dải này nhỏ và có quy luật nên hash trần là không an toàn.

### B.3. Chuẩn hoá văn bản tiếng Việt

Đây là phần đặc thù ngôn ngữ, thường bị làm sơ sài và gây sai lệch về sau.

| Vấn đề | Ví dụ | Cách xử lí |
|---|---|---|
| **Unicode dựng sẵn vs tổ hợp** | `ế` = 1 hay 3 code point | Chuẩn hoá NFC toàn bộ |
| **Kiểu đặt dấu thanh** | `hòa` vs `hoà`, `thủy` vs `thuỷ` | Thống nhất một kiểu (mặc định kiểu mới) |
| Ký tự vô hình, thẻ HTML | `&nbsp;`, `<span>`, zero-width | Gỡ bỏ |
| Teencode, viết tắt | `ko`, `dc`, `ntn`, `sv` | Giãn về dạng chuẩn, **đếm trước khi giãn** |
| Lặp ký tự | `hayyyy` | Rút về 1 ký tự |
| Emoji, emoticon | 😀 `:))` | Gỡ bỏ |
| Dấu câu | `!!!`, thiếu khoảng trắng sau dấu phẩy | Chuẩn hoá |
| URL | link tham khảo | Thay bằng `<URL>` |

**Điểm phương pháp cần lưu ý:** giãn teencode làm mất tín hiệu "dùng văn nói" —
vốn là căn cứ chấm tiêu chí *Diễn đạt*. Vì vậy hệ thống **đếm số lần xuất hiện
trước khi giãn** và giữ lại thành đặc trưng `teencode_rate`. Đây là nguyên tắc
chung: mỗi khi làm sạch xoá đi một tín hiệu, phải chuyển tín hiệu đó thành đặc
trưng, nếu không mô hình sẽ mù trước lỗi mà nó cần phạt.

Chỉ số `diacritic_ratio` (tỉ lệ ký tự mang dấu) được tính ở cả bản gốc và bản
sạch. Bài tiếng Việt viết đúng có tỉ lệ 0.18–0.35; dưới 0.05 là bài viết không
dấu và bị loại vì PhoBERT hoạt động rất kém trên văn bản không dấu.

### B.4. Kiểm định chất lượng — ba nhóm quyết định

Nguyên tắc: **không âm thầm xoá dữ liệu**. Mọi bản ghi bị loại đều xuất ra
`rejected.csv` kèm lí do, để báo cáo thống kê được tỉ lệ và nguyên nhân.

| Quyết định | Điều kiện | Hành động |
|---|---|---|
| `reject` | < 20 từ; > 5000 từ; `diacritic_ratio` < 0.05; > 30% ký tự không phải tiếng Việt; tỉ lệ từ khác nhau < 0.15 (spam lặp) | Loại khỏi bộ dữ liệu |
| `flag` | Thiếu dấu nhiều (0.05–0.12); nhiều dòng lặp; viết hoa toàn bộ; không tách được câu | Giữ lại, đánh dấu để xem xét |
| `keep` | Còn lại | Đưa vào bộ dữ liệu |

### B.5. Phát hiện trùng lặp

Hai mức, chạy **trong phạm vi từng đề bài**:

1. **Trùng tuyệt đối** — băm SHA1 nội dung đã chuẩn hoá. Bản sao bị loại, giữ lại
   một bản.
2. **Trùng gần đúng** — MinHash 128 hàm băm trên tập 5-shingle từ, lọc ứng viên
   bằng LSH banding (32 band), rồi xác nhận bằng Jaccard chính xác. Độ phức tạp
   gần tuyến tính, chạy được với hàng chục nghìn bài trên CPU thường.

Ngoài Jaccard, hệ thống tính thêm **containment** (tỉ lệ shingle của bài A nằm
trong bài B) vì Jaccard đánh giá thấp trường hợp "bài ngắn chép một phần bài dài".

Kết quả dùng cho hai việc:

- **Làm sạch**: loại bản sao **trước khi chia train/test**. Bỏ qua bước này là lỗi
  rò rỉ dữ liệu kinh điển — cùng một bài xuất hiện ở cả hai tập làm chỉ số đẹp giả tạo.
- **Chấm điểm**: đặc trưng `dup_ratio` kích hoạt quy định liêm chính học thuật (R05).

### B.6. Tỉ lệ chép đề bài

`prompt_copy_ratio` = tỉ lệ 4-shingle của bài làm trùng với đề bài. Sinh viên
không biết làm thường chép lại đề rồi viết vài câu — hành vi này phải bị phát hiện
và không được tính vào điểm nội dung (quy định R03).

### B.7. Báo cáo tiền xử lí

`clean_report.json` / `clean_report.txt` chứa: số bản ghi vào/ra/bị loại, phân bố
lí do loại, số lần can thiệp theo từng loại chuẩn hoá, số thông tin cá nhân đã
che, số nhóm trùng lặp. Đây là bảng số liệu cho mục "Mô tả bộ dữ liệu" của báo cáo.

---

## PHẦN C — GÁN NHÃN

### C.1. Ba vòng gán nhãn

| Vòng | Số bài | Ai chấm | Mục đích |
|---|---|---|---|
| **1. Thử** | 150–200 | Toàn bộ ≥ 2 giám khảo độc lập | Chốt rubric, đo QWK, chọn bài neo |
| **2. Chính** | Phần còn lại | 1 giám khảo/bài + 25–30% chấm kép + 5% bài neo ngầm | Sinh nhãn huấn luyện |
| **3. Mở rộng** | Theo nhu cầu | Ưu tiên bài mô hình bất định nhất | Active learning, tăng hiệu quả công sức |

**Vòng 1 là bắt buộc.** Rubric viết trên giấy hầu như luôn có chỗ mơ hồ mà chỉ lộ
ra khi hai người chấm cùng một bài. Sửa rubric ở vòng 1 tốn vài buổi; phát hiện
lỗi rubric sau khi đã gán nhãn 800 bài thì phải làm lại từ đầu.

### C.2. Chiến lược lấy mẫu

- **Vòng 1** — phân tầng theo độ dài bài (5 tầng, lấy đều). Độ dài là proxy tốt
  cho năng lực, giúp mẫu thử phủ đều dải điểm ngay từ đầu.
- **Vòng 2** — phân công vòng tròn (round-robin) để mỗi giám khảo chấm số lượng
  tương đương; 25–30% bài được chấm kép ngẫu nhiên để **giám sát chất lượng liên
  tục**, không chỉ đo một lần ở vòng 1.
- **Bài neo ngầm (seeded anchors)** — 5% lượt chấm là các bài đã có điểm chuẩn
  được hội đồng thống nhất, chèn ngẫu nhiên vào danh sách của từng giám khảo mà
  họ không biết. So điểm giám khảo với điểm chuẩn theo từng lô cho biết ai đang
  **trôi chuẩn** theo thời gian (thường do mệt mỏi sau nhiều giờ chấm).
- **Vòng 3** — lấy mẫu theo độ bất định: ưu tiên bài có độ lệch chuẩn dự đoán giữa
  các mô hình bootstrap cao nhất, có ràng buộc đa dạng theo đề bài. Với cùng ngân
  sách chấm tay, cách này cải thiện mô hình nhanh hơn lấy mẫu ngẫu nhiên.

### C.3. Quy trình chấm của giám khảo

1. Đọc trọn bài một lượt, **chưa cho điểm** — tránh hiệu ứng ấn tượng đầu tiên.
2. Đọc lượt hai, chấm **từng tiêu chí độc lập** theo thứ tự rubric.
3. Phân vân giữa hai mức → chọn mức thấp hơn và ghi chú lí do (quy ước thống nhất
   để giảm phương sai giữa giám khảo).
4. Ghi ít nhất **một bằng chứng trích nguyên văn** cho mỗi tiêu chí bị trừ điểm.
5. Chấm mù đôi: không xem điểm của người khác trước khi nộp.
6. Nghi vấn gian lận → **gắn cờ, không tự trừ điểm**.

Bốn nguyên tắc chống lỗi hệ thống:

- **Chấm theo mô tả, không theo cảm tính.** Mô tả mức không khớp thực tế → báo sửa
  rubric, không tự diễn giải riêng.
- **Không phạt hai lần cùng một lỗi** ở hai tiêu chí khác nhau.
- **Không thưởng độ dài.** Bài dài mà loãng không được điểm cao hơn bài ngắn mà chặt.
- **Bỏ qua chữ viết và trình bày** ngoài các yêu cầu đã nêu ở tiêu chí hình thức.

`viegrader handbook` sinh sổ tay này **từ chính file rubric đang chạy trong hệ
thống**, tránh tình trạng "rubric trên giấy" khác "rubric trong code" — lỗi thường
gặp làm hỏng kết quả nghiên cứu.

### C.4. Ngưỡng chất lượng gán nhãn

| Chỉ số | Ngưỡng | Ý nghĩa |
|---|---|---|
| QWK điểm tổng giữa 2 giám khảo | ≥ 0.70 | Rubric đủ rõ |
| QWK từng tiêu chí | ≥ 0.60 | Tiêu chí không mơ hồ |
| Adjacent agreement (lệch ≤ 1 điểm) | ≥ 0.90 | Không có bất đồng lớn |
| Krippendorff's α (thứ bậc) | ≥ 0.667 | Ngưỡng kết luận sơ bộ |
| ICC(2,k) | ≥ 0.75 | Điểm trung bình nhiều giám khảo đủ tin cậy |

Chưa đạt → **sửa rubric và tập huấn lại**, không ép mô hình học nhãn nhiễu. Giá
trị QWK giữa hai giám khảo cũng chính là **trần hiệu năng** của mô hình: không hệ
thống nào vượt được độ nhất quán của chính con người trên cùng dữ liệu đó.

### C.5. Phân xử bất đồng và sinh nhãn vàng

| Tình huống | Xử lí |
|---|---|
| 1 giám khảo | Lấy nguyên điểm, đánh dấu `single_scored` |
| 2 giám khảo, lệch ≤ 10% thang điểm | Trung bình |
| 2 giám khảo, lệch > 10% | Giám khảo thứ ba chấm mù → trung bình hai điểm gần nhau nhất |
| Lệch > 30% thang điểm | Họp hội đồng, ghi biên bản, cân nhắc sửa mô tả mức |

Sau khi hợp nhất, điểm từng tiêu chí được **neo về đúng các mức rubric cho phép**
và điểm tổng làm tròn theo `scale_step`. Điểm vàng dạng liên tục tuỳ tiện (ví dụ
2.37/4) không tồn tại trong rubric nên không được để lọt vào tập huấn luyện.

### C.6. Giám sát giám khảo

- **Độ khắt khe** (`rater_severity`): ước lượng bằng mô hình cộng tính rút gọn
  theo tinh thần Many-Facet Rasch — `điểm = μ + năng lực bài + độ dễ dãi giám khảo`.
  Giám khảo có `leniency` < −0.25 là khắt khe, > 0.25 là dễ dãi. Chênh lệch hệ
  thống cần được hiệu chỉnh hoặc ít nhất phải báo cáo.
- **Độ trôi chuẩn** (`rater_drift`): sai số trên bài neo theo từng lô. Nếu độ lệch
  tăng dần theo lô, cần giới hạn số bài mỗi phiên chấm.

---

## PHẦN D — QUY ĐỊNH CHẤM ĐIỂM TỰ ĐỘNG THEO RUBRIC

### D.1. Vì sao tách rule engine khỏi mô hình

Quy chế đào tạo có những điều khoản mang tính pháp lý: bài trống 0 điểm, nghi sao
chép chuyển hội đồng, bài không đủ độ dài bị giới hạn điểm. Những điều này **không
được để mô hình "học lấy"** — mô hình học từ dữ liệu sẽ chỉ tuân thủ *phần lớn*
thời gian, và không thể giải trình khi có khiếu nại.

Vì vậy `viegrader` chia đôi trách nhiệm:

```
Mô hình học máy  →  điểm thành phần theo tiêu chí (có thể sai, có độ bất định)
Rule engine      →  thi hành quy chế (tuyệt đối, kiểm tra được, ghi vết được)
```

Mọi quy định khai báo trong YAML, được đánh giá bằng AST giới hạn (không truy cập
được `import`, thuộc tính, builtins), và mỗi lần kích hoạt đều được ghi vào
`applied_rules` của kết quả.

### D.2. Chuỗi thi hành

```
1. Mô hình dự đoán điểm liên tục cho từng tiêu chí
2. Neo (snap) về đúng mức rubric gần nhất
3. Tính điểm tổng = Σ (điểm_tiêu_chí / max_tiêu_chí) × trọng_số × thang_điểm
4. THI HÀNH QUY ĐỊNH CHẤM  ← rule engine
5. Làm tròn theo scale_step
6. Quyết định phúc tra
7. Sinh phản hồi kèm bằng chứng
```

### D.3. Chín quy định trong rubric mẫu

| Mã | Điều kiện | Hành động | Căn cứ |
|---|---|---|---|
| R01 | `n_words < 20` | Tổng = 0, **dừng** | Không có bài làm |
| R02 | `sim_prompt < 0.15` | Chuyển phúc tra, cờ `NGHI_LAC_DE` | Máy không đủ thẩm quyền kết luận lạc đề |
| R03 | `prompt_copy_ratio > 0.60` | Tiêu chí nội dung = 0 | Chép đề không phải là nội dung |
| R04 | `n_words < 0.5 × min_words` | Trần điểm tổng = 50% thang | Không đủ dung lượng để đạt yêu cầu |
| R05 | `dup_ratio ≥ 0.80` | Chuyển phúc tra, cờ `NGHI_SAO_CHEP` | Liêm chính học thuật thuộc thẩm quyền hội đồng |
| R06 | `spell_error_rate > 0.15` | Trừ 0.5 điểm tổng | Lỗi chính tả dày đặc |
| R07 | `confidence < 0.50` | Chuyển phúc tra | Mô hình không đủ tự tin |
| R07b | `max_snap_gap > 0.90` | Chuyển phúc tra | Điểm nằm đúng ranh giới hai mức |
| R08 | `4.75 ≤ total ≤ 5.25` | Chuyển phúc tra | Sát ngưỡng đạt/không đạt |

### D.4. Bốn nguyên tắc thiết kế quy định

**1. Việc gì ảnh hưởng tới quyền lợi người học thì máy chỉ được gắn cờ, không được
kết luận.** Lạc đề và sao chép đều dùng `action: review`, không dùng `cap_total`.
Máy sai ở hai loại này gây hậu quả nặng và không sửa được bằng phúc khảo thông thường.

**2. Ràng buộc cứng đặt trước, quy định mềm đặt sau.** R01 có `stop: true` — bài
trống không cần chạy tiếp các quy định khác.

**3. Mỗi quy định phải truy được về một điều khoản trong đề cương học phần.** Quy
định không có căn cứ văn bản thì không đưa vào rubric, dù về kỹ thuật có vẻ hợp lý.

**4. Ngưỡng phải hiệu chỉnh trên dữ liệu thật, không lấy theo cảm tính.**
`suggest_review_threshold` sinh đường đánh đổi giữa khối lượng phúc tra và tỉ lệ
bắt được bài chấm sai; giảng viên chọn điểm làm việc phù hợp với nguồn lực.

### D.5. Cơ chế "biết mình không biết"

Chỉ số `confidence` của mỗi tiêu chí kết hợp hai nguồn bất định:

- **Bất định của mô hình** — độ lệch chuẩn giữa 5 mô hình bootstrap.
- **Bất định của việc neo mức** — khoảng cách từ điểm liên tục tới mức rubric gần
  nhất, chuẩn hoá theo nửa khoảng cách giữa hai mức. Giá trị gần 1 nghĩa là bài
  nằm đúng ranh giới, việc neo về mức nào cũng tuỳ tiện.

Nguồn thứ hai quan trọng không kém nguồn thứ nhất và thường bị bỏ sót: mô hình có
thể rất "chắc chắn" rằng bài này đáng 2.5/4 điểm, nhưng rubric chỉ có mức 2.0 và
3.0 — đây chính xác là loại bài cần người xem lại.

Trên dữ liệu mô phỏng, cơ chế này giúp phúc tra 10% số bài nhưng bắt được 83% số
bài mà mô hình chấm lệch quá 1 điểm.

### D.6. Phản hồi cho sinh viên

Bốn yêu cầu sư phạm với mọi nhận xét tự động:

1. **Gắn với tiêu chí rubric** — nêu rõ điểm từng tiêu chí và mô tả mức đạt được.
2. **Dẫn bằng chứng cụ thể** — từ nghi sai chính tả, cặp từ hay nhầm, ý cốt lõi
   còn thiếu, độ dài câu trung bình. Phần bằng chứng định lượng **luôn do hệ thống
   sinh**, kể cả khi có LLM, để bảo đảm chính xác.
3. **Nêu hành động sửa được** — "mỗi đoạn nên có một câu chủ đề đứng đầu", không
   phải "cần cải thiện lập luận".
4. **Không phán xét người viết** — nhận xét về bài làm, không về người học.

Mọi phản hồi đều kèm dòng thông báo về quyền phúc khảo; bài được chuyển phúc tra
ghi rõ điểm hiển thị chỉ là tham khảo.

---

## PHẦN E — THIẾT KẾ THỰC NGHIỆM VÀ ĐÁNH GIÁ

### E.1. Chia dữ liệu

| Kịch bản | Cách chia | Trả lời câu hỏi |
|---|---|---|
| **Trong đề** | Ngẫu nhiên phân tầng theo điểm, cùng tập đề | Mô hình chấm được bài mới của đề đã biết không? |
| **Ngoài đề** | Giữ nguyên 1–2 đề làm tập kiểm tra | Mô hình có khái quát sang đề mới không? |
| **Theo lớp** | Tách theo lớp/học kỳ | Có ổn định qua các khoá không? |

Bắt buộc loại trùng lặp **trước khi** chia.

### E.2. Mô hình cơ sở để so sánh

Bài báo cần ít nhất ba mốc so sánh, nếu không thì không chứng minh được đóng góp:

1. **Dự đoán theo trung bình** — luôn trả điểm trung bình tập huấn luyện.
2. **Chỉ độ dài** — hồi quy một biến `n_words`. Mốc này quan trọng nhất: nhiều
   nghiên cứu AES đạt tương quan cao chỉ vì học được "bài dài = điểm cao".
3. **Đặc trưng tường minh** — không dùng PhoBERT/LLM.

Rồi mới đến các cấu hình đề xuất: + PhoBERT, + LLM, và mô hình lai đầy đủ.

### E.3. Bộ chỉ số

| Chỉ số | Ngưỡng tham chiếu |
|---|---|
| QWK | ≥ 0.70 dùng được; ≥ 0.80 tương đương giám khảo thứ hai |
| Pearson / Spearman | báo cáo cả hai |
| MAE / RMSE | trên thang điểm gốc, dễ diễn giải cho giảng viên |
| Exact / Adjacent agreement | adjacent ≥ 0.90 |
| SMD | \|SMD\| ≤ 0.15 — không thiên lệch hệ thống |
| std_ratio | 0.85–1.15 — không dồn điểm về trung bình |
| QWK(máy,người) / QWK(người,người) | ≥ 0.90 mới coi là đạt chuẩn thay thế giám khảo thứ hai |

`std_ratio` hay bị bỏ qua nhưng rất quan trọng: mô hình hồi quy tối ưu MSE có xu
hướng dồn dự đoán về trung bình. Kết quả là MAE đẹp nhưng hệ thống mất khả năng
phân biệt bài giỏi với bài khá — vô dụng trong thực tế. Hiệu chỉnh isotonic trong
`TraitModel` xử lý vấn đề này.

### E.4. Kiểm tra thiên lệch

- **Theo độ dài bài** (`length_bias`) — chia 5 nhóm theo độ dài, so MAE và SMD.
  Chênh lệch lớn nghĩa là mô hình đang dùng độ dài làm lối tắt.
- **Theo đề bài, lớp, học kỳ** (`fairness_by_group`) — cảnh báo khi \|SMD\| > 0.25
  hoặc chênh MAE giữa các nhóm > 0.5 điểm.

### E.5. Đánh giá cơ chế phúc tra

Chỉ số quan trọng nhất là `recall_of_errors`: trong số bài mô hình chấm lệch quá
1 điểm, bao nhiêu phần trăm đã được chuyển phúc tra. Một hệ thống có QWK trung
bình nhưng `recall_of_errors` cao vẫn triển khai được an toàn; ngược lại thì không.

### E.6. Ghi nhận để tái lập

Báo cáo cần ghi rõ: phiên bản `viegrader`, mã băm rubric, tên và phiên bản mô hình
LLM (nếu dùng), seed ngẫu nhiên, cấu hình phần cứng, thời gian chấm trung bình mỗi
bài. Với mô-đun LLM, kết quả có thể thay đổi giữa các phiên bản mô hình — phải nêu
đây là giới hạn về tính tái lập.

---

## PHẦN F — CHECKLIST TRIỂN KHAI

**Trước khi thu thập**

- [ ] Có phê duyệt đạo đức / phiếu đồng ý
- [ ] Chốt số đề bài, số bài mỗi đề, thang điểm
- [ ] Rubric v1 đã qua `viegrader check-rubric`

**Sau khi thu thập**

- [ ] Chạy `viegrader clean`, đọc kỹ `rejected.csv`
- [ ] Tỉ lệ loại < 10%; nếu cao hơn, xem lại khâu thu thập
- [ ] Kiểm tra thủ công 20 bài xem việc ẩn danh có sót không

**Gán nhãn**

- [ ] Vòng thử 150–200 bài, chấm kép toàn bộ
- [ ] QWK điểm tổng ≥ 0.70 → mới sang vòng chính
- [ ] Chọn 3 bài neo cho mỗi mức điểm
- [ ] Vòng chính có 25–30% chấm kép và 5% bài neo ngầm
- [ ] Kiểm tra độ khắt khe và độ trôi chuẩn từng giám khảo

**Huấn luyện và đánh giá**

- [ ] Loại trùng lặp trước khi chia dữ liệu
- [ ] Có đủ ba mô hình cơ sở để so sánh
- [ ] Có kịch bản kiểm tra ngoài đề
- [ ] Kiểm tra thiên lệch độ dài
- [ ] Hiệu chỉnh ngưỡng phúc tra trên dữ liệu thật

**Trước khi dùng thật**

- [ ] Chạy song song với chấm tay ít nhất một học kỳ
- [ ] Công bố cho sinh viên biết bài được chấm có hỗ trợ của máy
- [ ] Có quy trình phúc khảo bằng người rõ ràng
- [ ] Không dùng điểm máy làm điểm cuối cùng khi chưa qua giai đoạn thử nghiệm

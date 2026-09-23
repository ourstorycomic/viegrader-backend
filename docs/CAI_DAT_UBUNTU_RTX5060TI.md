# Cài VieGrader 0.5 trên Ubuntu Server + RTX 5060 Ti 16 GB

Tài liệu này dùng cho Ubuntu Server 22.04/24.04 LTS, một GPU RTX 5060 Ti 16 GB.
Khuyến nghị máy có ít nhất 32 GB RAM, 100 GB SSD trống và Python 3.10–3.12.

## 1. Driver NVIDIA

RTX 5060 Ti là GPU Blackwell (`sm_120`). Dùng driver Linux 570.26 trở lên và
PyTorch 2.7 trở lên với CUDA 12.8. Không cần cài toàn bộ CUDA Toolkit nếu chỉ
chạy wheel PyTorch `cu128`; driver NVIDIA là thành phần bắt buộc.

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo reboot
```

Sau khi máy khởi động lại:

```bash
nvidia-smi
```

Nếu Secure Boot đang bật và `nvidia-smi` không thấy GPU, ký module theo hướng
dẫn của Ubuntu hoặc tắt Secure Boot trong firmware rồi cài lại driver.

## 2. Cài native (khuyến nghị để phát triển)

```bash
sudo mkdir -p /opt/viegrader/app
sudo chown -R "$USER":"$USER" /opt/viegrader
unzip VieGrader_0.5.1_Ubuntu_RTX5060Ti_CMD.zip -d /opt/viegrader/app
cd /opt/viegrader/app
chmod +x scripts/install_ubuntu_rtx5060ti.sh scripts/check_ubuntu_gpu.py
./scripts/install_ubuntu_rtx5060ti.sh
source .venv/bin/activate
viegrader hardware-check --strict-blackwell
```

Script cài PyTorch từ index `cu128`, các thư viện QLoRA/RAG/GUI/báo cáo và chạy
test. Không thay wheel Torch bằng gói CUDA khác sau bước này.

Tạo cấu hình và bí mật:

```bash
sudo mkdir -p /etc/viegrader
sudo cp config.ubuntu.example.json /etc/viegrader/config.ubuntu.json
sudo cp deploy/viegrader.env.example /etc/viegrader/viegrader.env
sudo chmod 600 /etc/viegrader/viegrader.env
sudo nano /etc/viegrader/viegrader.env
```

Đặt `HF_TOKEN` nếu model Hugging Face yêu cầu chấp nhận điều khoản truy cập.
Không ghi token vào JSON, notebook, Git hoặc báo cáo.

Chạy thử GUI:

```bash
set -a; source /etc/viegrader/viegrader.env; set +a
viegrader ubuntu-gui --config /etc/viegrader/config.ubuntu.json
```

Mặc định GUI nghe tại `127.0.0.1:7860`. Từ máy cá nhân có thể dùng:

```bash
ssh -L 7860:127.0.0.1:7860 user@server
```

rồi mở `http://127.0.0.1:7860`.

## 3. Chạy dịch vụ systemd

```bash
sudo useradd --system --home /opt/viegrader --shell /usr/sbin/nologin viegrader || true
sudo chown -R viegrader:viegrader /opt/viegrader
sudo mkdir -p /opt/viegrader/{data,artifacts,reports,logs,cache}
sudo chown -R viegrader:viegrader /opt/viegrader
sudo cp deploy/viegrader-gui.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now viegrader-gui
sudo systemctl status viegrader-gui
journalctl -u viegrader-gui -f
```

Sao chép `deploy/nginx-viegrader.conf` vào `/etc/nginx/sites-available/viegrader`,
sửa tên miền, tạo symlink sang `sites-enabled`, kiểm tra bằng `sudo nginx -t`
rồi reload Nginx. Khi công khai Internet, bắt buộc dùng HTTPS, firewall, đăng
nhập và không mở trực tiếp cổng Gradio.

## 4. Cấu hình QLoRA cho 16 GB VRAM

Preset mặc định: batch 1, sequence 1024, gradient accumulation 16, NF4 double
quantization, BF16 tự động, gradient checkpointing, `paged_adamw_8bit`, SDPA và
giới hạn tiến trình ở 92% VRAM. Bắt đầu bằng dữ liệu nhỏ trước:

```bash
viegrader train-qlora -i data/processed/clean.csv -g data/gold.csv \
  -r rubrics/bai_kiem_tra_mon_hoc.yaml -o artifacts/vistral-adapter \
  --model Viet-Mistral/Vistral-7B-Chat --max-length 1024 \
  --batch-size 1 --grad-accum 16
```

Theo dõi GPU ở terminal khác:

```bash
watch -n 1 nvidia-smi
```

Nếu hết VRAM: đóng tiến trình GPU khác; giảm `max-length` lần lượt 1024 → 768
→ 512; giữ batch size 1; giảm độ dài sinh khi suy luận. Không giảm gradient
accumulation để chữa OOM vì tham số này chủ yếu thay đổi batch hiệu dụng.

## 5. Baseline, RAG, ablation và báo cáo

Giữ nguyên TF-IDF và PhoBERT làm baseline. Chạy từng cấu hình ablation A–H trên
cùng split/seed, sau đó xuất báo cáo giai đoạn trong GUI hoặc CLI. Không dùng kết
quả mô phỏng làm kết quả nghiên cứu.

```bash
viegrader train --help
viegrader report --help
viegrader gui --host 127.0.0.1 --port 7860
```

Ablation A–H được chạy trong tab nghiên cứu của GUI hoặc qua API Python trong
`viegrader.ablation`; lệnh `report` gom artifact của từng giai đoạn để xuất báo cáo.

Tên lệnh chính xác và trường dữ liệu được hiển thị bằng `--help`; kiểm tra rubric
trước mọi lần huấn luyện bằng `viegrader check-rubric ...`.

## 6. Docker (tuỳ chọn)

Cài Docker Engine và NVIDIA Container Toolkit, rồi cấu hình runtime NVIDIA theo
tài liệu chính thức. Sau đó:

```bash
cp deploy/viegrader.env.example deploy/viegrader.env
docker compose -f compose.ubuntu.yml build
docker compose -f compose.ubuntu.yml up -d
docker compose -f compose.ubuntu.yml logs -f
```

Kiểm tra GPU trong container:

```bash
docker compose -f compose.ubuntu.yml exec viegrader nvidia-smi
docker compose -f compose.ubuntu.yml exec viegrader viegrader hardware-check --strict-blackwell
```

## 7. Xử lý lỗi thường gặp

- `no kernel image ... sm_120`: đang dùng wheel Torch cũ; cài lại Torch >=2.7
  từ index `cu128`, rồi chạy `hardware-check`.
- `torch.cuda.is_available() == False`: kiểm tra `nvidia-smi`, driver, Secure Boot
  và quyền GPU trong container.
- Lỗi bitsandbytes: kiểm tra phiên bản >=0.48, Torch cu128 và không trộn nhiều
  CUDA runtime trong cùng venv.
- Hugging Face 401/403: chấp nhận điều khoản model, tạo token chỉ có quyền cần
  thiết và nạp qua biến `HF_TOKEN`.
- GUI không mở: kiểm tra `systemctl status`, `journalctl`, cổng 7860 và Nginx.
- LibreOffice/PDF lỗi font: bảo đảm đã cài `fonts-dejavu` và font tiếng Việt phù hợp.

## 8. Kiểm tra trước nghiệm thu

```bash
source /opt/viegrader/app/.venv/bin/activate
python scripts/check_ubuntu_gpu.py
pytest -q
viegrader hardware-check --strict-blackwell --json reports/hardware.json
```

Lưu lại đầu ra kiểm tra GPU, phiên bản package, cấu hình huấn luyện, seed, split,
metric và báo cáo từng giai đoạn để có thể tái lập thí nghiệm.

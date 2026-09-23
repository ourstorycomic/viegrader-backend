# VieGrader 0.5.0 — Ubuntu Server và RTX 5060 Ti 16 GB

- Thêm kiểm tra GPU Blackwell `sm_120`, VRAM, BF16, CUDA runtime và phiên bản Torch.
- Thêm preset QLoRA 16 GB: NF4, checkpointing, SDPA, paged AdamW 8-bit, batch 1.
- Giảm mặc định sequence từ 2048 xuống 1024 và giới hạn bộ nhớ CUDA 92%.
- Thêm cấu hình Ubuntu, lệnh `hardware-check` và `ubuntu-gui`.
- Thêm script cài native, systemd, Nginx, Docker Compose và CUDA 12.8 image.
- Giữ đầy đủ baseline TF-IDF/PhoBERT, RAG, ablation A–H, BERTScore, SUS,
  Moodle, GUI/Hugging Face và báo cáo các giai đoạn từ bản 0.4.
- Nâng yêu cầu QLoRA lên PyTorch >=2.7 và bitsandbytes >=0.48.

GPU thật vẫn cần được kiểm tra bằng `viegrader hardware-check --strict-blackwell`
trên máy đích trước khi huấn luyện dài hạn.

#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys


def main() -> int:
    print("OS:", platform.platform())
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT,
        ).strip()
        print("nvidia-smi:", output)
    except Exception as exc:
        print("LỖI nvidia-smi:", exc)
        return 2

    from viegrader.hardware import detect_gpu
    info = detect_gpu(strict_blackwell=True)
    print(info.report())
    try:
        import torch
        a = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
        b = a @ a
        torch.cuda.synchronize()
        print("PyTorch BF16 matmul: OK", tuple(b.shape))
    except Exception as exc:
        print("LỖI kernel PyTorch:", exc)
        return 3
    try:
        import bitsandbytes as bnb
        print("bitsandbytes:", bnb.__version__)
    except Exception as exc:
        print("LỖI bitsandbytes:", exc)
        return 4
    disk = shutil.disk_usage(".")
    print(f"Disk free: {disk.free / 2**30:.1f} GB")
    print(json.dumps(info.to_dict(), ensure_ascii=False, indent=2))
    return 0 if info.ready_for_qlora else 5


if __name__ == "__main__":
    sys.exit(main())

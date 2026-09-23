from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import List, Optional, Tuple


def _version_tuple(value: Optional[str]) -> Tuple[int, ...]:
    if not value:
        return ()
    # Chỉ đọc phần phiên bản ở đầu; tránh biến "2.7.1+cu128" thành 2.7.1128.
    match = re.match(r"^(\d+(?:\.\d+)*)", str(value).strip())
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


@dataclass
class GPUInfo:
    available: bool = False
    name: str = ""
    capability: str = ""
    total_vram_gb: float = 0.0
    torch_version: str = ""
    torch_cuda: str = ""
    cudnn_version: str = ""
    arch_list: List[str] = field(default_factory=list)
    bf16_supported: bool = False
    blackwell: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ready_for_qlora(self) -> bool:
        return self.available and not self.errors and self.total_vram_gb >= 14.0

    def to_dict(self):
        return asdict(self) | {"ready_for_qlora": self.ready_for_qlora}

    def report(self) -> str:
        lines = [
            "===== KIỂM TRA GPU VIEGRADER =====",
            f"GPU                 : {self.name or 'không phát hiện'}",
            f"Compute capability  : {self.capability or '—'}",
            f"VRAM                : {self.total_vram_gb:.2f} GB",
            f"PyTorch             : {self.torch_version or '—'}",
            f"CUDA runtime wheel  : {self.torch_cuda or '—'}",
            f"cuDNN               : {self.cudnn_version or '—'}",
            f"BF16                : {self.bf16_supported}",
            f"Blackwell           : {self.blackwell}",
            f"QLoRA ready         : {self.ready_for_qlora}",
        ]
        if self.arch_list:
            lines.append("Torch arch list     : " + ", ".join(self.arch_list))
        lines.extend("LỖI: " + x for x in self.errors)
        lines.extend("CẢNH BÁO: " + x for x in self.warnings)
        return "\n".join(lines)


def detect_gpu(strict_blackwell: bool = False) -> GPUInfo:
    info = GPUInfo()
    try:
        import torch
    except ImportError:
        info.errors.append("Chưa cài PyTorch.")
        return info
    info.torch_version = str(torch.__version__)
    info.torch_cuda = str(torch.version.cuda or "")
    info.cudnn_version = str(torch.backends.cudnn.version() or "")
    if not torch.cuda.is_available():
        info.errors.append("torch.cuda.is_available() = False. Kiểm tra driver và wheel cu128.")
        return info
    info.available = True
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    major, minor = torch.cuda.get_device_capability(index)
    info.name = str(props.name)
    info.capability = f"{major}.{minor}"
    info.total_vram_gb = float(props.total_memory / 2**30)
    info.arch_list = list(torch.cuda.get_arch_list())
    info.bf16_supported = bool(torch.cuda.is_bf16_supported())
    info.blackwell = major >= 12 or "RTX 50" in info.name.upper()
    if info.blackwell:
        if _version_tuple(info.torch_version) < (2, 7):
            info.errors.append("Blackwell cần PyTorch >= 2.7.")
        if _version_tuple(info.torch_cuda) < (12, 8):
            info.errors.append("Blackwell cần PyTorch wheel dùng CUDA runtime >= 12.8.")
        if info.arch_list and not any("120" in arch for arch in info.arch_list):
            info.errors.append("Wheel PyTorch không chứa kiến trúc sm_120/compute_120.")
    elif strict_blackwell:
        info.errors.append("Không phát hiện GPU Blackwell/RTX 50 series.")
    if info.total_vram_gb < 14:
        info.errors.append("VRAM khả dụng dưới mức 14 GB yêu cầu cho preset Vistral-7B QLoRA.")
    elif info.total_vram_gb < 15.5:
        info.warnings.append("VRAM thấp hơn danh nghĩa 16 GB; giảm max_seq_length nếu OOM.")
    return info


def configure_torch_for_16gb(memory_fraction: float = 0.92) -> GPUInfo:
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    info = detect_gpu()
    if not info.ready_for_qlora:
        raise RuntimeError(info.report())
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction), 0)
    except (RuntimeError, AttributeError):
        info.warnings.append("Không đặt được memory fraction; tiếp tục với mặc định CUDA.")
    torch.cuda.empty_cache()
    return info

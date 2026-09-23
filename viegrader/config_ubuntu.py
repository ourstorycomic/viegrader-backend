from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class UbuntuServerConfig:
    host: str = "0.0.0.0"
    port: int = 7860
    data_dir: str = "/opt/viegrader/data"
    artifact_dir: str = "/opt/viegrader/artifacts"
    report_dir: str = "/opt/viegrader/reports"
    log_dir: str = "/opt/viegrader/logs"
    allowed_paths: List[str] = field(default_factory=list)
    share: bool = False
    memory_fraction: float = 0.92
    max_seq_length: int = 1024
    batch_size: int = 1
    gradient_accumulation_steps: int = 16

    @classmethod
    def load(cls, path: str | Path) -> "UbuntuServerConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        server, paths, qlora = raw.get("server", {}), raw.get("paths", {}), raw.get("qlora", {})
        obj = cls(
            host=str(server.get("host", "0.0.0.0")), port=int(server.get("port", 7860)),
            share=bool(server.get("share", False)),
            data_dir=str(paths.get("data_dir", "/opt/viegrader/data")),
            artifact_dir=str(paths.get("artifact_dir", "/opt/viegrader/artifacts")),
            report_dir=str(paths.get("report_dir", "/opt/viegrader/reports")),
            log_dir=str(paths.get("log_dir", "/opt/viegrader/logs")),
            allowed_paths=[str(x) for x in server.get("allowed_paths", [])],
            memory_fraction=float(qlora.get("memory_fraction", 0.92)),
            max_seq_length=int(qlora.get("max_seq_length", 1024)),
            batch_size=int(qlora.get("batch_size", 1)),
            gradient_accumulation_steps=int(qlora.get("gradient_accumulation_steps", 16)),
        )
        if not 1 <= obj.port <= 65535:
            raise ValueError("Cổng server không hợp lệ.")
        if not 0.5 <= obj.memory_fraction <= 0.98:
            raise ValueError("memory_fraction phải trong [0.5, 0.98].")
        return obj

    def prepare(self) -> None:
        for value in (self.data_dir, self.artifact_dir, self.report_dir, self.log_dir):
            Path(value).mkdir(parents=True, exist_ok=True)
        self.allowed_paths = self.allowed_paths or [
            self.data_dir, self.artifact_dir, self.report_dir, self.log_dir,
        ]
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("VIEGRADER_DATA_DIR", self.data_dir)
        os.environ.setdefault("VIEGRADER_ARTIFACT_DIR", self.artifact_dir)
        os.environ.setdefault("VIEGRADER_REPORT_DIR", self.report_dir)
        os.environ.setdefault("VIEGRADER_QLORA_MAX_LENGTH", str(self.max_seq_length))
        os.environ.setdefault("VIEGRADER_QLORA_BATCH_SIZE", str(self.batch_size))
        os.environ.setdefault(
            "VIEGRADER_QLORA_GRAD_ACCUM", str(self.gradient_accumulation_steps)
        )
        os.environ.setdefault("VIEGRADER_GPU_MEMORY_FRACTION", str(self.memory_fraction))

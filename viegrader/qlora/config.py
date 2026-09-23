from dataclasses import asdict, dataclass
from typing import Optional
from pathlib import Path
import json


@dataclass
class QLoRAConfig:
    model_name: str = "Viet-Mistral/Vistral-7B-Chat"
    output_dir: str = "artifacts/vistral_qlora"
    max_seq_length: int = 1024
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    use_bf16: Optional[bool] = None
    use_tf32: bool = True
    packing: bool = False
    dataloader_num_workers: int = 2
    memory_fraction: float = 0.92
    attn_implementation: str = "sdpa"
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    @classmethod
    def rtx_5060_ti_16gb(cls, **overrides):
        values = dict(
            max_seq_length=1024, batch_size=1, gradient_accumulation_steps=16,
            lora_r=16, lora_alpha=32, optim="paged_adamw_8bit",
            gradient_checkpointing=True, memory_fraction=0.92,
            attn_implementation="sdpa",
        )
        values.update(overrides)
        return cls(**values)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

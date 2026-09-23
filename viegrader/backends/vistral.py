from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .base import BackendPrediction, ScoringBackend
from ..models.llm_judge import SYSTEM_PROMPT, build_user_prompt
from ..schema import Rubric


@dataclass
class VistralConfig:
    model_name: str = "Viet-Mistral/Vistral-7B-Chat"
    adapter_path: Optional[str] = None
    load_in_4bit: bool = True
    max_new_tokens: int = 768
    max_input_tokens: int = 3072
    temperature: float = 0.0
    device_map: str = "cuda:0"
    torch_dtype: str = "auto"
    memory_fraction: float = 0.92
    attn_implementation: str = "sdpa"


class VistralBackend(ScoringBackend):
    """Suy luận Vistral cục bộ; chỉ tải model khi gọi ``load``/``predict``."""

    name = "vistral"

    def __init__(self, rubric: Rubric, cfg: VistralConfig | None = None):
        self.rubric, self.cfg = rubric, cfg or VistralConfig()
        self.model = self.tokenizer = None
        self.keywords: Dict[str, list[str]] = {}
        self.anchors: list[Dict[str, Any]] = []

    @property
    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> "VistralBackend":
        if self.model is not None:
            return self
        if not self.available:
            raise ImportError("Cài viegrader[qlora] để dùng VistralBackend.")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from ..hardware import configure_torch_for_16gb

        gpu = configure_torch_for_16gb(self.cfg.memory_fraction)
        dtype = torch.bfloat16 if gpu.bf16_supported else torch.float16

        quant = None
        if self.cfg.load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device_map = {"": 0} if str(self.cfg.device_map).startswith("cuda") else self.cfg.device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name, device_map=device_map,
            quantization_config=quant,
            torch_dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation=self.cfg.attn_implementation,
        )
        if self.cfg.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.cfg.adapter_path)
        self.model.eval()
        return self

    @staticmethod
    def _json(text: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _generate(self, system: str, user: str) -> str:
        import torch
        self.load()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            truncation=True, max_length=self.cfg.max_input_tokens,
        ).to(self.model.device)
        generation = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.cfg.temperature > 0:
            generation["temperature"] = self.cfg.temperature
        with torch.inference_mode():
            output = self.model.generate(encoded, **generation)
        return self.tokenizer.decode(output[0, encoded.shape[-1]:], skip_special_tokens=True)

    def predict(self, essays: pd.DataFrame) -> BackendPrediction:
        rows, conf, comments, flags, evidence = [], [], [], [], []
        for _, row in essays.iterrows():
            pid = str(row.get("prompt_id", ""))
            prompt = build_user_prompt(
                str(row.get("text", "")), self.rubric,
                str(row.get("prompt_text", "")),
                str(row.get("rag_context", "; ".join(self.keywords.get(pid, [])))),
                self.anchors,
            )
            data = self._json(self._generate(SYSTEM_PROMPT, prompt)) or {}
            criteria = data.get("criteria", {})
            rows.append({k: v.get("score") for k, v in criteria.items() if isinstance(v, dict)})
            evidence.append({k: list(v.get("evidence", [])) for k, v in criteria.items() if isinstance(v, dict)})
            conf.append(float(data.get("confidence", 0.5)))
            comments.append(str(data.get("overall_comment", "")))
            flags.append(list(data.get("flags", [])))
        return BackendPrediction(pd.DataFrame(rows, index=essays.index), conf, comments, flags, evidence,
                                 {"backend": self.name, "model": self.cfg.model_name})

    def close(self) -> None:
        self.model = self.tokenizer = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

"""Huấn luyện QLoRA với loss chỉ trên câu trả lời của assistant.

Thiết kế này khắc phục hai lỗi nghiêm trọng của pipeline SFT cũ:

1. Cắt phải toàn bộ hội thoại có thể làm mất một phần hoặc toàn bộ JSON đích.
2. Tính loss trên cả đề/rubric/bài làm khiến loss thấp nhưng mô hình học tiếp
   tục lời giải thay vì học nhiệm vụ chấm điểm.

Mỗi mẫu được đóng gói sao cho system prompt và toàn bộ assistant target luôn
được giữ. Khi quá dài, chỉ nội dung user bị rút gọn ở giữa, nhờ đó vẫn giữ phần
đầu (đề/rubric) và phần cuối (bài làm/yêu cầu JSON).
"""

from __future__ import annotations

import inspect
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence

from .config import QLoRAConfig
from ..artifacts import ArtifactManifest
from ..hardware import configure_torch_for_16gb


IGNORE_INDEX = -100
TRUNCATION_MARKER = "\n\n[... NỘI DUNG ĐÃ ĐƯỢC RÚT GỌN THEO GIỚI HẠN TOKEN ...]\n\n"


def _supported_kwargs(callable_obj: Any, values: Mapping[str, Any]) -> Dict[str, Any]:
    """Chỉ truyền các tham số được phiên bản thư viện hiện tại hỗ trợ."""
    signature = inspect.signature(callable_obj)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return dict(values)
    return {key: value for key, value in values.items() if key in signature.parameters}


def _as_token_ids(value: Any) -> List[int]:
    """Chuẩn hóa đầu ra tokenizer/chat template thành danh sách token ID."""
    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def _chat_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> List[int]:
    result = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    return _as_token_ids(result)


def _middle_truncate_text(tokenizer: Any, text: str, token_budget: int) -> str:
    """Giữ cả đầu và cuối văn bản trong một ngân sách token xác định."""
    if token_budget <= 0:
        return ""
    token_ids = _as_token_ids(
        tokenizer(text, add_special_tokens=False)["input_ids"]
    )
    if len(token_ids) <= token_budget:
        return text

    marker_ids = _as_token_ids(
        tokenizer(TRUNCATION_MARKER, add_special_tokens=False)["input_ids"]
    )
    if token_budget <= len(marker_ids) + 2:
        kept = token_ids[-token_budget:]
        return tokenizer.decode(kept, skip_special_tokens=False).strip()

    available = token_budget - len(marker_ids)
    head_count = max(1, math.ceil(available * 0.55))
    tail_count = max(1, available - head_count)
    kept_ids = token_ids[:head_count] + marker_ids + token_ids[-tail_count:]
    return tokenizer.decode(
        kept_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()


def _render_prompt_ids(tokenizer: Any, system: str, user: str) -> List[int]:
    return _chat_ids(
        tokenizer,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        add_generation_prompt=True,
    )


def _assistant_suffix_ids(
    tokenizer: Any,
    system: str,
    user: str,
    assistant: str,
) -> tuple[List[int], List[int]]:
    """Trả về prompt IDs và suffix assistant, kể cả token kết thúc lượt."""
    prompt_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": assistant},
    ]
    prompt_ids = _chat_ids(tokenizer, prompt_messages, add_generation_prompt=True)
    full_ids = _chat_ids(tokenizer, full_messages, add_generation_prompt=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "Chat template không có cấu trúc prefix ổn định; không thể tạo "
            "completion-only labels một cách an toàn."
        )
    assistant_ids = full_ids[len(prompt_ids) :]
    if not assistant_ids:
        raise ValueError("Assistant target không sinh ra token nào.")
    return prompt_ids, assistant_ids


def _fit_record(
    tokenizer: Any,
    record: Mapping[str, str],
    max_length: int,
) -> tuple[Dict[str, List[int]], Dict[str, Any]]:
    """Đóng gói một record, luôn giữ system và toàn bộ assistant target."""
    system = str(record.get("system", "")).strip()
    user = str(record.get("user", "")).strip()
    assistant = str(record.get("assistant", "")).strip()
    if not system or not user or not assistant:
        raise ValueError("Mỗi record phải có system, user và assistant khác rỗng.")

    original_prompt_ids, original_assistant_ids = _assistant_suffix_ids(
        tokenizer, system, user, assistant
    )
    original_full_length = len(original_prompt_ids) + len(original_assistant_ids)
    prompt_budget = max_length - len(original_assistant_ids)
    if prompt_budget <= 0:
        raise ValueError(
            f"Assistant target có {len(original_assistant_ids)} token, vượt "
            f"max_length={max_length}."
        )

    minimum_user = _middle_truncate_text(tokenizer, user, 1)
    minimum_prompt_ids = _render_prompt_ids(tokenizer, system, minimum_user)
    if len(minimum_prompt_ids) > prompt_budget:
        raise ValueError(
            "System/chat template quá dài nên không còn chỗ cho assistant "
            f"target trong max_length={max_length}."
        )

    fitted_user = user
    fitted_prompt_ids = original_prompt_ids
    was_truncated = len(fitted_prompt_ids) > prompt_budget

    if was_truncated:
        user_ids = _as_token_ids(
            tokenizer(user, add_special_tokens=False)["input_ids"]
        )
        low, high = 1, len(user_ids)
        best_user = minimum_user
        best_prompt_ids = minimum_prompt_ids

        while low <= high:
            middle = (low + high) // 2
            candidate_user = _middle_truncate_text(tokenizer, user, middle)
            candidate_prompt_ids = _render_prompt_ids(
                tokenizer, system, candidate_user
            )
            if len(candidate_prompt_ids) <= prompt_budget:
                best_user = candidate_user
                best_prompt_ids = candidate_prompt_ids
                low = middle + 1
            else:
                high = middle - 1

        fitted_user = best_user
        fitted_prompt_ids, fitted_assistant_ids = _assistant_suffix_ids(
            tokenizer, system, fitted_user, assistant
        )
    else:
        fitted_assistant_ids = original_assistant_ids

    input_ids = fitted_prompt_ids + fitted_assistant_ids
    if len(input_ids) > max_length:
        raise AssertionError(
            f"Lỗi đóng gói: {len(input_ids)} token > max_length={max_length}."
        )
    labels = [IGNORE_INDEX] * len(fitted_prompt_ids) + list(fitted_assistant_ids)
    supervised_tokens = sum(label != IGNORE_INDEX for label in labels)
    if supervised_tokens != len(fitted_assistant_ids) or supervised_tokens == 0:
        raise AssertionError("Completion-only labels không hợp lệ.")

    features = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
    audit = {
        "original_prompt_tokens": len(original_prompt_ids),
        "original_full_tokens": original_full_length,
        "packed_prompt_tokens": len(fitted_prompt_ids),
        "packed_full_tokens": len(input_ids),
        "assistant_tokens": len(fitted_assistant_ids),
        "supervised_tokens": supervised_tokens,
        "prompt_truncated": was_truncated,
        "assistant_complete": True,
    }
    return features, audit


def _build_tokenized_dataset(
    tokenizer: Any,
    records: Sequence[Mapping[str, str]],
    max_length: int,
) -> tuple[List[Dict[str, List[int]]], Dict[str, Any]]:
    if max_length < 256:
        raise ValueError("max_seq_length phải từ 256 trở lên.")
    features: List[Dict[str, List[int]]] = []
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for index, record in enumerate(records):
        try:
            item, audit = _fit_record(tokenizer, record, max_length)
            features.append(item)
            rows.append({"index": index, **audit})
        except Exception as exc:
            errors.append({"index": index, "error": str(exc)})

    if errors:
        preview = "; ".join(
            f"record {item['index']}: {item['error']}" for item in errors[:5]
        )
        raise ValueError(
            f"Không thể token hóa an toàn {len(errors)}/{len(records)} record. "
            f"{preview}"
        )
    if not features:
        raise ValueError("Không có record hợp lệ sau khi token hóa.")

    truncated = sum(bool(row["prompt_truncated"]) for row in rows)
    assistant_incomplete = sum(not bool(row["assistant_complete"]) for row in rows)
    audit_report = {
        "schema": "viegrader.qlora.tokenization_audit/v2",
        "total_records": len(rows),
        "max_seq_length": max_length,
        "completion_only_loss": True,
        "ignore_index": IGNORE_INDEX,
        "prompt_truncated_records": truncated,
        "prompt_truncated_rate": truncated / len(rows),
        "assistant_complete_records": len(rows) - assistant_incomplete,
        "assistant_incomplete_records": assistant_incomplete,
        "assistant_complete_rate": (len(rows) - assistant_incomplete) / len(rows),
        "original_prompt_tokens": {
            "min": min(row["original_prompt_tokens"] for row in rows),
            "mean": mean(row["original_prompt_tokens"] for row in rows),
            "max": max(row["original_prompt_tokens"] for row in rows),
        },
        "original_full_tokens": {
            "min": min(row["original_full_tokens"] for row in rows),
            "mean": mean(row["original_full_tokens"] for row in rows),
            "max": max(row["original_full_tokens"] for row in rows),
        },
        "packed_full_tokens": {
            "min": min(row["packed_full_tokens"] for row in rows),
            "mean": mean(row["packed_full_tokens"] for row in rows),
            "max": max(row["packed_full_tokens"] for row in rows),
        },
        "assistant_tokens": {
            "min": min(row["assistant_tokens"] for row in rows),
            "mean": mean(row["assistant_tokens"] for row in rows),
            "max": max(row["assistant_tokens"] for row in rows),
        },
        "supervised_tokens": {
            "min": min(row["supervised_tokens"] for row in rows),
            "mean": mean(row["supervised_tokens"] for row in rows),
            "max": max(row["supervised_tokens"] for row in rows),
        },
        "validation": {
            "lost_assistant_targets": 0,
            "partial_assistant_targets": 0,
            "all_assistant_targets_complete": assistant_incomplete == 0,
            "all_records_have_supervised_tokens": all(
                row["supervised_tokens"] > 0 for row in rows
            ),
        },
    }
    return features, audit_report


@dataclass
class _CompletionOnlyCollator:
    """Padding động, đồng thời loại token padding khỏi hàm loss."""

    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> Dict[str, Any]:
        import torch

        longest = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of > 1:
            longest = int(
                math.ceil(longest / self.pad_to_multiple_of) * self.pad_to_multiple_of
            )
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer chưa có pad_token_id.")

        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []
        for feature in features:
            size = len(feature["input_ids"])
            padding = longest - size
            batch_input_ids.append(list(feature["input_ids"]) + [pad_token_id] * padding)
            batch_attention_mask.append(list(feature["attention_mask"]) + [0] * padding)
            batch_labels.append(list(feature["labels"]) + [IGNORE_INDEX] * padding)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def train_qlora(
    records: Sequence[Dict[str, str]],
    cfg: QLoRAConfig | None = None,
):
    """Huấn luyện QLoRA completion-only, tối ưu cho RTX 5060 Ti 16 GB."""
    cfg = cfg or QLoRAConfig()
    if not records:
        raise ValueError("Không có instruction record để huấn luyện.")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError("Cài viegrader[qlora] trước khi huấn luyện QLoRA.") from exc

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu = configure_torch_for_16gb(cfg.memory_fraction)
    use_bf16 = gpu.bf16_supported if cfg.use_bf16 is None else bool(cfg.use_bf16)
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer không cung cấp pad_token hoặc eos_token hợp lệ.")
    tokenizer.padding_side = "right"

    tokenized_records, token_audit = _build_tokenized_dataset(
        tokenizer, records, cfg.max_seq_length
    )
    (output_dir / "tokenization_audit.json").write_text(
        json.dumps(token_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not token_audit["validation"]["all_assistant_targets_complete"]:
        raise RuntimeError("Có assistant target bị cắt; dừng huấn luyện để bảo vệ dữ liệu.")

    dataset = Dataset.from_list(tokenized_records)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=quantization,
        device_map="auto",
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=cfg.attn_implementation,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.gradient_checkpointing
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    args_values = {
        "output_dir": str(output_dir),
        "num_train_epochs": cfg.epochs,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "logging_steps": 5,
        "logging_strategy": "steps",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "report_to": "none",
        "seed": cfg.seed,
        "data_seed": cfg.seed,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "tf32": cfg.use_tf32,
        "optim": cfg.optim,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "max_grad_norm": cfg.max_grad_norm,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "remove_unused_columns": False,
    }
    training_args = TrainingArguments(
        **_supported_kwargs(TrainingArguments.__init__, args_values)
    )
    trainer_values: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "data_collator": _CompletionOnlyCollator(tokenizer),
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_values["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_values["tokenizer"] = tokenizer

    trainer = Trainer(**_supported_kwargs(Trainer.__init__, trainer_values))
    
    resume_path = os.environ.get("VIEGRADER_RESUME_CHECKPOINT")
    if resume_path and Path(resume_path).is_dir():
        print(f"RESUMING TRAINING FROM: {resume_path}")
        train_output = trainer.train(resume_from_checkpoint=resume_path)
    else:
        train_output = trainer.train()
        
    trainer.save_model(str(output_dir))

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    cfg.save(str(output_dir / "training_config.json"))
    metrics = {
        key: float(value) if isinstance(value, (int, float)) else str(value)
        for key, value in train_output.metrics.items()
    }
    (output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ArtifactManifest(
        model_type="qlora",
        base_model=cfg.model_name,
        adapter_path=str(output_dir),
        metadata={
            "n_records": len(records),
            "seed": cfg.seed,
            "gpu": gpu.to_dict(),
            "train_metrics": metrics,
            "tokenization_audit": token_audit,
            "completion_only_loss": True,
        },
    ).save(output_dir / "manifest.json")
    trainer.model.config.use_cache = True
    return trainer

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class HuggingFaceConfig:
    repo_id: str
    repo_type: str = "model"
    private: bool = True
    token_env: str = "HF_TOKEN"
    revision: str = "main"

    def token(self) -> str:
        value = os.environ.get(self.token_env, "").strip()
        if not value:
            raise ValueError(
                f"Chưa khai báo {self.token_env}. Hãy lưu token trong Secrets/biến môi trường."
            )
        return value


class HuggingFaceService:
    """Đăng nhập và đồng bộ artifact với Hub mà không ghi token vào file/log."""

    def __init__(self, cfg: HuggingFaceConfig):
        self.cfg = cfg

    def _api(self):
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError("Cài viegrader[gui] hoặc huggingface_hub.") from exc
        return HfApi(token=self.cfg.token())

    def whoami(self) -> Dict[str, Any]:
        data = self._api().whoami()
        return {
            "name": data.get("name", ""),
            "fullname": data.get("fullname", ""),
            "type": data.get("type", "user"),
        }

    def ensure_repo(self) -> str:
        if self.cfg.repo_type not in {"model", "dataset", "space"}:
            raise ValueError("repo_type phải là model, dataset hoặc space.")
        kwargs = dict(
            repo_id=self.cfg.repo_id, repo_type=self.cfg.repo_type,
            private=self.cfg.private, exist_ok=True,
        )
        if self.cfg.repo_type == "space":
            kwargs["space_sdk"] = "gradio"
        result = self._api().create_repo(**kwargs)
        return str(result)

    def upload_folder(
        self, folder: str | Path, commit_message: str,
        ignore_patterns: Optional[Iterable[str]] = None,
    ) -> str:
        path = Path(folder)
        if not path.is_dir():
            raise FileNotFoundError(path)
        self.ensure_repo()
        result = self._api().upload_folder(
            folder_path=str(path), repo_id=self.cfg.repo_id,
            repo_type=self.cfg.repo_type, revision=self.cfg.revision,
            commit_message=commit_message,
            ignore_patterns=list(ignore_patterns or [
                "checkpoint-*", "**/optimizer.pt", "**/scheduler.pt", "**/rng_state.pth",
                "**/__pycache__/**", "*.pyc",
            ]),
        )
        return str(result)

    def upload_file(self, path: str | Path, path_in_repo: str, commit_message: str) -> str:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        self.ensure_repo()
        result = self._api().upload_file(
            path_or_fileobj=str(file_path), path_in_repo=path_in_repo,
            repo_id=self.cfg.repo_id, repo_type=self.cfg.repo_type,
            revision=self.cfg.revision, commit_message=commit_message,
        )
        return str(result)

    def push_adapter(self, adapter_dir: str | Path) -> str:
        required = ["adapter_config.json", "training_config.json", "manifest.json"]
        missing = [name for name in required if not (Path(adapter_dir) / name).exists()]
        if missing:
            raise ValueError(f"Thư mục adapter thiếu: {missing}")
        return self.upload_folder(adapter_dir, "Upload VieGrader QLoRA adapter")

    def push_dataset_records(self, records, output_path: str | Path) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        dataset_cfg = HuggingFaceConfig(
            repo_id=self.cfg.repo_id, repo_type="dataset", private=self.cfg.private,
            token_env=self.cfg.token_env, revision=self.cfg.revision,
        )
        return HuggingFaceService(dataset_cfg).upload_file(
            path, path.name, "Upload anonymized VieGrader instruction dataset"
        )


def build_model_card(
    output_dir: str | Path, repo_id: str, base_model: str, rubric_name: str,
    metrics: Optional[Dict[str, float]] = None,
) -> Path:
    metric_lines = "\n".join(f"- {k}: {v:.4f}" for k, v in (metrics or {}).items())
    card = f"""---
language:
- vi
library_name: peft
base_model: {base_model}
pipeline_tag: text-generation
tags:
- automated-essay-scoring
- qlora
- vietnamese
---

# {repo_id}

Adapter QLoRA cho VieGrader, chấm bài tự luận tiếng Việt theo rubric.

- Base model: `{base_model}`
- Rubric: {rubric_name}
- Đầu ra: JSON gồm điểm theo tiêu chí, bằng chứng và phản hồi.

## Chỉ số đánh giá

{metric_lines or 'Chưa cập nhật. Không sử dụng kết quả mô phỏng làm kết quả nghiên cứu.'}

## Giới hạn sử dụng

Hệ thống hỗ trợ giảng viên, không tự động thay thế quyết định chấm. Bài có độ
tin cậy thấp, cờ liêm chính hoặc điểm biên phải được giảng viên phúc tra.
"""
    path = Path(output_dir) / "README.md"
    path.write_text(card, encoding="utf-8")
    return path

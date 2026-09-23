import os

import pandas as pd
import pytest

from viegrader.gui.handlers import ablation_gui, inspect_environment
from viegrader.huggingface.hub import build_model_card, HuggingFaceConfig, HuggingFaceService


def test_hf_config_requires_secret(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError):
        HuggingFaceConfig("user/model").token()


def test_hf_service_create_and_upload(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token")
    calls = []

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.append(("create", kwargs)); return "repo-url"

        def upload_folder(self, **kwargs):
            calls.append(("upload", kwargs)); return "commit-url"

    service = HuggingFaceService(HuggingFaceConfig("user/model"))
    monkeypatch.setattr(service, "_api", lambda: FakeApi())
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "training_config.json").write_text("{}")
    (tmp_path / "manifest.json").write_text("{}")
    assert service.push_adapter(tmp_path) == "commit-url"
    assert calls[0][0] == "create" and calls[1][0] == "upload"


def test_model_card(tmp_path):
    path = build_model_card(tmp_path, "user/model", "base/model", "Rubric A", {"QWK": 0.8})
    text = path.read_text(encoding="utf-8")
    assert "base/model" in text and "QWK: 0.8000" in text


def test_ablation_gui_export():
    frame, path = ablation_gui()
    assert list(frame.experiment_id) == list("ABCDEFGH")
    assert path.endswith(".csv")


def test_environment_report(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert "HF_TOKEN: chưa cấu hình" in inspect_environment()

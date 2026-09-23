import json
import os

from viegrader.config_ubuntu import UbuntuServerConfig
from viegrader.hardware import GPUInfo, _version_tuple
from viegrader.qlora.config import QLoRAConfig
from viegrader.cli import build_parser


def test_version_tuple():
    assert _version_tuple("2.7.1+cu128") == (2, 7, 1)
    assert _version_tuple(None) == ()


def test_rtx_5060_ti_preset():
    cfg = QLoRAConfig.rtx_5060_ti_16gb(output_dir="adapter")
    assert cfg.max_seq_length == 1024
    assert cfg.batch_size == 1
    assert cfg.gradient_accumulation_steps == 16
    assert cfg.optim == "paged_adamw_8bit"
    assert cfg.output_dir == "adapter"


def test_gpu_info_ready():
    good = GPUInfo(True, "RTX 5060 Ti", "12.0", 15.9, "2.7.1", "12.8", "91002",
                   ["sm_120"], True, True)
    assert good.ready_for_qlora
    assert good.to_dict()["blackwell"] is True
    bad = GPUInfo(False, errors=["Không có CUDA"])
    assert not bad.ready_for_qlora


def test_ubuntu_config_prepare(tmp_path):
    raw = {
        "server": {"host": "127.0.0.1", "port": 7861},
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "artifact_dir": str(tmp_path / "models"),
            "report_dir": str(tmp_path / "reports"),
            "log_dir": str(tmp_path / "logs"),
        },
        "qlora": {"memory_fraction": 0.9},
    }
    path = tmp_path / "ubuntu.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    cfg = UbuntuServerConfig.load(path)
    cfg.prepare()
    assert cfg.port == 7861
    assert (tmp_path / "models").is_dir()
    assert cfg.memory_fraction == 0.9
    assert os.environ["VIEGRADER_QLORA_MAX_LENGTH"] == "1024"


def test_cmd_parsers():
    parser = build_parser()
    assert parser.parse_args(["ablation-plan"]).cmd == "ablation-plan"
    assert parser.parse_args([
        "bertscore", "--generated", "a.csv", "--references", "b.csv"
    ]).generated_column == "feedback"
    moodle = parser.parse_args([
        "moodle", "push-grade", "--assignment-id", "1", "--user-id", "2",
        "--grade", "8.5", "--approved",
    ])
    assert moodle.approved is True

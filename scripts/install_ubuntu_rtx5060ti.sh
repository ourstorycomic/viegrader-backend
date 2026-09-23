#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${VIEGRADER_PYTHON:-python3}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "LỖI: chưa có NVIDIA driver hoặc nvidia-smi. Xem docs/CAI_DAT_UBUNTU_RTX5060TI.md."
  exit 2
fi

driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
driver_major="${driver_version%%.*}"
if [[ ! "$driver_major" =~ ^[0-9]+$ ]] || (( driver_major < 570 )); then
  echo "LỖI: driver $driver_version quá cũ cho Blackwell. Yêu cầu driver Linux >= 570.26."
  exit 3
fi

if [[ "${VIEGRADER_SKIP_APT:-0}" != "1" ]]; then
  sudo apt-get update
  sudo apt-get install -y \
    python3 python3-venv python3-dev build-essential git curl unzip \
    libgl1 libglib2.0-0 libreoffice poppler-utils fonts-dejavu-core nginx
fi

"$python_bin" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("VieGrader Ubuntu yêu cầu Python >= 3.10")
print("Python:", sys.version)
PY

"$python_bin" -m venv "$project_dir/.venv"
source "$project_dir/.venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# Blackwell/sm_120: luôn cài wheel chính thức có CUDA runtime 12.8.
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$project_dir/requirements-ubuntu-cu128.txt"
python -m pip install -e "$project_dir[gui,reports,qlora,evaluation,docs,api,moodle,hub]"

mkdir -p "$project_dir/data" "$project_dir/artifacts" "$project_dir/reports" "$project_dir/logs"
python "$project_dir/scripts/check_ubuntu_gpu.py"
python -m pytest -q "$project_dir/tests"

echo "Cài đặt hoàn tất."
echo "Kích hoạt: source $project_dir/.venv/bin/activate"
echo "Thiết lập CMD: cp $project_dir/config.cmd.example.env $project_dir/config.cmd.env"
echo "Trợ giúp: $project_dir/scripts/viegrader_cmd.sh help"

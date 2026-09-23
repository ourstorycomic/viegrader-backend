#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${1:-$project_dir/data/external/vietnamese-it-essays}"
dataset_slug="vokhoa/vietnamese-it-essays-for-aes-research"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Thiếu Kaggle CLI. Cài bằng: python -m pip install kaggle" >&2
  exit 2
fi
if [[ ! -f "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" && -z "${KAGGLE_KEY:-}" ]]; then
  echo "Thiếu thông tin xác thực Kaggle. Đặt kaggle.json trong ~/.kaggle hoặc khai báo biến KAGGLE_USERNAME/KAGGLE_KEY." >&2
  exit 2
fi

mkdir -p "$destination"
kaggle datasets download -d "$dataset_slug" -p "$destination" --unzip
viegrader prepare-dataset --source-dir "$destination" \
  -o "$project_dir/data/imported/vietnamese_it_aes"

echo "Đã tải và chuẩn hóa nguồn $dataset_slug"
echo "Đọc data/imported/vietnamese_it_aes/dataset_audit.json trước khi gán nhãn."

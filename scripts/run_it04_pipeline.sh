#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_file="${VIEGRADER_IT04_CONFIG:-$project_dir/config.it04.env}"
if [[ ! -f "$config_file" ]]; then
  echo "Thiếu $config_file. Hãy: cp config.it04.example.env config.it04.env"
  exit 2
fi
set -a
# shellcheck disable=SC1090
source "$config_file"
set +a

cd "${VIEGRADER_ROOT:-$project_dir}"
command_name="${1:-help}"
run_dir="${IT04_RUN:?Cần đặt IT04_RUN}"
rubric_dir="${IT04_RUBRICS:?Cần đặt IT04_RUBRICS}"

case "$command_name" in
  check)
    python -m viegrader.cli it04-validate-rubric --rubric-dir "$rubric_dir" \
      -o "$run_dir/02_rubric"
    ;;
  prepare)
    python -m viegrader.cli it04-prepare --source "${IT04_DATASET:?Cần đặt IT04_DATASET}" \
      --rubric-dir "$rubric_dir" -o "$run_dir/01_data"
    ;;
  train)
    python -m viegrader.cli it04-train --prepared "$run_dir/01_data" \
      -o "$run_dir/03_model" --alpha "${IT04_ALPHA:-12.0}"
    ;;
  grade)
    python -m viegrader.cli it04-grade -i "$run_dir/01_data/essays_split.csv" \
      -m "$run_dir/03_model/total_baseline.pkl" --rubric-dir "$rubric_dir" \
      -o "$run_dir/04_grading" --rubric-weight "${IT04_RUBRIC_WEIGHT:-0.70}"
    ;;
  evaluate)
    python -m viegrader.cli it04-evaluate --scores "$run_dir/04_grading/scores.csv" \
      --gold "$run_dir/01_data/gold_split.csv" -o "$run_dir/05_evaluation"
    ;;
  report)
    python -m viegrader.cli it04-report --root "$run_dir" -o "$run_dir/06_reports"
    ;;
  all)
    python -m viegrader.cli it04-run-all --source "${IT04_DATASET:?Cần đặt IT04_DATASET}" \
      --rubric-dir "$rubric_dir" -o "$run_dir" --alpha "${IT04_ALPHA:-12.0}" \
      --rubric-weight "${IT04_RUBRIC_WEIGHT:-0.70}"
    ;;
  build-qlora)
    python -m viegrader.cli dm-build-qlora -i "$run_dir/01_data/essays_split.csv" \
      -g "$run_dir/01_data/gold_split.csv" -o "$run_dir/03_model/qlora_train.jsonl"
    ;;
  train-qlora)
    python -m viegrader.cli hardware-check --strict-blackwell \
      --json "$run_dir/03_model/hardware.json"
    python -m viegrader.cli dm-train-qlora --records "$run_dir/03_model/qlora_train.jsonl" \
      -o "$run_dir/03_model/qlora_adapter" --model "${IT04_LLM_MODEL}" \
      --epochs "${IT04_EPOCHS:-3}" --max-length "${IT04_MAX_LENGTH:-2048}" \
      --batch-size 1 --grad-accum 16 --seed "${IT04_SEED:-42}"
    ;;
  help|*)
    printf '%s\n' \
      "Dùng: ./scripts/run_it04_pipeline.sh <lệnh>" \
      "check | prepare | train | grade | evaluate | report | all" \
      "build-qlora | train-qlora (tùy chọn GPU)"
    ;;
esac

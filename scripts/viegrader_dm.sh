#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_file="${VIEGRADER_DM_CONFIG:-$project_dir/config.dm.env}"
if [[ ! -f "$config_file" ]]; then
  echo "Thiếu $config_file. Hãy: cp config.dm.example.env config.dm.env"
  exit 2
fi
set -a
# shellcheck disable=SC1090
source "$config_file"
set +a

cd "${VIEGRADER_ROOT:-$project_dir}"
mkdir -p "$DM_PREPARED" "$DM_SPLITS" "$DM_ARTIFACTS" "$DM_REPORTS"
command_name="${1:-help}"

case "$command_name" in
  check)
    viegrader hardware-check --strict-blackwell --json "$DM_REPORTS/hardware.json"
    for rubric in "$DM_RUBRIC_DIR"/rubric_de_{1..5}.yaml; do
      viegrader check-rubric "$rubric"
    done
    ;;
  prepare)
    : "${VIEGRADER_DM_HMAC_SECRET:?Cần export VIEGRADER_DM_HMAC_SECRET}"
    viegrader dm-prepare --source-dir "$DM_SOURCE_DIR" --gold "$DM_GOLD" \
      --rubric-dir "$DM_RUBRIC_DIR" --catalog "$DM_CATALOG" \
      -o "$DM_PREPARED" --secret "$VIEGRADER_DM_HMAC_SECRET" \
      --conflict-policy quarantine
    ;;
  split)
    viegrader dm-split -i "$DM_PREPARED/essays.csv" -g "$DM_PREPARED/gold_total.csv" \
      -o "$DM_SPLITS" --seed "${DM_SEED:-42}"
    ;;
  baseline)
    viegrader dm-train-baseline -i "$DM_SPLITS/essays_split.csv" \
      -g "$DM_SPLITS/gold_split.csv" -o "$DM_ARTIFACTS/baseline"
    viegrader dm-evaluate -p "$DM_ARTIFACTS/baseline/predictions.csv" \
      -g "$DM_SPLITS/gold_split.csv" -o "$DM_REPORTS/baseline" --split test
    ;;
  build-qlora)
    viegrader dm-build-qlora -i "$DM_SPLITS/essays_split.csv" \
      -g "$DM_SPLITS/gold_split.csv" -o "$DM_SPLITS/qlora_train.jsonl"
    ;;
  qlora)
    viegrader hardware-check --strict-blackwell --json "$DM_REPORTS/hardware.json"
    viegrader dm-train-qlora --records "$DM_SPLITS/qlora_train.jsonl" \
      -o "$DM_ARTIFACTS/qlora_total" --model "$DM_MODEL" \
      --epochs "${DM_EPOCHS:-3}" --max-length "${DM_MAX_LENGTH:-2048}" \
      --batch-size 1 --grad-accum 16 --seed "${DM_SEED:-42}"
    ;;
  score-qlora)
    viegrader dm-score-qlora -i "$DM_SPLITS/essays_split.csv" \
      --adapter "$DM_ARTIFACTS/qlora_total" --model "$DM_MODEL" \
      -o "$DM_REPORTS/pred_qlora_total.csv" --split test \
      --runs "${DM_RUNS:-3}" --temperature 0.1
    ;;
  evaluate-qlora)
    viegrader dm-evaluate -p "$DM_REPORTS/pred_qlora_total.csv" \
      -g "$DM_SPLITS/gold_split.csv" -o "$DM_REPORTS/qlora" --split test
    ;;
  cpu-all)
    "$0" prepare
    "$0" split
    "$0" baseline
    "$0" build-qlora
    ;;
  help|*)
    printf '%s\n' \
      "Dùng: ./scripts/viegrader_dm.sh <lệnh>" \
      "check | prepare | split | baseline | build-qlora" \
      "qlora | score-qlora | evaluate-qlora | cpu-all"
    ;;
esac


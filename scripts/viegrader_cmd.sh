#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_file="${VIEGRADER_CMD_CONFIG:-$project_dir/config.cmd.env}"
runtime_root="${VIEGRADER_ROOT:-}"
if [[ ! -f "$config_file" ]]; then
  echo "Thiếu $config_file. Hãy: cp config.cmd.example.env config.cmd.env"
  exit 2
fi
set -a
# shellcheck disable=SC1090
source "$config_file"
set +a

cd "${runtime_root:-${VIEGRADER_ROOT:-$project_dir}}"
mkdir -p data/processed data/pilot forms artifacts reports models feedback

command_name="${1:-help}"
case "$command_name" in
  dataset-download)
    "$project_dir/scripts/download_selected_dataset.sh" "${DATASET_DOWNLOAD_DIR}"
    ;;
  dataset-prepare)
    viegrader prepare-dataset --source-dir "$DATASET_DOWNLOAD_DIR" -o "$DATASET_IMPORT_DIR"
    ;;
  check)
    viegrader hardware-check --strict-blackwell --json reports/hardware.json
    viegrader check-rubric "$RUBRIC"
    ;;
  clean)
    : "${VIEGRADER_HMAC_SECRET:?Cần export VIEGRADER_HMAC_SECRET}"
    viegrader clean -i "$RAW_INPUT" -o data/processed --secret "$VIEGRADER_HMAC_SECRET"
    ;;
  label)
    viegrader sample -i data/processed/clean.csv -n "${SAMPLE_SIZE:-200}" \
      --seed "${SEED:-42}" -o data/pilot.csv
    viegrader form -r "$RUBRIC" -i data/pilot.csv --raters "$RATERS" \
      --double-rate 0.30 --seed "${SEED:-42}" -o forms
    ;;
  gold)
    viegrader agreement -a "$ANNOTATIONS" -r "$RUBRIC" -o reports/agreement.json
    viegrader adjudicate -a "$ANNOTATIONS" -r "$RUBRIC" -o "$GOLD"
    ;;
  split)
    viegrader split-dataset -i data/processed/clean.csv -g "$GOLD" \
      -o data/splits --group-column student_hash --test-size 0.20 \
      --validation-size 0.10 --seed "${SEED:-42}"
    ;;
  baseline-tfidf)
    viegrader train -i data/processed/clean.csv -g "$GOLD" -r "$RUBRIC" \
      -o models/tfidf.pkl --encoder tfidf --test-size 0.2 --report reports/train_tfidf.txt
    ;;
  baseline-phobert)
    viegrader train -i data/processed/clean.csv -g "$GOLD" -r "$RUBRIC" \
      -o models/phobert.pkl --encoder phobert --test-size 0.2 --report reports/train_phobert.txt
    ;;
  qlora)
    viegrader hardware-check --strict-blackwell --json reports/hardware.json
    viegrader train-qlora -i data/splits/essays_train.csv -g data/splits/gold_train.csv -r "$RUBRIC" \
      -o artifacts/vistral_qlora --model "$BASE_MODEL" --epochs 3 \
      --max-length 1024 --batch-size 1 --grad-accum 16 --seed "${SEED:-42}" \
      --allow-unsplit
    ;;
  score-qlora)
    viegrader hardware-check --strict-blackwell --json reports/hardware.json
    viegrader score-qlora -i data/splits/essays_test.csv -r "$RUBRIC" \
      --adapter artifacts/vistral_qlora --model "$BASE_MODEL" \
      -o reports/pred_qlora.csv --runs "${QLORA_RUNS:-3}" --temperature 0.1
    ;;
  hf-push)
    : "${HF_TOKEN:?Cần export HF_TOKEN}"
    viegrader hf-push --path artifacts/vistral_qlora --repo "$HF_REPO"
    ;;
  rag)
    viegrader rag-index -i "$RAG_SOURCE" -o artifacts/rag_documents.json
    ;;
  score-tfidf)
    viegrader score -i "$NEW_INPUT" -m models/tfidf.pkl -o reports/pred_tfidf.xlsx \
      --feedback-dir feedback/tfidf
    ;;
  evaluate-tfidf)
    viegrader evaluate -p reports/pred_tfidf.xlsx -g "$GOLD_TEST" -r "$RUBRIC" \
      -o reports/evaluate_tfidf.json --qwk-hh "${QWK_HH:-0.70}"
    ;;
  ablation-plan)
    viegrader ablation-plan -o reports/ablation/ablation_plan.csv
    ;;
  ablation-evaluate)
    viegrader ablation-evaluate -g "$GOLD_TEST" --pred-dir reports/ablation \
      -o reports/ablation/ablation_summary.csv
    ;;
  bertscore)
    viegrader bertscore --generated reports/pred_tfidf.xlsx \
      --references "$REFERENCE_FEEDBACK" --generated-column feedback \
      --reference-column feedback -o reports/bertscore.json
    ;;
  sus)
    viegrader sus -i "$SUS_INPUT" -o reports/sus.json
    ;;
  moodle-assignments)
    : "${MOODLE_BASE_URL:?Thiếu MOODLE_BASE_URL}"
    : "${MOODLE_TOKEN:?Thiếu MOODLE_TOKEN}"
    viegrader moodle assignments --course-id "${2:?Cần course_id}" \
      -o reports/moodle_assignments.json
    ;;
  report)
    mapfile -t report_inputs < <(find reports -maxdepth 3 -type f \
      \( -name '*.csv' -o -name '*.xlsx' -o -name '*.json' -o -name '*.txt' \) \
      ! -path 'reports/stage_report/*' | sort)
    if (( ${#report_inputs[@]} == 0 )); then
      echo "Chưa có tệp kết quả trong reports/"; exit 3
    fi
    viegrader report -i "${report_inputs[@]}" -o reports/stage_report \
      --project "VieGrader – NCKH 2026" --formats xlsx,docx,pdf,html,json
    ;;
  api)
    viegrader serve -m "${2:-models/tfidf.pkl}" --host 0.0.0.0 --port 8000
    ;;
  help|*)
    printf '%s\n' \
      "Dùng: scripts/viegrader_cmd.sh <lệnh>" \
      "dataset-download | dataset-prepare | split" \
      "check | clean | label | gold | baseline-tfidf | baseline-phobert" \
      "qlora | score-qlora | hf-push | rag | score-tfidf | evaluate-tfidf" \
      "ablation-plan | ablation-evaluate | bertscore | sus" \
      "moodle-assignments <course_id> | report | api [model.pkl]"
    ;;
esac

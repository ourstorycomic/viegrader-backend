#!/usr/bin/env bash
# Re-evaluate five IT04 conditions on the same eligible test essays.
# Required: VG_ESSAYS, VG_GOLD, VG_ADAPTER, VG_RUN (absolute paths).
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

: "${VG_ESSAYS:?Đặt VG_ESSAYS là essays_split.csv của bộ 623 bài}"
: "${VG_GOLD:?Đặt VG_GOLD là gold_split.csv cùng bộ dữ liệu}"
: "${VG_ADAPTER:?Đặt VG_ADAPTER là thư mục qlora_adapter_v2_2048}"
: "${VG_RUN:?Đặt VG_RUN là thư mục kết quả riêng của thí nghiệm}"

VG_MODEL="${VG_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VG_MAX_TOKENS="${VG_MAX_TOKENS:-4608}"
VG_RUBRICS="${VG_RUBRICS:-$project_dir/data/toan_roi_rac/rubrics}"
VG_INDEX="${VG_INDEX:-$project_dir/data/toan_roi_rac/rag_it04.json}"
for path in "$VG_ESSAYS" "$VG_GOLD" "$VG_ADAPTER/adapter_config.json" "$VG_INDEX"; do
    if [[ ! -f "$path" ]]; then echo "Thiếu đầu vào: $path" >&2; exit 2; fi
done
if [[ "$VG_RUN" != /* ]]; then echo 'VG_RUN phải là đường dẫn tuyệt đối' >&2; exit 2; fi
mkdir -p "$VG_RUN"
python - "$VG_ESSAYS" "$VG_GOLD" <<'PY'
import sys, pandas as pd
essays, gold = (pd.read_csv(p) for p in sys.argv[1:])
for label, frame in (("Essays", essays), ("Gold", gold)):
    counts = frame.split.value_counts().to_dict()
    print(label, len(frame), counts)
    if len(frame) != 623 or counts.get('test') != 73 or counts.get('train') != 472:
        raise SystemExit(f"{label}: không phải bộ 623 bài (472 train, 73 test)")
if set(essays.essay_id.astype(str)) != set(gold.essay_id.astype(str)):
    raise SystemExit("Essay IDs không trùng khớp giữa hai tệp")
PY

if [[ ! -f "$VG_RUN/manifest.json" ]]; then
    python -m viegrader.rag.experiment prepare \
        --essays "$VG_ESSAYS" --index "$VG_INDEX" --output "$VG_RUN" \
        --model "$VG_MODEL" --max-tokens "$VG_MAX_TOKENS" --top-k 3
fi
python - "$VG_RUN" "$VG_ESSAYS" "$VG_INDEX" <<'PY'
import json, sys
from pathlib import Path
from viegrader.rag.experiment import sha
run, essays, index = map(Path, sys.argv[1:])
manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
for name, path in [('input_sha256', essays), ('index_sha256', index),
                   ('plain_sha256', run/'input_plain.csv'), ('rag_sha256', run/'input_rag.csv')]:
    if manifest[name] != sha(path):
        raise SystemExit(f"Đầu vào thay đổi: {path}. Tạo một VG_RUN mới rồi prepare lại.")
print('Tập bài ghép cặp:', manifest['n_paired_eligible'], '/', manifest['n_test'])
PY

if [[ ! -f "$VG_RUN/baseline/total_baseline.pkl" ]]; then
    python -m viegrader.cli dm-train-baseline -i "$VG_ESSAYS" -g "$VG_GOLD" \
        -o "$VG_RUN/baseline"
fi
if [[ ! -s "$VG_RUN/pred_tfidf_ridge.csv" ]]; then
    python -m viegrader.cli dm-score-baseline -i "$VG_RUN/input_plain.csv" \
        -m "$VG_RUN/baseline/total_baseline.pkl" -o "$VG_RUN/pred_tfidf_ridge.csv"
fi
if [[ ! -f "$VG_RUN/rubric_calibration.json" ]]; then
    python - "$VG_RUN" "$VG_ESSAYS" "$VG_GOLD" "$VG_RUBRICS" <<'PY'
import sys
from pathlib import Path
from viegrader.standardized_pipeline import calibrate_rubric_weight
run, essays, gold, rubrics = map(Path, sys.argv[1:])
result = calibrate_rubric_weight(run/'baseline/total_baseline.pkl', essays, gold,
                                  rubrics, run/'rubric_calibration.json')
print('validation hybrid:', result['n_hybrid_eligible'], 'weight:', result['selected_weight'])
PY
fi
weight="$(python - "$VG_RUN/rubric_calibration.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['selected_weight'])
PY
)"
if [[ ! -s "$VG_RUN/hybrid/scores.csv" ]]; then
    python -m viegrader.cli it04-grade -i "$VG_RUN/input_plain.csv" \
        -m "$VG_RUN/baseline/total_baseline.pkl" --rubric-dir "$VG_RUBRICS" \
        --rubric-weight "$weight" -o "$VG_RUN/hybrid"
fi

# GPU: only one Qwen process at a time. Each stage can be resumed.
if [[ ! -s "$VG_RUN/pred_zero_shot.csv" ]]; then
    python -m viegrader.rag.experiment run-zero-shot --output "$VG_RUN" \
        --model "$VG_MODEL" --max-tokens "$VG_MAX_TOKENS" --max-new-tokens 96
fi
if [[ ! -s "$VG_RUN/predictions.csv" ]]; then
    python -m viegrader.rag.experiment run --output "$VG_RUN" --model "$VG_MODEL" \
        --adapter "$VG_ADAPTER" --max-tokens "$VG_MAX_TOKENS" --max-new-tokens 96
fi
python -m viegrader.rag.experiment report --output "$VG_RUN" --gold "$VG_GOLD" \
    --baseline "$VG_RUN/pred_tfidf_ridge.csv" --hybrid "$VG_RUN/hybrid/scores.csv" \
    --bootstrap 1000
echo "Đã tạo báo cáo: $VG_RUN/report.json"

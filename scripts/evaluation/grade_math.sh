#!/usr/bin/env bash
set -euo pipefail

# Grade the output of scripts/data/generate_answers.sh.
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
MODEL_TAG="${MODEL_TAG:-${MODEL_NAME##*/}}"
SPLIT="${SPLIT:-test}"
DOMAIN=math
INPUT_CSV="${INPUT_CSV:-outputs/answers/${DOMAIN}/${MODEL_TAG}/${SPLIT}.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${INPUT_CSV%.csv}_graded.csv}"

python -m src.data.grade_answers \
    --csv_path "${INPUT_CSV}" \
    --eval_col "${EVAL_COL:-generation}" \
    --domain "${DOMAIN}" \
    --mode "${MODE:-multi}" \
    --output_path "${OUTPUT_CSV}"

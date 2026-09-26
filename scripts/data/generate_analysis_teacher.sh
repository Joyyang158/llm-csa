#!/usr/bin/env bash
set -euo pipefail

# Generate both teacher analyses; MODEL_NAME identifies the target model data.
: "${TOGETHER_API_KEY:?Set TOGETHER_API_KEY}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
MODEL_TAG="${MODEL_TAG:-${MODEL_NAME##*/}}"
DOMAIN="${DOMAIN:-math}"
SPLIT="${SPLIT:-train}"
INPUT_CSV="${INPUT_CSV:-outputs/answers/${DOMAIN}/${MODEL_TAG}/${SPLIT}_graded.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/analysis/${DOMAIN}/${MODEL_TAG}/${SPLIT}_teacher.csv}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-tput}"

python -m src.data.generate_analysis_teacher \
    --input_csv "${INPUT_CSV}" \
    --output_csv "${OUTPUT_CSV}" \
    --teacher_model "${TEACHER_MODEL}" \
    --max_tokens "${MAX_TOKENS:-1000}"

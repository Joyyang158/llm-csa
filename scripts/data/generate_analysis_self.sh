#!/usr/bin/env bash
set -euo pipefail

# Generate label-conditioned rationales from the target model.
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to the target base model}"
MODEL_TAG="${MODEL_TAG:-${MODEL_NAME##*/}}"
DOMAIN="${DOMAIN:-math}"
SPLIT="${SPLIT:-train}"
INPUT_CSV="${INPUT_CSV:-outputs/answers/${DOMAIN}/${MODEL_TAG}/${SPLIT}_graded.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/analysis/${DOMAIN}/${MODEL_TAG}/${SPLIT}_self.csv}"

python -m src.data.generate_analysis_self \
    --input_csv "${INPUT_CSV}" \
    --output_csv "${OUTPUT_CSV}" \
    --model_name "${MODEL_NAME}" \
    --model_type "${MODEL_TYPE:-qwen}" \
    --output_col "${OUTPUT_COL:-routing_analysis}" \
    --max_tokens "${MAX_TOKENS:-10000}"

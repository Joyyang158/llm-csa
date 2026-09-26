#!/usr/bin/env bash
set -euo pipefail

# Assemble teacher or self analyses without overriding caller-provided paths.
DOMAIN="${DOMAIN:-math}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
MODEL_TAG="${MODEL_TAG:-${MODEL_NAME##*/}}"
SFT_MODE="${SFT_MODE:-teacher}"
MODEL_CSV="${MODEL_CSV:-outputs/answers/${DOMAIN}/${MODEL_TAG}/train_graded.csv}"
ANALYSIS_CSV="${ANALYSIS_CSV:-outputs/analysis/${DOMAIN}/${MODEL_TAG}/train_${SFT_MODE}.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/sft/${DOMAIN}/${MODEL_TAG}/train_${SFT_MODE}.csv}"

python -m src.data.build_sft_dataset \
    --model_csv "${MODEL_CSV}" \
    --analysis_csv "${ANALYSIS_CSV}" \
    --output_csv "${OUTPUT_CSV}" \
    --mode "${SFT_MODE}" \
    --source_analysis_col "${SOURCE_ANALYSIS_COL:-routing_analysis}" \
    --analysis_col SFT_analysis

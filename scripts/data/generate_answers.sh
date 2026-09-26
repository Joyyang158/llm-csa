#!/usr/bin/env bash
set -euo pipefail

# Generate answers for exactly one explicitly selected base or trained model.
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to the base model or trained checkpoint}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"
MODEL_TAG="${MODEL_TAG:-${MODEL_NAME##*/}}"
DOMAIN="${DOMAIN:-math}"
SPLIT="${SPLIT:-test}"
INPUT_CSV="${INPUT_CSV:-dataset/${DOMAIN}/${SPLIT}.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/answers/${DOMAIN}/${MODEL_TAG}/${SPLIT}.csv}"
OUTPUT_COL="${OUTPUT_COL:-generation}"
NUM_GENERATIONS="${NUM_GENERATIONS:-5}"

case "${DOMAIN}" in
    math) NUM_FLAG=--num_generations ;;
    science) NUM_FLAG=--num_shuffles ;;
    *) echo "DOMAIN must be math or science" >&2; exit 1 ;;
esac
set --
if [ "${ENABLE_THINKING:-0}" = "1" ]; then set -- "$@" --enable_thinking; fi

python -m src.data.generate_answers \
    --domain "${DOMAIN}" \
    --model_name "${MODEL_NAME}" \
    --model_type "${MODEL_TYPE}" \
    --input_csv "${INPUT_CSV}" \
    --output_csv "${OUTPUT_CSV}" \
    --output_col "${OUTPUT_COL}" \
    --max_tokens "${MAX_TOKENS:-30000}" \
    --seed "${SEED:-3407}" \
    "${NUM_FLAG}" "${NUM_GENERATIONS}" \
    "$@"

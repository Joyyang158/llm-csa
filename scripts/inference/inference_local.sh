# Run a trained local CSA model on a CSV of queries (vLLM).
#
# Two modes (selected by NUM_GENERATIONS):
#   1   — single-shot: one decision + one analysis per row.
#   N>1 — rollout: N samples per row, with per-class counts and examples.


DOMAIN="${DOMAIN:-math}"
SPLIT="${SPLIT:-test}"
MODEL_NAME="${MODEL_NAME:-path/to/csa-checkpoint}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"

NUM_GENERATIONS="${NUM_GENERATIONS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-5000}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"

INPUT_CSV="${INPUT_CSV:-data/${DOMAIN}/${SPLIT}.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/csa/${DOMAIN}/${MODEL_NAME##*/}_${SPLIT}.csv}"

EXTRA_FLAGS=""
[ "${ENABLE_THINKING:-0}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --enable_thinking"
[ "${BINARY_ONLY:-0}" = "1" ]     && EXTRA_FLAGS="${EXTRA_FLAGS} --binary_only"

python -m src.csa.inference_local \
    --domain "${DOMAIN}" \
    --model_name "${MODEL_NAME}" \
    --model_type "${MODEL_TYPE}" \
    --input_csv "${INPUT_CSV}" \
    --output_csv "${OUTPUT_CSV}" \
    --num_generations "${NUM_GENERATIONS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    ${EXTRA_FLAGS}

# Diversity-Filtered Warm-up (DFW) subset construction
#
# Samples K rollouts per training query from the initial policy, keeps only
# queries on which both SELF_SOLVE and DELEGATE rollouts appear, and writes
# the resulting subset to ``OUTPUT_CSV``. This subset is the input to Stage 1
# of the two-stage RLVR procedure.

# Optional environment variables you may want to export before running:
#   export HF_HOME=/path/to/hf                      # HuggingFace cache directory
#   export HF_TOKEN=<your_hf_token>                 # for downloading gated HF models

DOMAIN="${DOMAIN:-math}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME=initial-policy-id-or-path}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"

INPUT_CSV="${INPUT_CSV:-dataset/${DOMAIN}/train.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-data/${DOMAIN}/dfw_subset_${MODEL_NAME##*/}.csv}"

NUM_ROLLOUTS="${NUM_ROLLOUTS:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"

python -m src.training.dfw \
    --domain "${DOMAIN}" \
    --input_csv "${INPUT_CSV}" \
    --output_csv "${OUTPUT_CSV}" \
    --model_name "${MODEL_NAME}" \
    --model_type "${MODEL_TYPE}" \
    --num_rollouts "${NUM_ROLLOUTS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}"

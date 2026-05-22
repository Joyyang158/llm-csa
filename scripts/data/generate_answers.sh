# Collect benchmark answers from a local vLLM model across multiple models.
# Edit MODELS / DOMAIN / paths to match your setup.
#
# Override via env vars:
#   DOMAIN=math|science  SPLIT=train|test


DOMAIN="${DOMAIN:-math}"
SPLIT="${SPLIT:-test}"
NUM_GENERATIONS="${NUM_GENERATIONS:-5}"     # math: generations; science: shuffles
MAX_TOKENS="${MAX_TOKENS:-30000}"

MODELS=(
    Qwen/Qwen3-0.6B
    Qwen/Qwen3-1.7B
    Qwen/Qwen3-4B
    Qwen/Qwen3-8B
)

INPUT_CSV="dataset/${DOMAIN}/${SPLIT}.csv"
OUTPUT_TEMPLATE="outputs/model_prediction_on_benchmark/${DOMAIN}/{model_short}/${SPLIT}.csv"

if [ "${DOMAIN}" = "math" ]; then
    NUM_FLAG="--num_generations ${NUM_GENERATIONS}"
else
    NUM_FLAG="--num_shuffles ${NUM_GENERATIONS}"
fi

for model in "${MODELS[@]}"; do
    model_short="${model##*/}"
    output_csv="${OUTPUT_TEMPLATE//\{model_short\}/${model_short}}"

    echo "---------------------------------------------------"
    echo "Generating answers: model=${model}"
    echo "  input : ${INPUT_CSV}"
    echo "  output: ${output_csv}"

    python -m src.data.generate_answers \
        --domain "${DOMAIN}" \
        --model_name "${model}" \
        --input_csv "${INPUT_CSV}" \
        --output_csv "${output_csv}" \
        --output_col benchmark_prediction_vllm \
        --max_tokens "${MAX_TOKENS}" \
        ${NUM_FLAG}
done

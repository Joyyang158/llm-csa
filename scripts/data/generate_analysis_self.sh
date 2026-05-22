# Generate self-written routing analysis via local vLLM.
#
# Override via env vars:
#   DOMAIN=math|science   SPLIT=train|test


DOMAIN="${DOMAIN:-math}"
SPLIT="${SPLIT:-train}"
MAX_TOKENS="${MAX_TOKENS:-10000}"

MODELS=(
    Qwen/Qwen3-0.6B
    Qwen/Qwen3-1.7B
    Qwen/Qwen3-4B
    Qwen/Qwen3-8B
)

INPUT_TEMPLATE="outputs/answers/${DOMAIN}/{model_short}/${SPLIT}_graded.csv"
OUTPUT_TEMPLATE="outputs/analysis/${DOMAIN}/{model_short}/${SPLIT}_self.csv"

for model in "${MODELS[@]}"; do
    model_short="${model##*/}"
    input_csv="${INPUT_TEMPLATE//\{model_short\}/${model_short}}"
    output_csv="${OUTPUT_TEMPLATE//\{model_short\}/${model_short}}"

    echo "---------------------------------------------------"
    echo "Self analysis: model=${model}"
    echo "  input : ${input_csv}"
    echo "  output: ${output_csv}"

    python -m src.data.generate_analysis_self \
        --input_csv "${input_csv}" \
        --output_csv "${output_csv}" \
        --model_name "${model}" \
        --output_col routing_analysis \
        --max_tokens "${MAX_TOKENS}"
done
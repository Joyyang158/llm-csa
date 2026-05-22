# Generate teacher-written routing analysis via the Together API.
#
# Override via env vars:
#   DOMAIN=math|science   SPLIT=train|test   TEACHER_MODEL=<together-id>
#
# Requires: TOGETHER_API_KEY in environment.


DOMAIN="${DOMAIN:-math}"
SPLIT="${SPLIT:-train}"
MAX_TOKENS="${MAX_TOKENS:-1000}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-tput}"

if [ -z "${TOGETHER_API_KEY:-}" ]; then
    echo "ERROR: TOGETHER_API_KEY is not set." >&2
    exit 1
fi

MODELS=(
    Qwen3-0.6B
    Qwen3-1.7B
    Qwen3-4B
    Qwen3-8B
)

INPUT_TEMPLATE="outputs/model_prediction_on_benchmark/${DOMAIN}/{model_short}/${SPLIT}_graded.csv"
OUTPUT_TEMPLATE="outputs/analysis/${DOMAIN}/{model_short}/${SPLIT}_teacher.csv"

for model_short in "${MODELS[@]}"; do
    input_csv="${INPUT_TEMPLATE//\{model_short\}/${model_short}}"
    output_csv="${OUTPUT_TEMPLATE//\{model_short\}/${model_short}}"

    echo "---------------------------------------------------"
    echo "Teacher analysis: target=${model_short}"
    echo "  teacher: ${TEACHER_MODEL}"
    echo "  input  : ${input_csv}"
    echo "  output : ${output_csv}"

    python -m src.data.generate_analysis_teacher \
        --input_csv "${input_csv}" \
        --output_csv "${output_csv}" \
        --teacher_model "${TEACHER_MODEL}" \
        --max_tokens "${MAX_TOKENS}"
done
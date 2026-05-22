# Grade science benchmark answers across one or more models.

MODELS=(
    Qwen3-0.6B
    Qwen3-1.7B
    Qwen3-4B
    Qwen3-8B
    Qwen3-14B
)

INPUT_TEMPLATE="outputs/answers/science/{model}/test.csv"
OUTPUT_TEMPLATE="outputs/answers/science/{model}/test_graded.csv"
EVAL_COL="${EVAL_COL:-generation}"
MODE="${MODE:-multi}"

for model in "${MODELS[@]}"; do
    input_csv="${INPUT_TEMPLATE//\{model\}/${model}}"
    output_csv="${OUTPUT_TEMPLATE//\{model\}/${model}}"

    echo "---------------------------------------------------"
    echo "Grading: model=${model}"
    echo "  input : ${input_csv}"
    echo "  output: ${output_csv}"

    python -m src.data.grade_answers \
        --csv_path "${input_csv}" \
        --eval_col "${EVAL_COL}" \
        --domain science \
        --mode "${MODE}" \
        --output_path "${output_csv}"
done

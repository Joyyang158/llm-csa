# Grade math benchmark answers across one or more models.
#
# Edit the MODELS array and INPUT_TEMPLATE / OUTPUT_TEMPLATE below to match
# your directory layout. Any "{model}" token in the templates is replaced
# with each model name.

MODELS=(
    Qwen3-0.6B
    Qwen3-1.7B
    Qwen3-4B
    Qwen3-8B
    Qwen3-14B
)

INPUT_TEMPLATE="outputs/answers/math/{model}/test.csv"
OUTPUT_TEMPLATE="outputs/answers/math/{model}/test_graded.csv"
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
        --domain math \
        --mode "${MODE}" \
        --output_path "${output_csv}"
done

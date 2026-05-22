# Compute the average tokenised length of generations in a CSV column.


CSV_PATH="${CSV_PATH:-data/math/answers.csv}"
COLUMN="${COLUMN:-generation}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-0.6B}"

python -m src.data.measure_token_length \
    --csv_path "${CSV_PATH}" \
    --column "${COLUMN}" \
    --tokenizer "${TOKENIZER}"

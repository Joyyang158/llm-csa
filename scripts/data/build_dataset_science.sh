# Build the science benchmark dataset (MMLU-Pro filtered to science).

OUTPUT_DIR="${1:-data/science}"

python -m src.data.build_dataset_science \
    --output_dir "${OUTPUT_DIR}" \
    --test_size 700

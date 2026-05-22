# Build the math benchmark dataset (GSM8K + MATH-500 + AIME).

OUTPUT_DIR="${1:-data/math}"

python -m src.data.build_dataset_math \
    --output_dir "${OUTPUT_DIR}"

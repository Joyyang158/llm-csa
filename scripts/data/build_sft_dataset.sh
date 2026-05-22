# Build the SFT training CSV by selecting per-row analysis from a
# teacher_binary output (which produced both label_0_analysis and
# label_1_analysis).


DOMAIN="${DOMAIN:-math}"
MODEL_CSV="${MODEL_CSV:-data/${DOMAIN}/answers_with_labels.csv}"
ANALYSIS_CSV="${ANALYSIS_CSV:-data/${DOMAIN}/analyses_teacher_binary.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-data/${DOMAIN}/sft_train.csv}"

MODEL_CSV=outputs/model_prediction_on_benchmark/math/Qwen3-1.7B/train.csv
ANALYSIS_CSV=outputs/analysis/math/Qwen3-1.7B/Qwen3-0.6B_train.csv
OUTPUT_CSV=outputs/sft/math/Qwen3-1.7B/train.csv

python -m src.data.build_sft_dataset \
    --model_csv "${MODEL_CSV}" \
    --analysis_csv "${ANALYSIS_CSV}" \
    --output_csv "${OUTPUT_CSV}" \
    --analysis_col SFT_analysis

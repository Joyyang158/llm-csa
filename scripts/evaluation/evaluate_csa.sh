# Evaluate CSA predictions against ground-truth correctness labels.
#
# Two modes (set GRADE_MODE):
#   column      — read is_correct from a column in CSV_PATH.
#   generations — grade raw generations from GT_CSV_PATH on the fly.


GRADE_MODE="${GRADE_MODE:-column}"
DOMAIN="${DOMAIN:-math}"
CSV_PATH="${CSV_PATH:-outputs/csa/${DOMAIN}/predictions.csv}"
PREDICTION_COL="${PREDICTION_COL:-decision}"

case "${GRADE_MODE}" in
    column)
        IS_CORRECT_COL="${IS_CORRECT_COL:-is_correct}"
        python -m src.csa.evaluate \
            --grade_mode column \
            --csv_path "${CSV_PATH}" \
            --prediction_col "${PREDICTION_COL}" \
            --is_correct_col "${IS_CORRECT_COL}"
        ;;
    generations)
        GT_CSV_PATH="${GT_CSV_PATH:-data/${DOMAIN}/answers.csv}"
        GENERATIONS_COL="${GENERATIONS_COL:-generation}"
        python -m src.csa.evaluate \
            --grade_mode generations \
            --csv_path "${CSV_PATH}" \
            --prediction_col "${PREDICTION_COL}" \
            --gt_csv_path "${GT_CSV_PATH}" \
            --generations_col "${GENERATIONS_COL}" \
            --domain "${DOMAIN}"
        ;;
    *)
        echo "[err] Unknown GRADE_MODE: ${GRADE_MODE}." >&2
        exit 1
        ;;
esac

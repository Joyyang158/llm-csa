#!/usr/bin/env bash
set -euo pipefail

# Default to labels re-measured on the evaluated checkpoint.
GRADE_MODE="${GRADE_MODE:-generations}"
DOMAIN="${DOMAIN:-math}"
CSV_PATH="${CSV_PATH:?Set CSV_PATH to CSA predictions}"
PREDICTION_COL="${PREDICTION_COL:-decision}"

case "${GRADE_MODE}" in
    column)
        python -m src.csa.evaluate \
            --grade_mode column \
            --csv_path "${CSV_PATH}" \
            --prediction_col "${PREDICTION_COL}" \
            --is_correct_col "${IS_CORRECT_COL:-is_correct}"
        ;;
    generations)
        GT_CSV_PATH="${GT_CSV_PATH:?Set GT_CSV_PATH to answers from the evaluated checkpoint}"
        python -m src.csa.evaluate \
            --grade_mode generations \
            --csv_path "${CSV_PATH}" \
            --prediction_col "${PREDICTION_COL}" \
            --gt_csv_path "${GT_CSV_PATH}" \
            --generations_col "${GENERATIONS_COL:-generation}" \
            --domain "${DOMAIN}"
        ;;
    *) echo "Unknown GRADE_MODE: ${GRADE_MODE}" >&2; exit 1 ;;
esac

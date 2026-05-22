# Capability Ratio (CR) — paper Eq. 8.
#
# Compares the post-training model's solve accuracy to the base model's solve
# accuracy on the same queries. Both PRE_CSV and POST_CSV must be generations
# CSVs produced by ``src/data/generate_answers.py`` over the SAME evaluation
# split.

DOMAIN="${DOMAIN:-math}"
PRE_CSV="${PRE_CSV:?Set PRE_CSV=/path/to/base-model/test.csv}"
POST_CSV="${POST_CSV:?Set POST_CSV=/path/to/post-training-model/test.csv}"
GENERATIONS_COL="${GENERATIONS_COL:-generation}"

python -m src.csa.capability_ratio \
    --pre_csv "${PRE_CSV}" \
    --post_csv "${POST_CSV}" \
    --generations_col "${GENERATIONS_COL}" \
    --domain "${DOMAIN}"

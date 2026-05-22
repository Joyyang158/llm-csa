# Upload a CSV to the HuggingFace Hub as a dataset.
#
# Required env var: HF_TOKEN (write-permission HF token).


CSV_PATH="${CSV_PATH:?Set CSV_PATH=/path/to/data.csv}"
REPO_ID="${REPO_ID:?Set REPO_ID=your-username/dataset-name}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "[err] HF_TOKEN is not set." >&2
    exit 1
fi

EXTRA_FLAGS=""
[ "${PRIVATE:-0}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --private"

python -m src.hub.upload_dataset \
    --csv_path "${CSV_PATH}" \
    --repo_id "${REPO_ID}" \
    ${EXTRA_FLAGS}

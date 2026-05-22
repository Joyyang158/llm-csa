# Upload a local model checkpoint folder to the HuggingFace Hub.
#
# Required env var: HF_TOKEN (write-permission HF token).


LOCAL_DIR="${LOCAL_DIR:?Set LOCAL_DIR=/path/to/checkpoint}"
REPO_ID="${REPO_ID:?Set REPO_ID=your-username/model-name}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "[err] HF_TOKEN is not set." >&2
    exit 1
fi

EXTRA_FLAGS=""
[ "${PRIVATE:-0}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --private"

python -m src.hub.upload_model \
    --local_dir "${LOCAL_DIR}" \
    --repo_id "${REPO_ID}" \
    ${EXTRA_FLAGS}

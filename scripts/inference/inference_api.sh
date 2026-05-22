# Run a closed-source API model as a CSA model.
#
# Required env var (depending on MODEL_FAMILY):
#   OPENAI_API_KEY     — for openai / gpt
#   GOOGLE_API_KEY     — for gemini
#   ANTHROPIC_API_KEY  — for claude
#   TOGETHER_API_KEY   — for together / togetherai


DOMAIN="${DOMAIN:-math}"
SPLIT="${SPLIT:-test}"
MODEL_FAMILY="${MODEL_FAMILY:-openai}"

# Pick a sensible default model per family
case "${MODEL_FAMILY,,}" in
  openai|gpt)             DEFAULT_MODEL="gpt-4o-mini" ;;
  gemini|google)          DEFAULT_MODEL="gemini-2.5-flash" ;;
  claude|anthropic)       DEFAULT_MODEL="claude-sonnet-4-5" ;;
  together|togetherai)    DEFAULT_MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo" ;;
  *)
    echo "[err] Unknown MODEL_FAMILY='${MODEL_FAMILY}'." >&2
    echo "      Use one of: openai | gemini | claude | together" >&2
    exit 1
    ;;
esac

MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL}}"
INPUT_CSV="${INPUT_CSV:-data/${DOMAIN}/${SPLIT}.csv}"
# Strip provider prefix from model name (e.g. meta-llama/Llama-3.3-70B → Llama-3.3-70B)
OUTPUT_CSV="${OUTPUT_CSV:-outputs/csa_api/${DOMAIN}/${MODEL_NAME##*/}_${SPLIT}.csv}"

python -m src.csa.inference_api \
  --domain "${DOMAIN}" \
  --model_family "${MODEL_FAMILY}" \
  --model_name "${MODEL_NAME}" \
  --input_csv "${INPUT_CSV}" \
  --output_csv "${OUTPUT_CSV}" \
  --save_every 50
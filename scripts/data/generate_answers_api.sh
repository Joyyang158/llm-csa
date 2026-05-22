# Collect benchmark answers from a hosted model.
#
# Supported providers (set via PROVIDER env var):
#   together   -> needs TOGETHER_API_KEY
#   openai     -> needs OPENAI_API_KEY      (GPT models)
#   anthropic  -> needs ANTHROPIC_API_KEY   (Claude models)
#   google     -> needs GOOGLE_API_KEY      (Gemini models)
#
# Override via env vars: DOMAIN, SPLIT, PROVIDER, MODEL_NAME, NUM_GENERATIONS,
# MAX_TOKENS, OUTPUT_CSV.


DOMAIN="${DOMAIN:-math}"
SPLIT="${SPLIT:-test}"
PROVIDER="${PROVIDER:-together}"
NUM_GENERATIONS="${NUM_GENERATIONS:-1}"
MAX_TOKENS="${MAX_TOKENS:-100000}"

# Default model + required API key per provider
case "${PROVIDER}" in
  together)
    DEFAULT_MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo"
    REQUIRED_KEY="TOGETHER_API_KEY"
    ;;
  openai)
    DEFAULT_MODEL="gpt-4o-mini"
    REQUIRED_KEY="OPENAI_API_KEY"
    ;;
  anthropic)
    DEFAULT_MODEL="claude-sonnet-4-5"
    REQUIRED_KEY="ANTHROPIC_API_KEY"
    ;;
  google)
    DEFAULT_MODEL="gemini-2.5-flash"
    REQUIRED_KEY="GOOGLE_API_KEY"
    ;;
  *)
    echo "[err] Unknown PROVIDER='${PROVIDER}'. Use: together|openai|anthropic|google" >&2
    exit 1
    ;;
esac

MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL}}"
INPUT_CSV="${INPUT_CSV:-data/${DOMAIN}/${SPLIT}.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/answers_api/${DOMAIN}/${MODEL_NAME##*/}/${SPLIT}.csv}"

# Validate key
if [ -z "${!REQUIRED_KEY:-}" ]; then
  echo "[err] ${REQUIRED_KEY} is not set (required for provider='${PROVIDER}')." >&2
  exit 1
fi

python -m src.data.generate_answers_api \
  --provider "${PROVIDER}" \
  --domain "${DOMAIN}" \
  --model_name "${MODEL_NAME}" \
  --input_csv "${INPUT_CSV}" \
  --output_csv "${OUTPUT_CSV}" \
  --output_col generation \
  --max_tokens "${MAX_TOKENS}" \
  --num_generations "${NUM_GENERATIONS}"
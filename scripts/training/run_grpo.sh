# GRPO training launcher (full-parameter, no LoRA).
#
# Usage:
#   bash scripts/training/run_grpo.sh scripts/training/grpo.yaml


# Optional environment variables you may want to export before running:
#   export TRITON_CACHE_DIR=/path/to/triton_cache   # Triton kernel cache location
#   export HF_HOME=/path/to/hf                      # HuggingFace cache directory
#   export WANDB_API_KEY=<your_wandb_key>           # for logging to Weights & Biases
#   export HF_TOKEN=<your_hf_token>                 # for downloading gated HF models

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <training_config.yaml>" >&2
    exit 1
fi

CONFIG="$1"
ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/training/deepspeed_zero3.yaml}"

if [ ! -f "${CONFIG}" ]; then
    echo "[err] Training config not found: ${CONFIG}" >&2
    exit 1
fi
if [ ! -f "${ACCEL_CONFIG}" ]; then
    echo "[err] Accelerate config not found: ${ACCEL_CONFIG}" >&2
    exit 1
fi

echo "Training config  : ${CONFIG}"
echo "Accelerate config: ${ACCEL_CONFIG}"

accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    -m src.training.grpo \
    --config "${CONFIG}"

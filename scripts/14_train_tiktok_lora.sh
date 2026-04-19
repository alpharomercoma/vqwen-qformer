#!/usr/bin/env bash
# Non-destructive: writes to checkpoints/tiktok-lora/, never touches stage-1.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
exec /home/alpha/vqwen/.venv/bin/python -m vqwen_qformer.train \
    --config configs/tiktok_lora.yaml \
    --output_dir checkpoints/tiktok-lora \
    --loss_log_jsonl logs/tiktok_lora_loss.jsonl \
    "$@"

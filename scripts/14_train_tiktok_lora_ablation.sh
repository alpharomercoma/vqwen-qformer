#!/usr/bin/env bash
# ABLATION: vision-only TikTok fine-tune (no audio transcript modality).
# Reproduces `alpharomercoma/vqwen-qformer-tiktok`. Writes to
# checkpoints/tiktok-lora-ablation/, never touches stage-1.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
exec "$REPO_ROOT/.venv/bin/python" -m vqwen_qformer.train \
    --config configs/tiktok_lora_ablation_no_transcript.yaml \
    --output_dir checkpoints/tiktok-lora-ablation \
    --loss_log_jsonl logs/tiktok_lora_ablation_loss.jsonl \
    "$@"

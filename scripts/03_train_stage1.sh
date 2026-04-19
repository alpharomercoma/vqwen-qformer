#!/usr/bin/env bash
# Launch stage-1 training on H200 with tuned env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# expandable_segments: avoids memory fragmentation during large activation allocs.
# Especially helps when varying seq-lengths hit the allocator.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HF transfer: fast downloads for any on-demand fetches (hub snapshots).
export HF_HUB_ENABLE_HF_TRANSFER=1

# Disable tokenizer parallelism warning when DataLoader num_workers > 0.
export TOKENIZERS_PARALLELISM=false

mkdir -p logs
exec /home/alpha/vqwen/.venv/bin/python -m vqwen_qformer.train \
    --config configs/stage1.yaml \
    --output_dir checkpoints/stage1 \
    --loss_log_jsonl logs/stage1_loss.jsonl \
    "$@"

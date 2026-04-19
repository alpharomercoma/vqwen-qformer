#!/usr/bin/env bash
# Precompute frozen Q-Former features for every unique image in mix665k.
# Run once, reused by all subsequent stage-2 trainings.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs data/stage2/qformer_cache
exec /home/alpha/vqwen/.venv/bin/python scripts/05_cache_features.py \
    --json_path data/stage2/llava_v1_5_mix665k.json \
    --image_root data/stage2/images \
    --cache_dir data/stage2/qformer_cache \
    --blip2_bundle_path models/blip2-frozen \
    --batch_size 256 \
    --num_workers 16 \
    "$@"

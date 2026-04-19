#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p logs data/tiktok_fps1/qformer_cache
exec /home/alpha/vqwen/.venv/bin/python scripts/05_cache_features.py \
    --json_path data/tiktok_fps1/tiktok_train.json \
    --image_root data/tiktok_fps1/frames \
    --cache_dir data/tiktok_fps1/qformer_cache \
    --blip2_bundle_path models/blip2-frozen \
    --batch_size 256 \
    --num_workers 16 \
    "$@"

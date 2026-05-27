#!/usr/bin/env bash
# Cache (B, 32, 768) Q-Former features for the v2 (with-transcript) training set.
# Points at data/tiktok_v2/ and writes data/tiktok_v2/qformer_cache/.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p logs data/tiktok_v2/qformer_cache
exec "$REPO_ROOT/.venv/bin/python" scripts/05_cache_features.py \
    --json_path data/tiktok_v2/tiktok_train.json \
    --image_root data/tiktok_v2/frames \
    --cache_dir data/tiktok_v2/qformer_cache \
    --blip2_bundle_path models/blip2-frozen \
    --batch_size 384 \
    --num_workers 16 \
    "$@"

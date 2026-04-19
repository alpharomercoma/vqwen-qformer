"""Download google/gemma-3-27b-it — an independent judge for the teacher↔GT
cross-compare (see scripts/19_cross_compare.py).

Different architecture family from Qwen (so its agreements are a real tie-break),
multimodal, and fully supported in our pinned transformers version.
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    load_dotenv(REPO_ROOT / ".env")
    repo_id = "google/gemma-3-27b-it"
    out = REPO_ROOT / "models" / "gemma-3-27b-it"
    if (out / "config.json").exists():
        print(f"[download] already present at {out}")
        return 0
    print(f"[download] {repo_id} -> {out}  (~62 GB)")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out),
        ignore_patterns=["*.bin", "pytorch_model.bin.index.json"],
    )
    print("[download] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

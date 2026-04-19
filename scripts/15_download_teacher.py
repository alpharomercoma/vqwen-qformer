"""Download Qwen3-VL-30B-A3B-Thinking into models/Qwen3-VL-teacher/.

Used as a teacher VLM for per-frame label distillation on the TikTok dataset.
~60 GB of safetensors — one-time cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    load_dotenv(REPO_ROOT / ".env")
    repo_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    out = REPO_ROOT / "models" / "Qwen3-VL-30B-A3B-Instruct"
    if (out / "config.json").exists():
        print(f"[download] already present at {out}")
        return 0
    print(f"[download] {repo_id} -> {out}  (~70 GB)")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out),
        ignore_patterns=["*.bin", "pytorch_model.bin.index.json"],
    )
    print("[download] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

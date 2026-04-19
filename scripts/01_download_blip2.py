"""Download + compact-extract the frozen vision+Q-Former bundle from
Salesforce/blip2-opt-2.7b.

Stage 1 (download): snapshot the full HF repo (~15 GB fp32, both pytorch_model
shards) — we need both because safetensors/pytorch shards interleave all
submodules.

Stage 2 (extract): load `Blip2Model`, pull out `vision_model`, `qformer`,
`query_tokens`, save them as a small bundle under `models/blip2-frozen/` in
bf16. Final size ~1.6 GB (EVA-ViT-G 1 GB + Q-Former 400 MB + queries 200 KB).

Stage 3 (cleanup): delete the 15 GB original unless --keep is passed.

Why this detour: Blip2 shards are not split by submodule; you can't download
just the Q-Former. Extracting once means subsequent training runs load a
compact, self-contained bundle.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--full_dir", default=str(REPO_ROOT / "models" / "blip2-opt-2.7b"))
    p.add_argument("--out_dir", default=str(REPO_ROOT / "models" / "blip2-frozen"))
    p.add_argument("--keep", action="store_true",
                   help="Keep the full 15 GB snapshot after extraction")
    args = p.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    full_dir = Path(args.full_dir)
    out_dir = Path(args.out_dir)

    if (out_dir / "vision_model.bin").exists() and (out_dir / "qformer.bin").exists():
        print(f"[download] bundle already extracted at {out_dir}")
        return 0

    if not (full_dir / "config.json").exists():
        print(f"[download] snapshot {args.repo} -> {full_dir}")
        # Avoid pulling both safetensors + pytorch_model shards (each is ~15 GB).
        snapshot_download(
            repo_id=args.repo,
            local_dir=str(full_dir),
            ignore_patterns=["*.bin", "pytorch_model.bin.index.json"],
        )
    else:
        print(f"[download] full snapshot already present at {full_dir}")

    print(f"[extract] loading Blip2Model in bf16 to compact out")
    from transformers import AutoImageProcessor, Blip2Model
    blip2 = Blip2Model.from_pretrained(str(full_dir), dtype=torch.bfloat16)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(blip2.vision_model.state_dict(), out_dir / "vision_model.bin")
    torch.save(blip2.qformer.state_dict(),      out_dir / "qformer.bin")
    torch.save({"query_tokens": blip2.query_tokens.data.cpu()},
               out_dir / "query_tokens.bin")
    # Preserve the Blip2Config so we can instantiate identical sub-modules later.
    blip2.config.save_pretrained(out_dir)
    # Save the image processor (Blip2Processor uses a BlipImageProcessor: 224x224).
    try:
        proc = AutoImageProcessor.from_pretrained(str(full_dir))
        proc.save_pretrained(out_dir)
    except Exception as e:
        print(f"[extract] image processor save skipped: {e}")
    print(f"[extract] wrote vision_model.bin, qformer.bin, query_tokens.bin to {out_dir}")

    # Free up disk
    if not args.keep:
        print(f"[cleanup] removing full snapshot at {full_dir}")
        shutil.rmtree(full_dir)

    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

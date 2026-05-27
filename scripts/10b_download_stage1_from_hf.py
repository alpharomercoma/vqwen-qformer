"""Pull the stage-1 linear projector from `alpharomercoma/vqwen-qformer-pretrain`.

The HF model stores the full BLIP-2-style checkpoint (vision + Q-Former + LM)
in sharded safetensors. We only need the linear projector
(`language_projection.{weight,bias}`, shape (2560, 768) + (2560,)), which the
stage-2 trainer loads via `load_stage1_projector` in `vqwen_qformer.model`.

We grab the index, locate the shard that contains the projector tensors, and
download only that shard (~252 MB instead of ~10 GB).  Output:
    checkpoints/stage1/projector.bin
with the format `{"projector.fc.weight": ..., "projector.fc.bias": ...}`
that `load_stage1_projector` expects after stripping the `projector.` prefix.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "alpharomercoma/vqwen-qformer-pretrain"
OUT_PATH = REPO_ROOT / "checkpoints" / "stage1" / "projector.bin"

# Candidate names for the projector tensors inside the safetensors shard.
# HF Blip2's saved name is `language_projection.{weight,bias}`. We also accept
# a couple of historical variants in case the upload script renamed them.
CAND_W = ["language_projection.weight", "projector.fc.weight", "projector.weight"]
CAND_B = ["language_projection.bias",   "projector.fc.bias",   "projector.bias"]


def main() -> int:
    if OUT_PATH.exists():
        print(f"[stage1] projector already at {OUT_PATH}; skipping.")
        return 0

    print(f"[stage1] fetching index from {REPO_ID}")
    idx_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    weight_map: dict[str, str] = idx["weight_map"]

    def pick(cands: list[str]) -> tuple[str, str]:
        for k in cands:
            if k in weight_map:
                return k, weight_map[k]
        raise KeyError(
            f"None of {cands} found in weight_map. Sample keys: "
            f"{list(weight_map)[:6]}"
        )

    w_key, w_shard = pick(CAND_W)
    b_key, b_shard = pick(CAND_B)
    shards = sorted({w_shard, b_shard})
    print(f"[stage1] projector keys: {w_key}, {b_key}  shards: {shards}")

    tensors: dict[str, torch.Tensor] = {}
    for shard in shards:
        print(f"[stage1] downloading shard {shard}")
        sp = hf_hub_download(repo_id=REPO_ID, filename=shard)
        with safe_open(sp, framework="pt") as f:
            keys = set(f.keys())
            if w_key in keys: tensors["weight"] = f.get_tensor(w_key)
            if b_key in keys: tensors["bias"]   = f.get_tensor(b_key)
    if "weight" not in tensors or "bias" not in tensors:
        print(f"[stage1] could not locate both projector tensors; got {list(tensors)}",
              file=sys.stderr)
        return 1

    w, b = tensors["weight"], tensors["bias"]
    # Sanity: Linear(768, 2560) ⇒ weight is (2560, 768), bias is (2560,).
    if tuple(w.shape) != (2560, 768) or tuple(b.shape) != (2560,):
        print(f"[stage1] unexpected projector shape: weight={tuple(w.shape)} "
              f"bias={tuple(b.shape)} (expected (2560,768) and (2560,))",
              file=sys.stderr)
        return 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # `load_stage1_projector` strips a leading `projector.` prefix, so save in
    # the `projector.fc.{weight,bias}` form.
    state = {
        "projector.fc.weight": w.to(torch.float32).contiguous(),
        "projector.fc.bias":   b.to(torch.float32).contiguous(),
    }
    torch.save(state, OUT_PATH)
    sz = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"[stage1] saved {OUT_PATH}  ({sz:.1f} MB,  weight={tuple(w.shape)} "
          f"bias={tuple(b.shape)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

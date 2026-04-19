"""LlavaPretrainDataset + LlavaInstructDataset — same as vqwen but imports from this pkg."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import IGNORE_INDEX
from .preprocess import preprocess_plain, preprocess_qwen


class LlavaPretrainDataset(Dataset):
    def __init__(self, json_path, image_root, tokenizer, image_processor, max_length: int = 2048):
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        with open(json_path) as f:
            self.records: List[Dict[str, Any]] = json.load(f)

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(self.image_root / rec["image"]).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        tokenized = preprocess_plain([rec["conversations"]], self.tokenizer, max_length=self.max_length)
        return {"input_ids": tokenized["input_ids"][0], "labels": tokenized["labels"][0], "pixel_values": pixel_values}


class LlavaInstructDataset(Dataset):
    def __init__(self, json_path, image_root, tokenizer, image_processor, max_length: int = 2048,
                 num_query_tokens: int = 32):
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        with open(json_path) as f:
            raw_records: List[Dict[str, Any]] = json.load(f)

        known_exists: Dict[str, bool] = {}
        dropped = 0
        self.records: List[Dict[str, Any]] = []
        for r in raw_records:
            img_rel = r.get("image")
            if img_rel:
                if img_rel not in known_exists:
                    known_exists[img_rel] = (self.image_root / img_rel).exists()
                if not known_exists[img_rel]:
                    dropped += 1
                    continue
            self.records.append(r)
        if dropped:
            print(f"[LlavaInstructDataset] dropped {dropped} records with missing images "
                  f"(kept {len(self.records)}/{len(raw_records)})")

        crop_size = self.image_processor.crop_size
        self._zero_img_shape = (3, crop_size["height"], crop_size["width"])

        # Q-Former splices `num_query_tokens` per image (vs 576 for MLP path).
        # Length proxy: chars + constant image bias ≈ 3×tokens.
        img_bias = num_query_tokens * 3
        self.lengths: List[int] = [
            sum(len(t["value"]) for t in r["conversations"])
            + (img_bias if r.get("image") else 0)
            for r in self.records
        ]

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        attempts = 0
        max_attempts = 5
        while True:
            rec = self.records[idx]
            has_image = "image" in rec and rec["image"]
            try:
                if has_image:
                    image = Image.open(self.image_root / rec["image"]).convert("RGB")
                    pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
                else:
                    pixel_values = torch.zeros(self._zero_img_shape, dtype=torch.float32)
                tokenized = preprocess_qwen([rec["conversations"]], self.tokenizer,
                                            has_image=has_image, max_length=self.max_length)
                return {"input_ids": tokenized["input_ids"][0],
                        "labels": tokenized["labels"][0],
                        "pixel_values": pixel_values}
            except Exception as e:
                attempts += 1
                print(f"[LlavaInstructDataset] idx={idx} failed ({type(e).__name__}: {e}); "
                      f"fallback (attempt {attempts}/{max_attempts})")
                if attempts >= max_attempts:
                    raise
                idx = (idx + 1) % len(self.records)


class LlavaInstructCachedDataset(Dataset):
    """Stage-2 dataset that reads pre-computed Q-Former features from disk
    instead of decoding images + running ViT-G + Q-Former per step.

    Expects a cache dir produced by `scripts/05_cache_features.py`:
      features.bin       (N, Q, H)  -- torch.save bf16 tensor, lazy-mmap at load
      image_to_idx.json  {image_relpath: idx}
      meta.json

    For text-only samples we emit an all-zero feature tensor; the model's
    fast-path still runs the projector on it but the output is discarded
    (there's no <image> sentinel so it's never spliced).
    """

    def __init__(self, json_path, cache_dir, tokenizer, max_length: int = 2048,
                 num_query_tokens: int = 32, expected_dtype: str = "bfloat16"):
        self.tokenizer = tokenizer
        self.max_length = max_length

        cache_dir = Path(cache_dir)
        with open(cache_dir / "image_to_idx.json") as f:
            self.image_to_idx = json.load(f)
        with open(cache_dir / "meta.json") as f:
            meta = json.load(f)
        if meta.get("dtype") != expected_dtype:
            raise RuntimeError(
                f"feature cache dtype ({meta.get('dtype')}) != expected ({expected_dtype}); "
                "re-run scripts/05_cache_features.py with matching --dtype"
            )
        # mmap for large tensors to avoid pegging RAM at dataset init.
        self.features = torch.load(cache_dir / "features.bin", map_location="cpu",
                                    weights_only=True, mmap=True)
        self.num_query_tokens = meta["num_query_tokens"]
        self.hidden_size = meta["hidden_size"]
        # Paths whose feature rows are zero (image-decode failed during caching).
        # Drop these records to match live-compute LlavaInstructDataset behavior
        # (which filters missing / unreadable images upfront).
        failed_set = set(meta.get("failed_indices") or [])
        idx_to_path = {v: k for k, v in self.image_to_idx.items()}
        failed_paths = {idx_to_path[i] for i in failed_set if i in idx_to_path}

        with open(json_path) as f:
            raw_records: List[Dict[str, Any]] = json.load(f)
        missing = failed = 0
        self.records: List[Dict[str, Any]] = []
        for r in raw_records:
            img = r.get("image")
            if img:
                if img not in self.image_to_idx:
                    missing += 1
                    continue
                if img in failed_paths:
                    failed += 1
                    continue
            self.records.append(r)
        if missing or failed:
            print(f"[LlavaInstructCachedDataset] dropped {missing} uncached + {failed} failed-decode "
                  f"records (kept {len(self.records)}/{len(raw_records)})")

        self._zero_features = torch.zeros(self.num_query_tokens, self.hidden_size,
                                          dtype=self.features.dtype)
        img_bias = num_query_tokens * 3
        self.lengths: List[int] = [
            sum(len(t["value"]) for t in r["conversations"])
            + (img_bias if r.get("image") else 0)
            for r in self.records
        ]

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = rec.get("image")
        feats = self.features[self.image_to_idx[img]].clone() if img else self._zero_features.clone()
        tokenized = preprocess_qwen([rec["conversations"]], self.tokenizer,
                                    has_image=bool(img), max_length=self.max_length)
        return {"input_ids": tokenized["input_ids"][0],
                "labels": tokenized["labels"][0],
                "qformer_features": feats}


@dataclass
class DataCollatorForSupervisedDataset:
    """Handles both `pixel_values` (vision+qformer run live) and
    `qformer_features` (features pre-cached) dataset shapes."""
    pad_token_id: int

    def __call__(self, instances):
        input_ids_list = [x["input_ids"] for x in instances]
        labels_list = [x["labels"] for x in instances]
        max_len = max(t.size(0) for t in input_ids_list)
        B = len(instances)
        input_ids = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
            L = ids.size(0)
            input_ids[i, :L] = ids
            labels[i, :L] = lab
            attention_mask[i, :L] = 1

        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        if "pixel_values" in instances[0]:
            batch["pixel_values"] = torch.stack([x["pixel_values"] for x in instances])
        if "qformer_features" in instances[0]:
            batch["qformer_features"] = torch.stack([x["qformer_features"] for x in instances])
        return batch

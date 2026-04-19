# vqwen-qformer

MiniGPT-4-style vision-language model: **EVA-ViT-G + Q-Former + Linear + Qwen3-4B**, specialized for TikTok sludge-content detection via a teacher-student distillation loop with independent cross-judge validation.

Published weights: https://huggingface.co/alpharomercoma/vqwen-qformer-tiktok

## Architecture

```
Image (224×224)
    → EVA-ViT-G/14      (frozen, from Salesforce/blip2-opt-2.7b)
    → Q-Former 12-layer (frozen, 32 pretrained query tokens)
    → Linear 768→2560   (trained)
    → Qwen3-4B          (frozen stage-1; LoRA stage-2)
```

Loads as a stock `Blip2ForConditionalGeneration` — no `trust_remote_code`.

## Pipeline overview

1. **Stage 1 — feature alignment** (`scripts/02-03_*`): train only the Linear projector on `liuhaotian/LLaVA-Pretrain` 558 K. CLIP + Q-Former + Qwen3 all frozen.
2. **Stage 2 — instruction tuning** (`scripts/04-06_*`): LoRA on Qwen3 + continued projector training on `liuhaotian/LLaVA-Instruct-150K` (mix665k).
3. **TikTok specialization** (`scripts/11-30_*`):
   - Extract 1-fps frames from tiktok-sludge v13 videos
   - Teacher-distill labels with `Qwen3-VL-30B-A3B-Instruct` (bs=8 batched on H200, ~2 h for 7 K frames)
   - Cross-validate 250 teacher↔GT disagreements with `Gemma-3-27b-it` as an independent judge
   - Build multi-task conversations (classify / layout / describe / coupled-explain / refuse)
   - LoRA fine-tune (r=16, α=32, dropout=0.15) on cached frozen features
   - Dual-benchmark eval: raw GT and Gemma-cleaned labels
   - Export as `Blip2ForConditionalGeneration` + push to HF Hub

## Final model (v7)

- **96.7 %** classify accuracy on held-out 300 test videos vs Gemma-validated cleaned labels
- **89.0 %** vs raw human GT (7.7 pp lower because Gemma confirmed ~10-12 % GT label noise)
- Zero fabricated show/game/channel mentions — descriptions grounded in visible content only
- Multi-turn classify+explain is internally consistent (2-turn coupled training)
- Graceful refusal on specific-content queries

See [the HF model card](https://huggingface.co/alpharomercoma/vqwen-qformer-tiktok) for detailed eval results and usage.

## Repository layout

```
src/vqwen_qformer/    # core training + inference package
├── model.py          # VQwenQFormerForCausalLM (skip-vision for cached-features mode)
├── projector.py      # 1-layer Linear 768 → 2560
├── dataset.py        # pretrain / instruct / cached-features datasets
├── preprocess.py     # Qwen3 chat template + image-token splicing
├── train.py          # QFormerTrainer: LoRA + projector + best-ckpt callback + eval + early-stop
└── generate.py       # inference helpers

scripts/
├── 01_download_blip2.py          # fetch + extract vision+qformer bundle
├── 02_smoke_test.py              # 20-step loss-drop gate
├── 03_train_stage1.sh            # launch stage 1
├── 04_train_stage2.sh            # launch stage 2 (mix665k)
├── 05_cache_features.py          # precompute frozen ViT-G + Q-Former outputs
├── 06_cache_stage2_features.sh   # stage-2 feature cache launcher
├── 11_extract_tiktok_1fps.py     # ffmpeg 1-fps frame extraction
├── 12_build_tiktok_convs.py      # conv builder using teacher-as-GT labels
├── 13_cache_tiktok.sh            # cache TikTok features
├── 14_train_tiktok_lora.sh       # TikTok LoRA launcher
├── 15_download_teacher.py        # Qwen3-VL-30B-A3B-Instruct
├── 16_label_with_teacher.py      # batched teacher labeling (bs=8)
├── 16b_label_test.py             # label held-out test split with same teacher
├── 18_download_judge.py          # Gemma-3-27b-it judge
├── 19_cross_compare.py           # Gemma judges teacher↔GT disputes
├── 20_eval_tiktok_test.py        # classify eval on test
├── 22_eval_dual_benchmark.py     # eval vs both raw GT and cleaned labels
├── 23_ab_test_batching.py        # batched-vs-serial quality check
├── 24_ab_test_bs_sweep.py        # find optimal batch size on H200
└── 30_export_hf.py               # build + save Blip2ForConditionalGeneration

configs/
├── stage1.yaml       # LLaVA-Pretrain alignment
├── stage2.yaml       # mix665k instruction tuning
└── tiktok_lora.yaml  # TikTok specialization
```

## Reproducing the TikTok pipeline

```bash
# 1. Prereqs: vqwen stage-1 checkpoint at checkpoints/stage1/projector.bin
uv sync

# 2. Download frozen BLIP-2 components (~1.6 GB after extraction)
python scripts/01_download_blip2.py

# 3. Extract TikTok 1-fps frames (requires v13 MP4s)
python scripts/11_extract_tiktok_1fps.py

# 4. Download teacher + judge (~120 GB combined, one-time)
python scripts/15_download_teacher.py
python scripts/18_download_judge.py

# 5. Teacher-label ~8.5K frames (batch=8 on H200, ~2 h)
python scripts/16_label_with_teacher.py --frames_per_video 5 --batch_size 8
python scripts/16b_label_test.py  # held-out test labels

# 6. Judge the disputes (~40 min Gemma-3)
python scripts/19_cross_compare.py

# 7. Build multi-task conversations (teacher-as-GT)
python scripts/12_build_tiktok_convs.py

# 8. Cache Q-Former features for all frames (~1 min)
bash scripts/13_cache_tiktok.sh

# 9. LoRA fine-tune (~12 min)
bash scripts/14_train_tiktok_lora.sh

# 10. Dual-benchmark eval
python scripts/22_eval_dual_benchmark.py --tag v7

# 11. Export to HF
python scripts/30_export_hf.py --test_generate --test_image <path> --force
```

## Hardware

Single NVIDIA H200 141 GB. All runs use bf16, SDPA attention, fused AdamW, Liger-Kernel (for stage-2 Qwen3-4B). Feature caching + batched teacher inference bring the teacher-labeling pass from 10 h (serial) to ~2 h (bs=8), at 97.5 % decision-parity with serial.

## License

Apache 2.0 for training / inference code. Base models retain original licenses: `Salesforce/blip2-opt-2.7b` (BSD-3), `Qwen/Qwen3-4B` (Apache 2.0), `google/gemma-3-27b-it` (Gemma).

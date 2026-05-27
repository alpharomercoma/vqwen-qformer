# vqwen-qformer

MiniGPT-4-style vision-language model: **EVA-ViT-G/14 + Q-Former + Linear + Qwen3-4B**, with Whisper-V3-Turbo transcripts concatenated into the language-model input. Specialized for TikTok sludge-content detection.

The canonical stage-2 pipeline (`*_v2`) trains on the Kaggle release
[`jobisaacong/tiktok-sludge-dataset-500`](https://www.kaggle.com/datasets/jobisaacong/tiktok-sludge-dataset-500)
with the audio-transcript modality enabled. The vision-only pipeline that
produced the original publish (`*_ablation_no_transcript`) is retained as a
labeled baseline — see [Ablations](#ablations).

Published weights (vision-only ablation):
https://huggingface.co/alpharomercoma/vqwen-qformer-tiktok
Stage-1 projector:
https://huggingface.co/alpharomercoma/vqwen-qformer-pretrain

## Architecture

```
Image (224×224)
    → EVA-ViT-G/14         (frozen, from Salesforce/blip2-opt-2.7b)   → (B, 257, 1408)
    → Q-Former 12-layer    (frozen, 32 pretrained query tokens)       → (B, 32, 768)
    → Linear 768→2560      (trained, ~2M params)                      → (B, 32, 2560)
                                                                       ↘
Audio (full clip) → Whisper-V3-Turbo → "Audio transcript: …" token stream ─┤
                                                                       ↗
    → Qwen3-4B             (frozen stage-1; LoRA stage-2)
```

Note: the `32` in `(B, 32, …)` is the Q-Former *sequence length* (number of
learnable query tokens), not a feature dimension. The Linear projector maps
each token's 768-d feature to 2560-d to match Qwen3-4B's input embedding
space, preserving the 32-token sequence.

Loads as a stock `Blip2ForConditionalGeneration` — no `trust_remote_code`.

## Pipeline overview

1. **Stage 1 — feature alignment** (`scripts/02-03_*`): train only the Linear projector on `liuhaotian/LLaVA-Pretrain` 558 K. CLIP + Q-Former + Qwen3 all frozen.
2. **Stage 2 — instruction tuning** (`scripts/04-06_*`): LoRA on Qwen3 + continued projector training on `liuhaotian/LLaVA-Instruct-150K` (mix665k).
3. **TikTok specialization v2 — canonical, with audio transcripts** (`scripts/10–14_*_v2`):
   - Download the Kaggle release (`jobisaacong/tiktok-sludge-dataset-500`)
   - Pull the stage-1 projector from `alpharomercoma/vqwen-qformer-pretrain` (only the small Linear, ~6 MB shard slice)
   - Extract 1-fps frames; build multi-task conversations with `"Audio transcript: <text>\n<image>\n<question>"` from the Kaggle Whisper-V3-Turbo transcripts
   - LoRA fine-tune (r=32, α=64, dropout=0.10) with the linear projector FROZEN (`train_projector: false`) on cached Q-Former features
   - Dual-benchmark eval with `--with_transcript`
   - Export as `Blip2ForConditionalGeneration` + push to HF Hub

   Why freeze the projector in stage 2? The stage-1 projector was aligned on 558 K LLaVA-Pretrain pairs; letting it drift further on the narrower 35 K-sample sludge dataset hurt downstream accuracy by ~0.8 pp in a controlled ablation. Freezing preserves the broader alignment and lets the LoRA adapters absorb all the domain shift.

## Results

### Canonical v2 (with audio transcripts, frozen projector) — published

[`alpharomercoma/vqwen-qformer-tiktok-v2`](https://huggingface.co/alpharomercoma/vqwen-qformer-tiktok-v2)

Held-out 300-video Kaggle test split, frame-level (7,380 frames):
- **99.04 %** accuracy
- **99.19 %** precision
- **99.22 %** recall
- **99.21 %** F1
- 0 unparseable replies

Video-level (majority vote across frames):
- **99.00 %** accuracy / F1 (149 of 150 sludge + 148 of 150 non-sludge correct)

### Vision-only ablation (no transcripts, trained projector) — baseline

[`alpharomercoma/vqwen-qformer-tiktok`](https://huggingface.co/alpharomercoma/vqwen-qformer-tiktok)

Same test split:
- **89.0 %** vs raw human GT
- **96.7 %** vs Gemma-validated cleaned labels
- Zero fabricated show/game/channel mentions — descriptions grounded in visible content only
- Multi-turn classify+explain internally consistent
- Graceful refusal on specific-content queries

The +10 pp gap between the two demonstrates the contribution of the Whisper-V3-Turbo audio modality.

### Projector freeze vs. train (controlled ablation on the v2 pipeline)

Why `train_projector: false` in `configs/tiktok_lora_v2.yaml`? An ablation
trained the v2 model under both regimes — same data, same LoRA hyperparams,
same seed sensitivity, only flipping the projector's `requires_grad`:

| Stage-2 projector | Frame-level accuracy | Frame-level F1 | Video-level accuracy |
|---|---|---|---|
| Trained at `projector_lr=2e-5` | 98.27 % | 98.57 % | 98.00 % |
| **Frozen (published config)** | **99.04 %** | **99.21 %** | **99.00 %** |

Best `eval_loss` was essentially tied (1.0967 frozen vs 1.0977 trained, both
at step 150). The freeze acts as regularisation: the stage-1 projector was
aligned on 558 K LLaVA-Pretrain pairs, and letting it drift further on the
narrower 35 K-sample sludge dataset slightly hurt downstream accuracy.

To reproduce the trained-projector ablation: copy `configs/tiktok_lora_v2.yaml`,
change `train_projector: false` → `train_projector: true`, and re-launch
`scripts/14_train_tiktok_lora_v2.sh --config <your_copy.yaml>`.

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
├── 01_download_blip2.py              # fetch + extract vision+qformer bundle
├── 02_smoke_test.py                  # 20-step loss-drop gate
├── 03_train_stage1.sh                # launch stage 1
├── 04_train_stage2.sh                # launch stage 2 (mix665k)
├── 05_cache_features.py              # precompute frozen ViT-G + Q-Former outputs
├── 06_cache_stage2_features.sh       # stage-2 feature cache launcher
│
│  # — TikTok v2 (canonical, with transcripts) —
├── 10_download_kaggle_dataset.py     # kaggle CLI wrapper for jobisaacong/tiktok-sludge-dataset-500
├── 10b_download_stage1_from_hf.py    # extract stage-1 Linear from alpharomercoma/vqwen-qformer-pretrain
├── 11_extract_tiktok_1fps_v2.py      # ffmpeg 1-fps frame extraction over Kaggle layout
├── 12_build_tiktok_convs_v2.py       # conv builder with "Audio transcript:" preamble
├── 13_cache_tiktok_v2.sh             # cache Q-Former features for v2 train set
├── 14_train_tiktok_lora_v2.sh        # TikTok-with-transcript LoRA launcher
│
│  # — ablation: vision-only (reproduces alpharomercoma/vqwen-qformer-tiktok) —
├── 11_extract_tiktok_1fps_ablation.py
├── 12_build_tiktok_convs_ablation.py
├── 13_cache_tiktok_ablation.sh
├── 14_train_tiktok_lora_ablation.sh
│
│  # — distillation utilities (used only by the ablation path) —
├── 15_download_teacher.py            # Qwen3-VL-30B-A3B-Instruct
├── 16_label_with_teacher.py          # batched teacher labeling (bs=8)
├── 16b_label_test.py                 # label held-out test split with same teacher
├── 18_download_judge.py              # Gemma-3-27b-it judge
├── 19_cross_compare.py               # Gemma judges teacher↔GT disputes
│
├── 20_eval_tiktok_test.py            # classify eval on test
├── 22_eval_dual_benchmark.py         # eval vs both raw GT and cleaned labels (--with_transcript)
├── 23_ab_test_batching.py            # batched-vs-serial quality check
├── 24_ab_test_bs_sweep.py            # find optimal batch size on H200
└── 30_export_hf.py                   # build + save Blip2ForConditionalGeneration

configs/
├── stage1.yaml                                # LLaVA-Pretrain alignment
├── stage2.yaml                                # mix665k instruction tuning
├── tiktok_lora_v2.yaml                        # canonical (with transcripts)
└── tiktok_lora_ablation_no_transcript.yaml    # baseline (vision-only)
```

## Reproducing the TikTok pipeline (v2, canonical)

```bash
# 1. Install deps
uv sync

# 2. Download frozen BLIP-2 components (~1.6 GB after extraction)
python scripts/01_download_blip2.py

# 3. Download Kaggle TikTok sludge dataset (~30 GB) — needs ~/.kaggle/kaggle.json
python scripts/10_download_kaggle_dataset.py

# 4. Pull stage-1 projector from HF (~252 MB shard slice, extracts ~6 MB)
python scripts/10b_download_stage1_from_hf.py

# 5. Extract 1-fps frames from Kaggle MP4s
python scripts/11_extract_tiktok_1fps_v2.py

# 6. Build multi-task conversations with audio-transcript preamble
python scripts/12_build_tiktok_convs_v2.py

# 7. Cache Q-Former features (~1 min)
bash scripts/13_cache_tiktok_v2.sh

# 8. LoRA fine-tune with transcripts (~25–35 min on H200)
bash scripts/14_train_tiktok_lora_v2.sh

# 9. Dual-benchmark eval (with transcripts at inference)
python scripts/22_eval_dual_benchmark.py \
    --checkpoint checkpoints/tiktok-lora-v2 \
    --manifest data/tiktok_v2/frames_manifest.jsonl \
    --frames_root data/tiktok_v2/frames \
    --text_root data/tiktok_v2/kaggle_root/text \
    --with_transcript --tag v8

# 10. Export to HF
python scripts/30_export_hf.py --test_generate --test_image <path> --force
```

## Ablations

### Vision-only (reproduces `alpharomercoma/vqwen-qformer-tiktok`)

This is the original pipeline that fine-tuned **without** the audio modality.
Useful as a baseline row in the multimodal-advantage analysis. It requires the
legacy v13 payload at `/home/alpha/vqwen/data/tiktok_sludge_v13/` and the
teacher-distillation artifacts (Qwen3-VL teacher + Gemma-3 judge).

```bash
python scripts/11_extract_tiktok_1fps_ablation.py
python scripts/15_download_teacher.py
python scripts/18_download_judge.py
python scripts/16_label_with_teacher.py --frames_per_video 5 --batch_size 8
python scripts/16b_label_test.py
python scripts/19_cross_compare.py
python scripts/12_build_tiktok_convs_ablation.py
bash scripts/13_cache_tiktok_ablation.sh
bash scripts/14_train_tiktok_lora_ablation.sh
python scripts/22_eval_dual_benchmark.py \
    --checkpoint checkpoints/tiktok-lora-ablation --tag ablation
```

## Hardware

Single NVIDIA H200 141 GB. All runs use bf16, SDPA attention, fused AdamW, Liger-Kernel (for stage-2 Qwen3-4B). Feature caching + batched teacher inference bring the teacher-labeling pass from 10 h (serial) to ~2 h (bs=8), at 97.5 % decision-parity with serial.

## License

Apache 2.0 for training / inference code. Base models retain original licenses: `Salesforce/blip2-opt-2.7b` (BSD-3), `Qwen/Qwen3-4B` (Apache 2.0), `google/gemma-3-27b-it` (Gemma).

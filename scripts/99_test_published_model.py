"""Standalone test: load alpharomercoma/vqwen-qformer-tiktok from HF Hub and
eval on the held-out 300-video TikTok test split.

No access to project code — uses only stock transformers + pillow + datasets/hub.

Outputs:
  results/summary.json             # aggregated accuracy + per-layout breakdown
  results/per_frame/<video_id>.json # one file per frame with all 3 Q&A outputs
                                     + ground truth + hit/miss
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

MODEL_ID = "alpharomercoma/vqwen-qformer-tiktok"
# Point to a local frames dir that matches frames_manifest.jsonl layout
# (e.g. {batch}/{video_id}_f{idx}.jpg). The manifest is published under
# labels/ in the Kaggle dataset (jobisaacong/tiktok-sludge-dataset-500).
FRAMES_ROOT = Path(os.environ.get("FRAMES_ROOT", "data/tiktok_sludge_frames"))
MANIFEST = Path(os.environ.get("FRAMES_MANIFEST", FRAMES_ROOT / "frames_manifest.jsonl"))

CLASSIFY_PROMPT = ("Does this image show two or more unrelated visual scenes displayed at the same time "
                   "(split-screen, picture-in-picture, or collage)? Answer with only 'yes' or 'no'.")
LAYOUT_PROMPT = "What layout type is shown in this image? Reply with one word or phrase."
DESCRIBE_PROMPT = "Describe the video this frame is from."
EXPLAIN_PROMPT_SLUDGE = "Explain why this is sludge content."
EXPLAIN_PROMPT_NONSLUDGE = "Explain why this is not sludge content."


def parse_yes_no(text: str):
    t = text.strip().lower()
    first = (t.split() or [""])[0].rstrip(".,!?:;")
    if first in ("yes", "y", "true"): return True
    if first in ("no", "n", "false"): return False
    if "yes" in t[:30] and "no " not in t[: t.find("yes") + 3]: return True
    if "no" in t[:30]: return False
    return None


def ask(model, processor, image, question: str, max_new: int, dtype):
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=processor.tokenizer.pad_token_id,
                                  eos_token_id=processor.tokenizer.eos_token_id)
    return processor.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=MODEL_ID)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--output_dir", default="results")
    p.add_argument("--limit", type=int, default=0, help="cap frames for a quick smoke run")
    p.add_argument("--include_describe", action="store_true",
                   help="also ask the describe prompt (slower, 256 new tokens)")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    out_root = Path(args.output_dir)
    per_frame_dir = out_root / "per_frame"
    per_frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] loading {args.model_id} from HF Hub ...", file=sys.stderr)
    t0 = time.time()
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_id, dtype=dtype, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    # Defensive: ensure chat_template is set even if AutoProcessor skipped it.
    if not getattr(processor, "chat_template", None):
        from huggingface_hub import hf_hub_download
        tpath = hf_hub_download(args.model_id, "chat_template.jinja")
        with open(tpath) as f:
            processor.chat_template = f.read()
        print("[eval] self-heal: loaded chat_template.jinja manually", file=sys.stderr)
    print(f"[eval] loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    # Filter manifest to held-out test split (never seen during training).
    records = []
    with open(MANIFEST) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test":
                records.append(r)
    if args.limit > 0:
        records = records[: args.limit]
    print(f"[eval] {len(records)} held-out test frames", file=sys.stderr)

    stats = Counter()
    per_layout = {}
    t0 = time.time()
    for i, r in enumerate(records):
        fp = FRAMES_ROOT / r["frame_path"]
        gt_cls = r.get("classification")
        gt_layout = (r.get("layout_category") or "").strip()

        try:
            image = Image.open(fp).convert("RGB")
            out_classify = ask(model, processor, image, CLASSIFY_PROMPT, 8, dtype)
            out_layout = ask(model, processor, image, LAYOUT_PROMPT, 12, dtype)
            out_describe = ask(model, processor, image, DESCRIBE_PROMPT, 256, dtype)
            # Explain: pick the ground-truth-paired prompt so the training-time
            # template matches (the model was trained on "...why this is sludge"
            # when sludge, and "...why this is not sludge" when not).
            explain_q = EXPLAIN_PROMPT_SLUDGE if gt_cls else EXPLAIN_PROMPT_NONSLUDGE
            out_explain = ask(model, processor, image, explain_q, 256, dtype)
        except Exception as e:
            stats["errors"] += 1
            out_classify = f"__ERROR__: {type(e).__name__}: {e}"
            out_layout = out_describe = out_explain = None
            explain_q = None

        pred_sludge = parse_yes_no(out_classify) if isinstance(out_classify, str) else None
        classify_hit = (pred_sludge == bool(gt_cls)) if (pred_sludge is not None and gt_cls is not None) else None

        rec = {
            "video_id": r["video_id"],
            "batch": r["batch"],
            "frame_path": r["frame_path"],
            "ground_truth": {
                "classification": gt_cls,
                "layout_category": gt_layout,
                "enriched_summary": r.get("enriched_summary"),
            },
            "model_outputs": {
                "classify": {"prompt": CLASSIFY_PROMPT, "reply": out_classify,
                             "parsed_sludge": pred_sludge,
                             "hit": classify_hit},
                "layout":   {"prompt": LAYOUT_PROMPT, "reply": out_layout,
                             "hit": (gt_layout.lower() in (out_layout or "").lower()
                                     or (out_layout or "").lower() in gt_layout.lower())
                                    if gt_layout and out_layout else None},
                "describe": {"prompt": DESCRIBE_PROMPT, "reply": out_describe},
                "explain":  {"prompt": explain_q, "reply": out_explain},
            },
        }

        # Save per-frame JSON
        (per_frame_dir / f"{r['video_id']}.json").write_text(json.dumps(rec, indent=2))

        # Accumulate stats
        layout_key = gt_layout or "unknown"
        if layout_key not in per_layout:
            per_layout[layout_key] = Counter()
        if pred_sludge is None:
            stats["unparseable"] += 1
            per_layout[layout_key]["unparseable"] += 1
        elif gt_cls is None:
            stats["no_gt"] += 1
        else:
            stats["covered"] += 1
            per_layout[layout_key]["covered"] += 1
            if classify_hit:
                stats["correct"] += 1
                per_layout[layout_key]["correct"] += 1

        if (i + 1) % 25 == 0:
            dt = time.time() - t0
            acc = stats["correct"] / stats["covered"] if stats["covered"] else 0.0
            print(f"[eval] {i+1}/{len(records)}  {dt/(i+1):.2f}s/frame  acc={acc:.3f} ({stats['covered']} covered)",
                  file=sys.stderr)

    summary = {
        "model_id": args.model_id,
        "n_total": len(records),
        "n_covered": stats["covered"],
        "n_correct": stats["correct"],
        "n_unparseable": stats["unparseable"],
        "n_errors": stats["errors"],
        "classify_accuracy_on_covered": (stats["correct"] / stats["covered"]) if stats["covered"] else None,
        "classify_accuracy_on_all": stats["correct"] / len(records) if records else None,
        "per_layout": {k: {"n": v["covered"], "correct": v["correct"],
                           "unparseable": v["unparseable"],
                           "acc": (v["correct"] / v["covered"]) if v["covered"] else None}
                       for k, v in per_layout.items()},
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())

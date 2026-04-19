"""Evaluate the TikTok-LoRA checkpoint on the held-out test split.

Uses the existing 1-mid-frame-per-video manifest at
  /home/alpha/vqwen/data/tiktok_sludge_frames/frames_manifest.jsonl
which spans ALL splits (train/validate/test). We filter to split=='test'
(300 videos) — this is the only split the model has never seen.

For each test frame we ask one of:
  - classification: "Does this image show split-screen/PiP/collage layout?"
  - layout:        "What layout type is shown?"
  - description:   "Describe the video this frame is from."
and compare the model's reply against the label.

Outputs:
  results/tiktok_lora_eval/<mode>_<ckpt>.jsonl
  results/tiktok_lora_eval/<mode>_<ckpt>_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vqwen_qformer.generate import generate_caption, load_trained_model      # noqa: E402
from vqwen_qformer.model import build_image_processor, build_tokenizer      # noqa: E402

FRAMES_ROOT = Path("/home/alpha/vqwen/data/tiktok_sludge_frames")
MANIFEST = FRAMES_ROOT / "frames_manifest.jsonl"

CLASSIFY_PROMPT = (
    "Does this image show two or more unrelated visual scenes displayed at the same time "
    "(split-screen, picture-in-picture, or collage)? Answer with only 'yes' or 'no'."
)
LAYOUT_PROMPT = "What layout type is shown in this image? Reply with one word or phrase."
DESCRIBE_PROMPT = "Describe the video this frame is from."


def parse_yes_no(text: str):
    t = text.strip().lower()
    first = (t.split() or [""])[0].rstrip(".,!?:;")
    if first in ("yes", "y", "true"): return True
    if first in ("no", "n", "false"): return False
    if "yes" in t[:30] and "no " not in t[: t.find("yes") + 3]: return True
    if "no" in t[:30]: return False
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["classify", "layout", "describe"], required=True)
    p.add_argument("--checkpoint", default="checkpoints/tiktok-lora")
    p.add_argument("--output_dir", default="results/tiktok_lora_eval")
    p.add_argument("--max_new_tokens", type=int, default=16)
    args = p.parse_args()

    ckpt = REPO_ROOT / args.checkpoint
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] loading {ckpt}", file=sys.stderr)
    model = load_trained_model(ckpt, device="cuda")
    with open(ckpt / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    tok = build_tokenizer(cfg["llm_model_path"])
    ip  = build_image_processor(cfg["blip2_bundle_path"])

    # Filter to test split
    records = []
    with open(MANIFEST) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test":
                records.append(r)
    print(f"[eval] {len(records)} held-out test frames", file=sys.stderr)

    if args.mode == "classify":
        prompt = CLASSIFY_PROMPT; max_new = 8
    elif args.mode == "layout":
        prompt = LAYOUT_PROMPT; max_new = 12
    else:
        prompt = DESCRIBE_PROMPT; max_new = args.max_new_tokens

    tag = ckpt.name
    out_file = out_dir / f"{args.mode}_{tag}.jsonl"
    stats = Counter()
    per_layout = {}      # layout -> (tp,fp,tn,fn,unparseable)
    t0 = time.time()
    with open(out_file, "w") as out:
        for i, r in enumerate(records):
            fp = FRAMES_ROOT / r["frame_path"]
            try:
                gen = generate_caption(model, tok, ip, fp,
                                       prompt="<image>\n" + prompt,
                                       max_new_tokens=max_new,
                                       do_sample=False,
                                       chat_template=True)
            except Exception as e:
                gen = f"__ERROR__: {type(e).__name__}: {e}"
                stats["errors"] += 1

            rec = {
                "video_id": r["video_id"], "batch": r["batch"],
                "frame_path": r["frame_path"],
                "generated": gen,
                "label_classification": r.get("classification"),
                "layout_category": r.get("layout_category"),
                "enriched_summary": r.get("enriched_summary"),
            }

            if args.mode == "classify":
                pred = parse_yes_no(gen)
                gt = r.get("classification")
                rec["predicted"] = pred
                layout = (r.get("layout_category") or "unknown") or "unknown"
                if layout not in per_layout:
                    per_layout[layout] = Counter()
                if pred is None:
                    stats["unparseable"] += 1
                    per_layout[layout]["unparseable"] += 1
                elif gt is None:
                    stats["no_label"] += 1
                else:
                    stats["covered"] += 1
                    hit = bool(pred) == bool(gt)
                    stats["correct"] += int(hit)
                    per_layout[layout]["covered"] += 1
                    per_layout[layout]["correct"] += int(hit)
            elif args.mode == "layout":
                gt_layout = (r.get("layout_category") or "").strip().lower()
                pred_text = gen.strip().lower()
                rec["predicted"] = pred_text
                if gt_layout:
                    stats["covered"] += 1
                    stats["correct"] += int(gt_layout in pred_text or pred_text in gt_layout)

            out.write(json.dumps(rec) + "\n")
            if (i + 1) % 25 == 0:
                dt = time.time() - t0
                extra = ""
                if stats["covered"]:
                    extra = f"  acc={stats['correct']/stats['covered']:.3f}"
                print(f"[eval] {i+1}/{len(records)} {dt/(i+1):.2f}s/frame{extra}", file=sys.stderr)

    summary = {
        "checkpoint": str(ckpt),
        "mode": args.mode,
        "n_total": len(records),
        "n_covered": stats["covered"],
        "n_correct": stats["correct"],
        "n_unparseable": stats["unparseable"],
        "n_errors": stats["errors"],
        "accuracy": (stats["correct"] / stats["covered"]) if stats["covered"] else None,
    }
    if args.mode == "classify":
        summary["per_layout"] = {
            k: {"n": v["covered"], "correct": v["correct"], "unparseable": v["unparseable"],
                "acc": (v["correct"] / v["covered"]) if v["covered"] else None}
            for k, v in per_layout.items()
        }
    with open(out_dir / f"{args.mode}_{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

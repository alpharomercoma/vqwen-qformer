"""Evaluate a checkpoint on the TikTok test split against TWO benchmarks:

1. GT benchmark: original human-labeled `classification` field
2. Cleaned benchmark: GT overridden where gemma+teacher both disagree with GT
   (i.e. for disputed frames, use gemma's judgment as the true label)
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

from vqwen_qformer.generate import generate_caption, load_trained_model  # noqa: E402
from vqwen_qformer.model import build_image_processor, build_tokenizer  # noqa: E402

FRAMES_ROOT = Path("/home/alpha/vqwen/data/tiktok_sludge_frames")
MANIFEST = FRAMES_ROOT / "frames_manifest.jsonl"
TEACHER_LABELS_TEST = REPO_ROOT / "data" / "tiktok_fps1" / "teacher_labels_test.jsonl"
CROSS_COMPARE = REPO_ROOT / "data" / "tiktok_fps1" / "cross_compare.jsonl"

CLASSIFY_PROMPT = ("Does this image show two or more unrelated visual scenes displayed at the same time "
                   "(split-screen, picture-in-picture, or collage)? Answer with only 'yes' or 'no'.")


def parse_yes_no(text: str):
    t = text.strip().lower()
    first = (t.split() or [""])[0].rstrip(".,!?:;")
    if first in ("yes", "y", "true"): return True
    if first in ("no", "n", "false"): return False
    if "yes" in t[:30]: return True
    if "no" in t[:30]: return False
    return None


def build_cleaned_labels():
    """Override GT where teacher + gemma consensus disagree with it."""
    cleaned = {}
    # Start with GT
    with open(MANIFEST) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test":
                cleaned[r["video_id"]] = bool(r.get("classification"))
    # For disputed test frames, apply gemma's judgment
    overrides = 0
    with open(CROSS_COMPARE) as f:
        for l in f:
            r = json.loads(l)
            if r.get("split_source") != "test": continue
            judge = (r.get("judge_sludge") or "").lower()
            if not judge: continue
            judged = judge.startswith("y")
            if cleaned.get(r["video_id"]) != judged:
                cleaned[r["video_id"]] = judged
                overrides += 1
    print(f"[cleaned] {overrides} GT labels overridden by gemma judgment", file=sys.stderr)
    return cleaned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/tiktok-lora")
    p.add_argument("--output_dir", default="results/dual_eval")
    p.add_argument("--tag", default="v6")
    args = p.parse_args()

    ckpt = REPO_ROOT / args.checkpoint
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned = build_cleaned_labels()

    print(f"[eval] loading {ckpt}", file=sys.stderr)
    model = load_trained_model(ckpt, device="cuda")
    with open(ckpt / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    tok = build_tokenizer(cfg["llm_model_path"])
    ip  = build_image_processor(cfg["blip2_bundle_path"])

    records = []
    with open(MANIFEST) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test":
                records.append(r)

    stats_gt = Counter(); stats_cleaned = Counter()
    out_file = out_dir / f"{args.tag}_per_frame.jsonl"
    t0 = time.time()
    with open(out_file, "w") as out:
        for i, r in enumerate(records):
            fp = FRAMES_ROOT / r["frame_path"]
            gen = generate_caption(model, tok, ip, fp,
                                   prompt="<image>\n" + CLASSIFY_PROMPT,
                                   max_new_tokens=8, do_sample=False,
                                   chat_template=True)
            pred = parse_yes_no(gen)
            gt_cls = bool(r.get("classification"))
            cl_cls = cleaned.get(r["video_id"], gt_cls)
            rec = {
                "video_id": r["video_id"],
                "gt": gt_cls, "cleaned": cl_cls, "pred": pred,
                "reply": gen, "gt_hit": pred == gt_cls if pred is not None else None,
                "cleaned_hit": pred == cl_cls if pred is not None else None,
            }
            out.write(json.dumps(rec) + "\n")
            if pred is None:
                stats_gt["unparseable"] += 1; stats_cleaned["unparseable"] += 1
            else:
                stats_gt["covered"] += 1; stats_cleaned["covered"] += 1
                if pred == gt_cls: stats_gt["correct"] += 1
                if pred == cl_cls: stats_cleaned["correct"] += 1

            if (i + 1) % 50 == 0:
                dt = time.time() - t0
                gt_acc = stats_gt["correct"] / max(1, stats_gt["covered"])
                cl_acc = stats_cleaned["correct"] / max(1, stats_cleaned["covered"])
                print(f"[eval] {i+1}/{len(records)} {dt/(i+1):.2f}s  GT={gt_acc:.3f}  cleaned={cl_acc:.3f}",
                      file=sys.stderr)

    summary = {
        "checkpoint": str(ckpt),
        "n_total": len(records),
        "n_unparseable": stats_gt["unparseable"],
        "gt_correct": stats_gt["correct"], "gt_acc": stats_gt["correct"] / max(1, stats_gt["covered"]),
        "cleaned_correct": stats_cleaned["correct"], "cleaned_acc": stats_cleaned["correct"] / max(1, stats_cleaned["covered"]),
        "overrides_applied": sum(1 for r in records if cleaned.get(r["video_id"]) != bool(r.get("classification"))),
    }
    with open(out_dir / f"{args.tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())

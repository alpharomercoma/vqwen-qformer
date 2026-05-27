"""Evaluate a checkpoint on the TikTok test split against TWO benchmarks:

1. GT benchmark: original human-labeled `classification` field
2. Cleaned benchmark: GT overridden where gemma+teacher both disagree with GT
   (i.e. for disputed frames, use gemma's judgment as the true label).
   Skipped automatically when the cross_compare.jsonl artifact is absent
   (e.g. the v2 Kaggle pipeline does not produce one).

Supports both the legacy vision-only ablation dataset (default paths below)
and the v2 with-transcript Kaggle dataset (pass `--manifest` /
`--frames_root` / `--text_root` / `--with_transcript`).
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

# Legacy ablation defaults — used when --manifest / --frames_root not given.
DEFAULT_FRAMES_ROOT = Path("/home/alpha/vqwen/data/tiktok_sludge_frames")
DEFAULT_MANIFEST = DEFAULT_FRAMES_ROOT / "frames_manifest.jsonl"
DEFAULT_CROSS_COMPARE = REPO_ROOT / "data" / "tiktok_fps1" / "cross_compare.jsonl"

CLASSIFY_PROMPT = ("Does this image show two or more unrelated visual scenes displayed at the same time "
                   "(split-screen, picture-in-picture, or collage)? Answer with only 'yes' or 'no'.")
MAX_TRANSCRIPT_CHARS = 600


def _load_transcript(text_json_path: Path) -> str:
    """Mirror of `12_build_tiktok_convs_v2._load_transcript` for inference."""
    if not text_json_path.exists():
        return ""
    try:
        with open(text_json_path) as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    if isinstance(obj, dict):
        t = obj.get("text")
        if isinstance(t, str) and t.strip(): return t.strip()
        segs = obj.get("segments")
        if isinstance(segs, list):
            parts = [s["text"].strip() for s in segs
                     if isinstance(s, dict) and isinstance(s.get("text"), str) and s["text"].strip()]
            if parts: return " ".join(parts)
    elif isinstance(obj, list):
        parts = [s["text"].strip() for s in obj
                 if isinstance(s, dict) and isinstance(s.get("text"), str) and s["text"].strip()]
        if parts: return " ".join(parts)
    return ""


def _truncate(t: str, n: int = MAX_TRANSCRIPT_CHARS) -> str:
    if len(t) <= n:
        return t
    cut = t.rfind(" ", 0, n)
    return (t[:cut] if cut > 0 else t[:n]).rstrip() + "…"


def parse_yes_no(text: str):
    t = text.strip().lower()
    first = (t.split() or [""])[0].rstrip(".,!?:;")
    if first in ("yes", "y", "true"): return True
    if first in ("no", "n", "false"): return False
    if "yes" in t[:30]: return True
    if "no" in t[:30]: return False
    return None


def build_cleaned_labels(manifest: Path, cross_compare: Path):
    """Override GT where teacher + gemma consensus disagree with it.

    Returns (cleaned_dict, used_cross_compare_bool). If `cross_compare` is
    missing we fall back to plain GT and report that to the caller.
    """
    cleaned = {}
    with open(manifest) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test":
                cleaned[r["video_id"]] = bool(r.get("classification"))
    if not cross_compare.exists():
        print(f"[cleaned] no cross_compare.jsonl at {cross_compare}; "
              "skipping cleaned benchmark.", file=sys.stderr)
        return cleaned, False
    overrides = 0
    with open(cross_compare) as f:
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
    return cleaned, True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/tiktok-lora")
    p.add_argument("--output_dir", default="results/dual_eval")
    p.add_argument("--tag", default="v6")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                   help="Path to frames_manifest.jsonl (v1 ablation or v2 Kaggle).")
    p.add_argument("--frames_root", default=str(DEFAULT_FRAMES_ROOT),
                   help="Root directory whose subpaths are referenced by manifest['frame_path']. "
                        "For v2 use data/tiktok_v2/frames.")
    p.add_argument("--cross_compare", default=str(DEFAULT_CROSS_COMPARE),
                   help="Cross-compare artifact used for the 'cleaned' benchmark. "
                        "Skipped when missing.")
    p.add_argument("--text_root", default=None,
                   help="Root directory holding per-video Whisper transcripts "
                        "(e.g. data/tiktok_v2/kaggle_root/text). Required when "
                        "--with_transcript is set.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--with_transcript", dest="with_transcript", action="store_true",
                   help="Prepend the per-video transcript to the classify prompt at inference.")
    g.add_argument("--no_transcript", dest="with_transcript", action="store_false",
                   help="Disable transcript prepending (back-compat default).")
    p.set_defaults(with_transcript=False)
    args = p.parse_args()

    if args.with_transcript and not args.text_root:
        print("[eval] --with_transcript requires --text_root", file=sys.stderr)
        return 2

    manifest = Path(args.manifest)
    frames_root = Path(args.frames_root)
    text_root = Path(args.text_root) if args.text_root else None
    cross_compare = Path(args.cross_compare)

    ckpt = REPO_ROOT / args.checkpoint
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned, used_cleaned = build_cleaned_labels(manifest, cross_compare)

    print(f"[eval] loading {ckpt}", file=sys.stderr)
    model = load_trained_model(ckpt, device="cuda")
    with open(ckpt / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    tok = build_tokenizer(cfg["llm_model_path"])
    ip  = build_image_processor(cfg["blip2_bundle_path"])

    records = []
    with open(manifest) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test":
                records.append(r)

    transcript_cache: dict[str, str] = {}

    def _prompt_for(rec) -> str:
        if not args.with_transcript:
            return "<image>\n" + CLASSIFY_PROMPT
        vid = rec["video_id"]
        if vid not in transcript_cache:
            tp = rec.get("transcript_path")
            if not tp:
                # Fall back to <batch>/<video_id>.json under text_root.
                batch = rec.get("batch") or Path(rec["frame_path"]).parts[0]
                tp = f"{batch}/{vid}.json"
            transcript_cache[vid] = _truncate(_load_transcript(text_root / tp))
        t = transcript_cache[vid]
        head = f"Audio transcript: {t}\n" if t else ""
        return f"{head}<image>\n{CLASSIFY_PROMPT}"

    stats_gt = Counter(); stats_cleaned = Counter()
    out_file = out_dir / f"{args.tag}_per_frame.jsonl"
    t0 = time.time()
    with open(out_file, "w") as out:
        for i, r in enumerate(records):
            fp = frames_root / r["frame_path"]
            gen = generate_caption(model, tok, ip, fp,
                                   prompt=_prompt_for(r),
                                   max_new_tokens=8, do_sample=False,
                                   chat_template=True)
            pred = parse_yes_no(gen)
            gt_cls = bool(r.get("classification"))
            cl_cls = cleaned.get(r["video_id"], gt_cls)
            rec_out = {
                "video_id": r["video_id"],
                "gt": gt_cls, "cleaned": cl_cls, "pred": pred,
                "reply": gen, "gt_hit": pred == gt_cls if pred is not None else None,
                "cleaned_hit": pred == cl_cls if pred is not None else None,
                "with_transcript": args.with_transcript,
            }
            out.write(json.dumps(rec_out) + "\n")
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
        "with_transcript": args.with_transcript,
        "used_cleaned_benchmark": used_cleaned,
    }
    if used_cleaned:
        summary["cleaned_correct"] = stats_cleaned["correct"]
        summary["cleaned_acc"] = stats_cleaned["correct"] / max(1, stats_cleaned["covered"])
        summary["overrides_applied"] = sum(
            1 for r in records if cleaned.get(r["video_id"]) != bool(r.get("classification"))
        )
    with open(out_dir / f"{args.tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())

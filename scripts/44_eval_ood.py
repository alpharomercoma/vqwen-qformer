"""Evaluate one or more checkpoints on the OOD live-TikTok set.

Reads:
  data/tiktok_v2/ood_live/manifest.jsonl              (one row per video, frames + text_path)
  data/tiktok_v2/ood_live/gemma4_video_labels.jsonl   (Gemma-4-31B-IT majority vote)

Runs each checkpoint -> video-level majority vote -> metrics against Gemma-4
labels (auxiliary gold).

Outputs:
  data/tiktok_v2/ood_live/ood_eval_<tag>.jsonl   (per-video predictions)
  data/tiktok_v2/ood_live/ood_eval_summary.json  (head-to-head summary)
"""

from __future__ import annotations

import argparse
import json
import re
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vqwen_qformer.generate import generate_caption, load_trained_model  # noqa: E402
from vqwen_qformer.model import build_image_processor, build_tokenizer  # noqa: E402

OOD_ROOT = REPO_ROOT / "data" / "tiktok_v2" / "ood_live"

CLASSIFY_PROMPT = ("Does this image show two or more unrelated visual scenes displayed at the same time "
                   "(split-screen, picture-in-picture, or collage)? Answer with only 'yes' or 'no'.")
MAX_TRANSCRIPT_CHARS = 600


def _truncate(t: str, n: int = MAX_TRANSCRIPT_CHARS) -> str:
    if len(t) <= n:
        return t
    cut = t.rfind(" ", 0, n)
    return (t[:cut] if cut > 0 else t[:n]).rstrip() + "…"


def parse_yes_no(text: str):
    s = (text or "").strip().lower()
    if not s:
        return None
    s = re.sub(r"[^\w\s]", " ", s)
    parts = s.split()
    if not parts:
        return None
    first = parts[0]
    if first in ("yes", "y", "yeah", "yep"):
        return True
    if first in ("no", "n", "nope"):
        return False
    if "yes" in parts[:5]:
        return True
    if "no" in parts[:5]:
        return False
    return None


def eval_checkpoint(ckpt_path: Path, manifest_records: list[dict],
                    with_transcript: bool, out_path: Path) -> list[dict]:
    print(f"[ood-eval] loading {ckpt_path} (with_transcript={with_transcript})", file=sys.stderr)
    model = load_trained_model(ckpt_path, device="cuda")
    with open(ckpt_path / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    tok = build_tokenizer(cfg["llm_model_path"])
    ip = build_image_processor(cfg["blip2_bundle_path"])

    transcripts: dict[str, str] = {}
    for r in manifest_records:
        if not with_transcript:
            continue
        tp = OOD_ROOT / r["text_path"]
        if not tp.exists():
            transcripts[r["video_id"]] = ""
            continue
        text = (json.loads(tp.read_text()).get("text") or "").strip()
        transcripts[r["video_id"]] = _truncate(text)

    results = []
    t0 = time.time()
    with out_path.open("w") as f_out:
        for i, r in enumerate(manifest_records, 1):
            vid = r["video_id"]
            transcript = transcripts.get(vid, "") if with_transcript else ""
            head = f"Audio transcript: {transcript}\n" if transcript else ""
            prompt = f"{head}<image>\n{CLASSIFY_PROMPT}"

            per_frame_preds = []
            for frame in r["frames"]:
                fp = OOD_ROOT / frame["frame_path"]
                gen = generate_caption(model, tok, ip, fp,
                                       prompt=prompt,
                                       max_new_tokens=8, do_sample=False,
                                       chat_template=True)
                pred = parse_yes_no(gen)
                per_frame_preds.append({"frame_index": frame["frame_index"],
                                        "time_s": frame["time_s"],
                                        "pred": pred, "reply": gen})
            votes = [p["pred"] for p in per_frame_preds if p["pred"] is not None]
            video_label = (sum(votes) > len(votes) / 2) if votes else None
            rec_out = {
                "video_id": vid,
                "url": r["url"],
                "with_transcript": with_transcript,
                "video_pred": video_label,
                "n_votes": len(votes),
                "per_frame": per_frame_preds,
            }
            f_out.write(json.dumps(rec_out) + "\n")
            results.append(rec_out)
            if i % 10 == 0:
                print(f"[ood-eval] {i}/{len(manifest_records)}  {(time.time()-t0)/i:.2f}s/vid",
                      file=sys.stderr)

    # Free model memory before next checkpoint
    del model
    torch.cuda.empty_cache()
    return results


def bootstrap_ci(records: list[dict], key_pred: str, key_truth: str,
                 n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    parseable = [r for r in records if r[key_pred] is not None and r[key_truth] is not None]
    if not parseable:
        return 0.0, 0.0, 0.0
    n = len(parseable)
    point = sum(1 for r in parseable if r[key_pred] == r[key_truth]) / n
    rng = random.Random(seed)
    accs = []
    for _ in range(n_boot):
        sample = [parseable[rng.randint(0, n - 1)] for _ in range(n)]
        accs.append(sum(1 for r in sample if r[key_pred] == r[key_truth]) / n)
    accs.sort()
    return point, accs[int(0.025 * n_boot)], accs[int(0.975 * n_boot) - 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(OOD_ROOT / "manifest.jsonl"))
    p.add_argument("--gemma4-labels", default=str(OOD_ROOT / "gemma4_video_labels.jsonl"))
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One or more checkpoint paths (e.g. checkpoints/tiktok-lora)")
    p.add_argument("--tags", nargs="+", required=True,
                   help="Tag for each checkpoint, same length as --checkpoints")
    p.add_argument("--out-dir", default=str(OOD_ROOT))
    args = p.parse_args()

    assert len(args.checkpoints) == len(args.tags), "checkpoints/tags length mismatch"

    records = [json.loads(l) for l in Path(args.manifest).open()]
    print(f"[ood-eval] manifest: {len(records)} videos", file=sys.stderr)

    gemma_labels = {r["video_id"]: r for r in (json.loads(l) for l in Path(args.gemma4_labels).open())}
    print(f"[ood-eval] gemma4 labels: {len(gemma_labels)} videos", file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run every (checkpoint, transcript-mode) combination
    all_eval = []
    for ckpt_path, tag in zip(args.checkpoints, args.tags):
        ckpt_p = (REPO_ROOT / ckpt_path).resolve()
        for with_t in (True, False):
            mode = "with_transcript" if with_t else "no_transcript"
            out = out_dir / f"ood_eval_{tag}_{mode}.jsonl"
            eval_records = eval_checkpoint(ckpt_p, records, with_t, out)
            all_eval.append({"tag": tag, "with_transcript": with_t, "records": eval_records,
                             "out_path": str(out)})

    # Build the summary head-to-head table
    summary = {"n_total": len(records),
               "n_gemma_labeled": sum(1 for r in records if gemma_labels.get(r["video_id"], {}).get("gemma4_is_sludge") is not None),
               "results": []}
    for run in all_eval:
        joined = []
        for r in run["records"]:
            gemma = gemma_labels.get(r["video_id"], {}).get("gemma4_is_sludge")
            joined.append({"pred": r["video_pred"], "gold": gemma})
        n_parseable = sum(1 for r in joined if r["pred"] is not None and r["gold"] is not None)
        if n_parseable == 0:
            continue
        point, lo, hi = bootstrap_ci(joined, "pred", "gold")
        c = Counter((r["gold"], r["pred"]) for r in joined if r["pred"] is not None and r["gold"] is not None)
        summary["results"].append({
            "tag": run["tag"], "with_transcript": run["with_transcript"],
            "n_parseable": n_parseable,
            "acc_vs_gemma": point, "ci_low": lo, "ci_high": hi,
            "confusion": {f"({k[0]},{k[1]})": v for k, v in sorted(c.items())},
            "out_path": run["out_path"],
        })

    sumpath = out_dir / "ood_eval_summary.json"
    sumpath.write_text(json.dumps(summary, indent=2))
    print(f"[ood-eval] wrote {sumpath}", file=sys.stderr)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

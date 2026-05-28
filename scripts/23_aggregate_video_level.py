"""Aggregate per-frame eval output to a video-level number with bootstrap CI.

The dual-benchmark eval (`scripts/22_eval_dual_benchmark.py`) writes per-frame
`{video_id, gt, cleaned, pred, ...}` records. For a sludge detector the right
unit is the *video* — sludge is a video-level property, and per-frame accuracy
is artificially inflated by correlated frames from the same video. This script
collapses frames into videos by majority vote and reports:

  - video-level accuracy (vs ground-truth and vs cleaned labels)
  - per-frame accuracy (cross-check vs the existing summary)
  - 95 % bootstrap CI on the video-level number (1,000 resamples, percentile)
  - confusion-matrix counts

Run:
  /home/alpha/vqwen-qformer/.venv/bin/python scripts/23_aggregate_video_level.py \
    results/dual_eval/with_transcript_per_frame.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple


def majority_vote(preds: List[bool]) -> bool:
    if not preds:
        return False
    yes = sum(1 for p in preds if p)
    return yes > len(preds) / 2


def aggregate(records: List[dict]) -> Tuple[List[dict], int]:
    """Return per-video records and the count of dropped (unparseable) frames."""
    by_vid: dict[str, dict] = defaultdict(lambda: {"preds": [], "gt": None, "cleaned": None})
    dropped = 0
    for r in records:
        if r.get("pred") is None:
            dropped += 1
            continue
        v = by_vid[r["video_id"]]
        v["preds"].append(bool(r["pred"]))
        # gt/cleaned are video-level; they should be consistent across frames
        v["gt"] = bool(r["gt"]); v["cleaned"] = bool(r["cleaned"])

    out = []
    for vid, v in by_vid.items():
        out.append({
            "video_id": vid,
            "pred": majority_vote(v["preds"]),
            "gt": v["gt"],
            "cleaned": v["cleaned"],
            "n_frames": len(v["preds"]),
            "frame_yes_share": sum(v["preds"]) / max(1, len(v["preds"])),
        })
    return out, dropped


def acc(records: List[dict], key: str) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r["pred"] == r[key]) / len(records)


def bootstrap_ci(records: List[dict], key: str, n_boot: int = 1000,
                 seed: int = 42, q_lo: float = 0.025, q_hi: float = 0.975) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(records)
    if n == 0:
        return 0.0, 0.0
    accs: List[float] = []
    for _ in range(n_boot):
        sample = [records[rng.randint(0, n - 1)] for _ in range(n)]
        accs.append(acc(sample, key))
    accs.sort()
    return accs[int(q_lo * n_boot)], accs[int(q_hi * n_boot) - 1]


def confusion(records: List[dict], key: str) -> Counter:
    c: Counter = Counter()
    for r in records:
        t = r[key]; p = r["pred"]
        c[(t, p)] += 1
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("per_frame_jsonl", type=Path,
                   help="Path to <tag>_per_frame.jsonl from 22_eval_dual_benchmark.py")
    p.add_argument("--boot", type=int, default=1000)
    args = p.parse_args()

    records = [json.loads(l) for l in args.per_frame_jsonl.open()]
    print(f"[agg] loaded {len(records)} per-frame records from {args.per_frame_jsonl}")

    videos, dropped_frames = aggregate(records)
    print(f"[agg] {len(videos)} videos; {dropped_frames} unparseable frames dropped")

    # Frame-level (sanity check vs the existing summary)
    parseable = [r for r in records if r.get("pred") is not None]
    fl_gt = acc(parseable, "gt"); fl_cl = acc(parseable, "cleaned")
    print(f"[frame-level] vs GT     : {fl_gt*100:.2f}%  ({sum(1 for r in parseable if r['pred']==r['gt'])}/{len(parseable)})")
    print(f"[frame-level] vs cleaned: {fl_cl*100:.2f}%")

    # Video-level (primary)
    vl_gt = acc(videos, "gt"); vl_cl = acc(videos, "cleaned")
    lo_gt, hi_gt = bootstrap_ci(videos, "gt", n_boot=args.boot)
    lo_cl, hi_cl = bootstrap_ci(videos, "cleaned", n_boot=args.boot)
    print(f"[video-level] vs GT     : {vl_gt*100:.2f}%  (95% CI {lo_gt*100:.2f}-{hi_gt*100:.2f})  n={len(videos)}")
    print(f"[video-level] vs cleaned: {vl_cl*100:.2f}%  (95% CI {lo_cl*100:.2f}-{hi_cl*100:.2f})")

    # Confusion matrix (video-level, vs GT)
    c = confusion(videos, "gt")
    print(f"[confusion vs GT]  TP={c[(True,True)]}  FP={c[(False,True)]}  TN={c[(False,False)]}  FN={c[(True,False)]}")

    # Optional: persist summary
    out = args.per_frame_jsonl.with_name(args.per_frame_jsonl.stem.replace("_per_frame", "_video_level_summary") + ".json")
    summary = {
        "source": str(args.per_frame_jsonl),
        "n_videos": len(videos),
        "n_frames_total": len(records),
        "n_frames_dropped": dropped_frames,
        "frame_level": {"gt_acc": fl_gt, "cleaned_acc": fl_cl},
        "video_level": {
            "gt_acc": vl_gt, "cleaned_acc": vl_cl,
            "gt_ci_lo": lo_gt, "gt_ci_hi": hi_gt,
            "cleaned_ci_lo": lo_cl, "cleaned_ci_hi": hi_cl,
        },
        "confusion_gt": {f"({k[0]},{k[1]})": v for k, v in sorted(c.items())},
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"[agg] wrote {out}")


if __name__ == "__main__":
    sys.exit(main())

"""Extract ~1-fps frames from the TikTok sludge v13 MP4s (train + validate splits).

Output layout (`data/tiktok_fps1/`):
  frames/{batch}/{video_id}_f{sec}.jpg      # one frame at ~every 1 s
  frames_manifest.jsonl                     # per-frame record with labels

`frames_manifest.jsonl` each record:
  { "video_id": "...", "batch": "Sludge_Batch_1", "frame_path": "Sludge_Batch_1/ABC_f0.jpg",
    "time_s": 0.5, "is_sludge": true, "layout_category": "horizontal",
    "summary": "...", "classification": true, "split": "train" }

We *skip* the test split here so we can hold it out for eval later.
Max 30 frames/video to bound disk + compute on long videos.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VQWEN_ROOT = Path("/home/alpha/vqwen")
V13_ROOT = VQWEN_ROOT / "data" / "tiktok_sludge_v13" / "payload"
LABELS_ROOT = VQWEN_ROOT / "data" / "tiktok_sludge"
OUT_ROOT = REPO_ROOT / "data" / "tiktok_fps1"
BATCHES = ["Sludge_Batch_1", "Sludge_Batch_2", "Non_Sludge_Batch_1", "Non_Sludge_Batch_2"]


def probe_duration(mp4: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(mp4)],
        capture_output=True, text=True,
    )
    try: return float(r.stdout.strip())
    except ValueError: return 0.0


def extract_1fps(mp4: Path, out_dir: Path, max_frames: int = 30) -> list[tuple[Path, float]]:
    """One JPEG per second. Returns [(path, timestamp_s), ...]."""
    vid = mp4.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = probe_duration(mp4)
    if dur <= 0: return []
    # centered per-second: 0.5, 1.5, 2.5, ... (skip very edge frames)
    ts = [t + 0.5 for t in range(int(dur))]
    if len(ts) > max_frames:
        # Evenly spaced subsample when very long
        step = len(ts) / max_frames
        ts = [ts[int(i * step)] for i in range(max_frames)]
    paths = []
    for t in ts:
        out = out_dir / f"{vid}_f{int(t)}.jpg"
        if out.exists() and out.stat().st_size > 0:
            paths.append((out, t))
            continue
        subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(mp4),
             "-frames:v", "1", "-q:v", "3", "-y", str(out)],
            check=False,
        )
        if out.exists() and out.stat().st_size > 0:
            paths.append((out, t))
    return paths


def build_label_lookup() -> dict:
    lookup: dict = {}
    for sp in ("train", "validate", "test"):
        for r in json.load(open(LABELS_ROOT / "split" / f"{sp}.json")):
            lookup[r["id"]] = {
                "classification": r["classification"],
                "batch_name": r["batch_name"],
                "split": sp,
            }
    for line in open(LABELS_ROOT / "enriched_classifications.jsonl"):
        r = json.loads(line)
        vid = r["video_id"]
        if vid in lookup:
            lookup[vid]["summary"] = r.get("summary", "")
            lookup[vid]["layout_category"] = r.get("layout_category", "")
            lookup[vid]["is_sludge"] = r.get("is_sludge", None)
    return lookup


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max_frames", type=int, default=30)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--splits", nargs="+", default=["train", "validate"])
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    labels = build_label_lookup()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_ROOT / "frames_manifest.jsonl"

    jobs = []
    for batch in BATCHES:
        src = V13_ROOT / "video" / batch
        out = OUT_ROOT / "frames" / batch
        mp4s = sorted(src.glob("*.mp4"))
        # Keep only train+validate; skip test
        mp4s = [m for m in mp4s if labels.get(m.stem, {}).get("split") in args.splits]
        if args.limit > 0: mp4s = mp4s[: args.limit]
        for mp4 in mp4s:
            jobs.append((mp4, out, args.max_frames))

    print(f"[fps1] extracting up to {args.max_frames} frames/video from {len(jobs)} "
          f"videos ({'+'.join(args.splits)} splits)...")
    t0 = time.time()
    n_done = n_failed = n_frames = 0
    with open(manifest_path, "w") as mf, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_1fps, mp4, out, mf_): mp4 for mp4, out, mf_ in jobs}
        for fut in as_completed(futs):
            mp4 = futs[fut]
            try:
                paths = fut.result()
            except Exception as e:
                print(f"[fps1] {mp4.name} ERROR: {e}", flush=True)
                n_failed += 1; continue
            if not paths:
                n_failed += 1; continue
            vid = mp4.stem
            batch = mp4.parent.name
            lab = labels.get(vid, {})
            for fp, t in paths:
                rec = {
                    "video_id": vid, "batch": batch,
                    "frame_path": str(fp.relative_to(OUT_ROOT / "frames")),
                    "time_s": round(t, 2),
                    "classification": lab.get("classification"),
                    "is_sludge": lab.get("is_sludge"),
                    "layout_category": lab.get("layout_category"),
                    "summary": lab.get("summary"),
                    "split": lab.get("split"),
                }
                mf.write(json.dumps(rec) + "\n")
                n_frames += 1
            n_done += 1
            if n_done % 200 == 0:
                print(f"[fps1] {n_done}/{len(jobs)} videos done ({n_frames} frames) "
                      f"in {time.time()-t0:.0f}s", flush=True)

    print(f"[fps1] done: {n_done} videos, {n_frames} frames, {n_failed} failed. "
          f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

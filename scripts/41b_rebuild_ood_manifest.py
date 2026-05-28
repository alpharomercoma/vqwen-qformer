"""Rebuild the OOD manifest from files on disk.

The download script (41_download_ood_tiktoks.py) overwrites manifest.jsonl
at the end of its run, but the run was killed mid-way (intentionally, to
stop YouTube rate-limit waste). The on-disk mp4/audio/frames artefacts
from the killed run remain; this script regenerates a manifest by
scanning the filesystem and joining each video_id against the URL pools
to recover metadata.

Outputs:
  data/tiktok_v2/ood_live/manifest.jsonl   (rebuilt, supersedes the stale one)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OOD_ROOT = REPO_ROOT / "data" / "tiktok_v2" / "ood_live"


def video_duration(mp4: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp4)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def main():
    # Build a vid_id -> metadata lookup from every pool file
    pool_lookup: dict[str, dict] = {}
    for f in ["reddit_url_pool.jsonl", "youtube_url_pool.jsonl",
              "youtube_url_pool_v2.jsonl", "tiktok_creator_pool.jsonl",
              "combined_url_pool.jsonl", "combined_url_pool_v2.jsonl"]:
        fp = OOD_ROOT / f
        if not fp.exists():
            continue
        for line in fp.open():
            r = json.loads(line)
            # Pool entries don't carry the canonical yt-dlp ID, only the URL.
            # We'll match on the FILENAME video_id later (yt-dlp's resolved id).
            # Keep candidate metadata keyed by URL for late join.
            pool_lookup[r["url"]] = r

    video_dir = OOD_ROOT / "video"
    audio_dir = OOD_ROOT / "audio"
    frames_dir_root = OOD_ROOT / "frames"

    # Build per-video manifest rows
    rows = []
    for mp4 in sorted(video_dir.glob("*.mp4")):
        vid = mp4.stem
        wav = audio_dir / f"{vid}.wav"
        frames_dir = frames_dir_root / vid

        if not wav.exists() or not frames_dir.exists():
            continue

        frames = []
        for jpg in sorted(frames_dir.glob("f*.jpg")):
            try:
                idx = int(jpg.stem[1:])
            except ValueError:
                continue
            # Time = (idx / (n-1)) * duration. We don't know duration without ffprobe.
            # Use placeholder; the eval doesn't strictly need precise times.
            frames.append({
                "frame_path": str(jpg.relative_to(OOD_ROOT)),
                "frame_index": idx,
                "time_s": float(idx),
            })
        if not frames:
            continue

        # Try to find the canonical URL/source via vid match
        candidate = None
        for url, meta in pool_lookup.items():
            if vid in url:
                candidate = meta
                break

        rows.append({
            "video_id": vid,
            "url": (candidate["url"] if candidate else f"unknown://{vid}"),
            "source_pool": (candidate or {}).get("source_pool", "unknown"),
            "subreddit": (candidate or {}).get("subreddit"),
            "creator": (candidate or {}).get("creator"),
            "platform": (candidate or {}).get("platform"),
            "candidate_label_sludge": (candidate or {}).get("candidate_label_sludge"),
            "mp4_path": str(mp4.relative_to(OOD_ROOT)),
            "wav_path": str(wav.relative_to(OOD_ROOT)),
            "frames": frames,
            "n_frames": len(frames),
        })

    out = OOD_ROOT / "manifest.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Stats
    from collections import Counter
    print(f"rebuilt manifest: {len(rows)} videos -> {out}")
    print("by source_pool:", Counter(r["source_pool"] for r in rows))
    print("by candidate_label_sludge:", Counter(r["candidate_label_sludge"] for r in rows))
    n_known = sum(1 for r in rows if not r["url"].startswith("unknown://"))
    print(f"URL-matched: {n_known}/{len(rows)}")


if __name__ == "__main__":
    main()

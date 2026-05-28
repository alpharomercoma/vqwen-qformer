"""Download Reddit-sourced TikTok URLs via yt-dlp; extract 1-fps frames + 16kHz WAV.

Inputs:
  data/tiktok_v2/ood_live/reddit_url_pool.jsonl  (from 40_scrape_reddit_tiktoks.py)

Outputs:
  data/tiktok_v2/ood_live/video/<video_id>.mp4
  data/tiktok_v2/ood_live/audio/<video_id>.wav
  data/tiktok_v2/ood_live/frames/<video_id>/f<idx>.jpg   (1 fps, capped at MAX_FRAMES)
  data/tiktok_v2/ood_live/manifest.jsonl                 (one row per successfully processed video)

Skipped or failed downloads are logged to ood_live/failures.jsonl. We dedupe
on the TikTok video numeric id pulled out of the URL.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
OOD_ROOT = REPO_ROOT / "data" / "tiktok_v2" / "ood_live"
VIDEO_RE = re.compile(r"tiktok\.com/(?:@[^/]+/)?video/(\d+)", re.I)
# Filter out non-video TikTok URLs (profile pages, legal links, etc.) so we
# don't waste yt-dlp invocations on URLs that obviously have no video.
NON_VIDEO_HINTS = ("/legal/", "/business/", "/discover", "/explore", "/login")

MAX_FRAMES_PER_VIDEO = 4  # match v2 train/test sampling density


def _video_id_fast(url: str) -> Optional[str]:
    """Best-effort ID guess from URL alone; yt-dlp gives the canonical ID."""
    m = VIDEO_RE.search(url)
    return m.group(1) if m else None


def _resolve_id(url: str) -> Optional[str]:
    """Ask yt-dlp for the canonical video ID; handles short URLs (tiktok.com/t/X)."""
    fast = _video_id_fast(url)
    if fast:
        return fast
    r = subprocess.run(
        ["yt-dlp", "--print", "id", "--skip-download", "--no-warnings", url],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        return None
    out = r.stdout.strip().split()
    return out[0] if out else None


def _ytdlp_download(url: str, out_mp4: Path) -> bool:
    cmd = [
        "yt-dlp",
        "-q",
        "--no-warnings",
        "--no-progress",
        "--restrict-filenames",
        "-f", "mp4/best",
        "-o", str(out_mp4),
        url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return r.returncode == 0 and out_mp4.exists() and out_mp4.stat().st_size > 1024


def _extract_audio(mp4: Path, wav: Path) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
         "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)],
        capture_output=True, timeout=60,
    )
    return r.returncode == 0 and wav.exists() and wav.stat().st_size > 1024


def _video_duration(mp4: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp4)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def _extract_frames(mp4: Path, out_dir: Path, n: int) -> list[dict]:
    """Sample n frames evenly across the clip. Returns list of {frame_path, time_s}."""
    duration = _video_duration(mp4)
    if duration <= 0:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if n == 1:
        times = [duration / 2]
    else:
        # Pin both endpoints to avoid the constant first-frame-is-black edge case.
        step = (duration - 0.5) / (n - 1)
        times = [0.25 + i * step for i in range(n)]
    for i, t in enumerate(times):
        out = out_dir / f"f{i}.jpg"
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}",
             "-i", str(mp4), "-frames:v", "1", "-q:v", "2", str(out)],
            capture_output=True, timeout=30,
        )
        if r.returncode == 0 and out.exists() and out.stat().st_size > 1024:
            rows.append({"frame_path": str(out.relative_to(OOD_ROOT)),
                         "time_s": float(round(t, 2)),
                         "frame_index": i})
    return rows


def process_one(record: dict, force: bool) -> dict:
    """Returns a manifest row on success, or a failure record."""
    url = record["url"]
    if any(h in url for h in NON_VIDEO_HINTS):
        return {"_status": "skip-non-video", "url": url}
    vid = _resolve_id(url)
    if not vid:
        return {"_status": "skip-no-id", "url": url}

    video_dir = OOD_ROOT / "video"
    audio_dir = OOD_ROOT / "audio"
    frames_dir = OOD_ROOT / "frames" / vid
    video_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    mp4 = video_dir / f"{vid}.mp4"
    wav = audio_dir / f"{vid}.wav"

    if force and mp4.exists():
        mp4.unlink()
    if not mp4.exists() and not _ytdlp_download(url, mp4):
        return {"_status": "download-failed", "url": url, "video_id": vid}

    if not wav.exists() and not _extract_audio(mp4, wav):
        return {"_status": "audio-failed", "url": url, "video_id": vid}

    frames = _extract_frames(mp4, frames_dir, MAX_FRAMES_PER_VIDEO)
    if not frames:
        return {"_status": "frames-failed", "url": url, "video_id": vid}

    return {
        "_status": "ok",
        "video_id": vid,
        "url": url,
        "subreddit": record.get("subreddit"),
        "source": record.get("source"),
        "candidate_label_sludge": record.get("candidate_label_sludge"),
        "mp4_path": str(mp4.relative_to(OOD_ROOT)),
        "wav_path": str(wav.relative_to(OOD_ROOT)),
        "frames": frames,
        "n_frames": len(frames),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default=str(OOD_ROOT / "reddit_url_pool.jsonl"))
    p.add_argument("--manifest", default=str(OOD_ROOT / "manifest.jsonl"))
    p.add_argument("--failures", default=str(OOD_ROOT / "failures.jsonl"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force-redownload", action="store_true")
    args = p.parse_args()

    pool_path = Path(args.pool)
    rows = [json.loads(l) for l in pool_path.open()]
    print(f"[ood-dl] {len(rows)} candidate URLs", file=sys.stderr)

    OOD_ROOT.mkdir(parents=True, exist_ok=True)
    ok: list[dict] = []
    fail: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, r, args.force_redownload): r for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res.get("_status") == "ok":
                ok.append(res)
            else:
                fail.append(res)
            if i % 10 == 0:
                print(f"[ood-dl] {i}/{len(rows)}  ok={len(ok)}  fail={len(fail)}", file=sys.stderr)

    with Path(args.manifest).open("w") as f:
        for r in ok:
            f.write(json.dumps(r) + "\n")
    with Path(args.failures).open("w") as f:
        for r in fail:
            f.write(json.dumps(r) + "\n")

    # Stats
    by_status = {}
    for r in fail:
        s = r["_status"]
        by_status[s] = by_status.get(s, 0) + 1
    print(f"[ood-dl] OK: {len(ok)}  FAIL: {len(fail)}", file=sys.stderr)
    print(f"[ood-dl] failure breakdown: {by_status}", file=sys.stderr)
    print(f"[ood-dl] wrote {args.manifest} and {args.failures}", file=sys.stderr)


if __name__ == "__main__":
    main()

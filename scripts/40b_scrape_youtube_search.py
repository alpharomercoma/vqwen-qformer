"""Pull YouTube Shorts URLs via yt-dlp's ytsearch for OOD evaluation.

Mirrors the paper's stated dataset collection methodology: "ethical scraping
from public TikTok and YouTube Shorts feeds, searching sludge-related
hashtags". TikTok scraping has been blocked since 2024.04 (yt-dlp's TikTok
tag/search extractors are broken), so we use YouTube Shorts here.

The script issues a battery of search queries, deduplicates by video_id, and
saves URLs along with a *weak prior* (sludge / non-sludge based on the query
intent — NOT a ground-truth label). The auxiliary-gold labels come later
from Gemma-4-31B-IT (scripts/42_label_ood_with_gemma4.py).

Output:
  data/tiktok_v2/ood_live/youtube_url_pool.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List


# Each query is paired with a weak prior. The model never sees these labels —
# they're only used to balance the search budget across classes.
SLUDGE_QUERIES = [
    # Multi-pane / split-screen / brainrot patterns — closest to the
    # paper's definition of sludge (deliberate multimodal misalignment).
    "tiktok sludge compilation",
    "tiktok brain rot compilation",
    "subway surfers tiktok parkour",
    "split screen tiktok brainrot",
    "satisfying content sludge tiktok",
    "minecraft parkour family guy tiktok",
    "sludge content compilation 2026",
    "tiktok story time subway surfers",
    "split screen reddit story",
    "two videos at once tiktok",
    "multi pane brainrot",
    "skibidi toilet tiktok compilation",
    "tiktok asmr sludge",
    "satisfying split screen video",
    "minecraft parkour reddit ai voice",
    # User-suggested terms: targeting short-form & known sludge creators
    "subway surfers gameplay tiktok shorts",
    "minecraft parkour tiktok shorts",
    "family guy clips subway surfers shorts",
    "reddit ai voice subway surfers shorts",
    "story time minecraft parkour shorts",
    "tiktok brainrot shorts compilation",
    "satisfying content side by side shorts",
    "tiktok overlay gameplay shorts",
]

NONSLUDGE_QUERIES = [
    "tiktok funny moments compilation",
    "youtube shorts dance",
    "youtube shorts cute pets",
    "youtube shorts cooking",
    "viral tiktok 2026",
    "tiktok dance challenge",
    "tiktok food recipe shorts",
    "youtube shorts comedy",
    "tiktok pranks compilation",
    "youtube shorts vlog",
    "tiktok talking head review",
    "youtube shorts science explainer",
    "tiktok make up tutorial",
    "youtube shorts skits",
    "tiktok pet videos cute",
]


def ytsearch(query: str, n: int, timeout: int = 60) -> List[dict]:
    """Returns a list of {video_id, title, channel, duration, view_count, ...}."""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--no-warnings",
        "--print-json",
        f"ytsearch{n}:{query}",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[ytsearch] '{query}' timed out", file=sys.stderr)
        return []
    if r.returncode != 0:
        print(f"[ytsearch] '{query}' failed: {r.stderr.strip()[:200]}", file=sys.stderr)
        return []
    out: List[dict] = []
    for line in r.stdout.splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/tiktok_v2/ood_live/youtube_url_pool.jsonl")
    p.add_argument("--per-query", type=int, default=20,
                   help="Top-N results per query (yt-dlp caps under ~50).")
    p.add_argument("--sleep", type=float, default=1.0)
    p.add_argument("--max-duration-s", type=float, default=180.0,
                   help="Skip Shorts longer than this (we want short-form, not full vids).")
    p.add_argument("--min-duration-s", type=float, default=5.0,
                   help="Skip ultra-short clips (likely thumbnails or filler).")
    args = p.parse_args()

    pool: dict[str, dict] = {}

    def run_block(queries, weak_prior, tag):
        for q in queries:
            results = ytsearch(q, args.per_query)
            kept = 0
            for r in results:
                vid = r.get("id") or r.get("video_id")
                if not vid:
                    continue
                dur = r.get("duration") or 0
                if dur and (dur < args.min_duration_s or dur > args.max_duration_s):
                    continue
                if vid in pool:
                    continue
                pool[vid] = {
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "video_id_youtube": vid,
                    "platform": "youtube",
                    "query": q,
                    "query_class": tag,
                    "candidate_label_sludge": weak_prior,
                    "title": (r.get("title") or "")[:200],
                    "channel": r.get("channel"),
                    "duration": dur,
                    "view_count": r.get("view_count"),
                }
                kept += 1
            print(f"[ytsearch] '{q}' kept {kept} (running total {len(pool)})", file=sys.stderr)
            time.sleep(args.sleep)

    run_block(SLUDGE_QUERIES, True, "sludge")
    run_block(NONSLUDGE_QUERIES, False, "nonsludge")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in pool.values():
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    by_prior = Counter(r["candidate_label_sludge"] for r in pool.values())
    print(f"[ytsearch] wrote {len(pool)} unique URLs to {out}", file=sys.stderr)
    print(f"[ytsearch] by weak prior: {by_prior}", file=sys.stderr)


if __name__ == "__main__":
    main()

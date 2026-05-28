"""Pull TikTok URLs from Reddit's public JSON endpoints for OOD evaluation.

This is the URL-sourcing step before yt-dlp download. We hit three subs:

  - r/brainrot   — primary source of sludge-style content.
  - r/tiktok     — general TikTok cross-posts; mostly non-sludge.
  - r/fyp        — small "for you" cross-post community.

For each, we fetch from /new and /hot endpoints (different selection bias —
new is recency-biased, hot is upvote-biased) and aggregate. Reddit's public
JSON is unauthenticated but rate-limited; we sleep between requests.

Bias disclosure for the paper: Reddit cross-posts over-represent "remarkable"
content. The user-facing FYP feed has more middle-of-feed clips. We use this
source because yt-dlp's TikTok search/hashtag extractors are currently broken
(2024.04.09 release), and we can't write through TikTok's anti-scraping.

Output:
  data/tiktok_v2/ood_live/reddit_url_pool.jsonl
    one row per unique TikTok URL with {url, subreddit, source, title, score,
    upvote_ratio, created_utc, candidate_label}. `candidate_label` is a *weak*
    prior from which sub the URL came from — NOT a ground-truth label.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Iterable, List

import urllib.request


SUBS = [
    # (name, weak_prior_is_sludge) — weak label only, do not use for eval.
    ("brainrot", True),
    ("skibiditoilet", True),       # skibidi-toilet adjacent sludge-style content
    ("nichetok", True),            # niche TikTok cross-posts, often sludge
    ("tiktok", False),
    ("fyp", False),
]

TIKTOK_RE = re.compile(r"https?://(?:www\.|vm\.|m\.)?tiktok\.com/[^\s\"'\)]+", re.I)


def _fetch_json(url: str, ua: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def _extract_tiktok_urls(text: str) -> List[str]:
    if not text:
        return []
    return list({m.group(0).rstrip(".,;:") for m in TIKTOK_RE.finditer(text)})


def fetch_sub(sub: str, sort: str, after: str = "", limit: int = 100, ua: str = "") -> List[dict]:
    """Single page of /r/<sub>/<sort>.json. Returns the list of post dicts."""
    qs = urllib.parse.urlencode({"limit": str(limit), "after": after or "", "raw_json": "1"})
    url = f"https://www.reddit.com/r/{sub}/{sort}.json?{qs}"
    data = _fetch_json(url, ua)
    return [c["data"] for c in data.get("data", {}).get("children", [])]


def harvest_subreddit(sub: str, weak_label: bool, sorts: Iterable[str], pages: int,
                      sleep_s: float, ua: str) -> List[dict]:
    out: List[dict] = []
    for sort in sorts:
        after = ""
        for page in range(pages):
            try:
                posts = fetch_sub(sub, sort, after=after, ua=ua)
            except Exception as e:  # noqa: BLE001
                print(f"[reddit] /r/{sub}/{sort} page {page}: {e}", file=sys.stderr)
                break
            if not posts:
                break
            for post in posts:
                # TikTok URLs can live in `url`, `url_overridden_by_dest`, or selftext.
                candidates: List[str] = []
                for k in ("url", "url_overridden_by_dest"):
                    v = post.get(k)
                    if v and isinstance(v, str):
                        candidates.extend(_extract_tiktok_urls(v))
                candidates.extend(_extract_tiktok_urls(post.get("selftext", "") or ""))
                for url in candidates:
                    out.append({
                        "url": url,
                        "subreddit": sub,
                        "source": sort,
                        "title": post.get("title", "")[:200],
                        "score": post.get("score"),
                        "upvote_ratio": post.get("upvote_ratio"),
                        "created_utc": post.get("created_utc"),
                        "candidate_label_sludge": weak_label,
                    })
            after = posts[-1].get("name") if posts else ""
            if not after:
                break
            time.sleep(sleep_s)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/tiktok_v2/ood_live/reddit_url_pool.jsonl")
    p.add_argument("--pages", type=int, default=5,
                   help="Pages per sort (Reddit caps to ~100 posts/page).")
    p.add_argument("--sleep", type=float, default=1.5,
                   help="Seconds between requests (Reddit unauth rate limit).")
    p.add_argument("--ua", default="vqwen-research-bot/0.1 by /u/alpharomercoma")
    args = p.parse_args()

    all_rows: List[dict] = []
    for sub, weak in SUBS:
        rows = harvest_subreddit(sub, weak, ["new", "hot"], pages=args.pages,
                                 sleep_s=args.sleep, ua=args.ua)
        print(f"[harvest] r/{sub}: {len(rows)} candidate URLs", file=sys.stderr)
        all_rows.extend(rows)

    # Deduplicate by URL while preserving first-seen metadata.
    seen: dict[str, dict] = {}
    for r in all_rows:
        seen.setdefault(r["url"], r)
    deduped = list(seen.values())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in deduped:
            f.write(json.dumps(r) + "\n")

    # Per-sub stats
    by_sub = {}
    for r in deduped:
        by_sub.setdefault(r["subreddit"], 0)
        by_sub[r["subreddit"]] += 1
    print(f"[harvest] wrote {len(deduped)} unique TikTok URLs to {out}", file=sys.stderr)
    print(f"[harvest] by subreddit: {by_sub}", file=sys.stderr)


if __name__ == "__main__":
    main()

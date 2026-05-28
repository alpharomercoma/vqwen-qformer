"""Build a leakage-resistant train/val/test split for the TikTok sludge dataset.

The original Kaggle split (data/tiktok_v2/kaggle_root/split/{train,validate,test}.json)
is video-ID-disjoint but does not separate near-duplicate content. Because TikTok
URLs in this dataset are anonymized (`@username` is literal) and mp4 metadata is
stripped, we cannot derive an "uploader" field. Instead we cluster videos by
*perceptual audio fingerprint* (Chromaprint via fpcalc) — videos that share a
background audio track (Subway Surfers clips, popular music loops, etc.) get
grouped together. A group-disjoint split prevents the model from memorizing
audio→label associations across train/test.

Outputs:
  data/tiktok_v2/creator_map.jsonl       — {video_id, group_id}
  data/tiktok_v2/kaggle_root/split_audio_grouped/{train,validate,test}.json
                                          — same schema as the original splits
                                            ([{id, classification, batch_name}, ...])

Configurable knobs at the top of main(). Run with the project venv:
  /home/alpha/vqwen-qformer/.venv/bin/python scripts/15_build_creator_split.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "tiktok_v2" / "kaggle_root"
OUT_MAP_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "tiktok_v2" / "creator_map.jsonl"
OUT_FP_CACHE_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "tiktok_v2" / "audio_fp_cache.jsonl"


def _fpcalc_one(args: Tuple[str, str]) -> Tuple[str, List[int]]:
    """Return (video_id, raw-fingerprint-as-list-of-uint32)."""
    vid, wav_path = args
    try:
        out = subprocess.run(
            ["fpcalc", "-raw", "-length", "30", wav_path],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout
        fp_line = next(l for l in out.splitlines() if l.startswith("FINGERPRINT="))
        ints = [int(x) for x in fp_line[len("FINGERPRINT="):].split(",") if x]
        return vid, ints
    except Exception as e:  # noqa: BLE001
        print(f"[fpcalc] {vid}: {e}", file=sys.stderr)
        return vid, []


def fingerprint_all(rows: List[dict], root: Path, cache: Path, workers: int) -> Dict[str, np.ndarray]:
    """Fingerprint every wav with fpcalc; cache to disk so reruns are fast."""
    have: Dict[str, List[int]] = {}
    if cache.exists():
        for line in cache.open():
            d = json.loads(line)
            have[d["video_id"]] = d["fp"]
        print(f"[fp] loaded {len(have)} cached fingerprints from {cache}")

    todo: List[Tuple[str, str]] = []
    for r in rows:
        if r["video_id"] in have:
            continue
        wav = root / r["wav_path"]
        if not wav.exists():
            print(f"[fp] missing wav for {r['video_id']}, skipping")
            continue
        todo.append((r["video_id"], str(wav)))

    if todo:
        print(f"[fp] computing {len(todo)} fingerprints with {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (vid, fp) in enumerate(ex.map(_fpcalc_one, todo, chunksize=8), 1):
                have[vid] = fp
                if i % 100 == 0:
                    print(f"[fp] {i}/{len(todo)}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w") as f:
            for vid, fp in have.items():
                f.write(json.dumps({"video_id": vid, "fp": fp}) + "\n")
        print(f"[fp] wrote {len(have)} fingerprints to {cache}")

    return {vid: np.array(fp, dtype=np.uint32) for vid, fp in have.items() if fp}


def pairwise_ber_clusters(fps: Dict[str, np.ndarray], threshold: float) -> Dict[str, str]:
    """Union videos whose Chromaprint BER (on aligned prefix) is below `threshold`.

    Returns {video_id -> group_id (the canonical video_id of the cluster root)}.
    Pure numpy + Python; runs in seconds on 2k videos.
    """
    ids = list(fps.keys())
    parent = {v: v for v in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Precompute bit-unpacked uint8 views (each fp int becomes 4 bytes / 32 bits)
    packed = {vid: arr.view(np.uint8) for vid, arr in fps.items()}

    n = len(ids)
    for i in range(n):
        ai = packed[ids[i]]
        for j in range(i + 1, n):
            bj = packed[ids[j]]
            m = min(len(ai), len(bj))
            if m < 60 * 4:  # need ≥60 ints (~7.5s of audio) to compare
                continue
            xor = np.bitwise_xor(ai[:m], bj[:m])
            popcount = int(np.unpackbits(xor).sum())
            ber = popcount / (m * 8)
            if ber < threshold:
                union(ids[i], ids[j])
        if (i + 1) % 200 == 0:
            print(f"[union] {i+1}/{n}")

    return {v: find(v) for v in ids}


def stratified_group_split(
    rows: List[dict],
    vid_to_group: Dict[str, str],
    frac_test: float,
    frac_val: float,
    seed: int,
) -> Dict[str, List[dict]]:
    """Assign whole groups to train/val/test, balancing video counts per class.

    For each label (sludge/non-sludge) separately we walk a shuffled list of
    groups and greedily fill test → val → train until each bucket's video-count
    target is met. This keeps the class ratio inside each split close to 50/50
    even when group sizes vary.
    """
    by_id = {r["video_id"]: r for r in rows}
    label_to_groups: Dict[bool, Dict[str, List[str]]] = {True: defaultdict(list), False: defaultdict(list)}
    for vid, gid in vid_to_group.items():
        if vid not in by_id:
            continue
        label_to_groups[by_id[vid]["is_sludge"]][gid].append(vid)

    out = {"train": [], "validate": [], "test": []}
    rng = random.Random(seed)

    for label, group_map in label_to_groups.items():
        groups = list(group_map.keys())
        rng.shuffle(groups)
        n_vids = sum(len(v) for v in group_map.values())
        target_test = round(frac_test * n_vids)
        target_val = round(frac_val * n_vids)
        # Greedy fill: test, then val, rest to train.
        ct = cv = 0
        for gid in groups:
            members = group_map[gid]
            if ct + len(members) <= target_test or ct < target_test:
                bucket = "test"; ct += len(members)
            elif cv + len(members) <= target_val or cv < target_val:
                bucket = "validate"; cv += len(members)
            else:
                bucket = "train"
            for vid in members:
                r = by_id[vid]
                out[bucket].append({
                    "id": vid,
                    "classification": bool(r["is_sludge"]),
                    "batch_name": r["batch"],
                })

    return out


def assert_disjoint(out: Dict[str, List[dict]], vid_to_group: Dict[str, str]) -> None:
    seen_groups: Dict[str, str] = {}
    for split_name, items in out.items():
        for it in items:
            gid = vid_to_group[it["id"]]
            if gid in seen_groups and seen_groups[gid] != split_name:
                raise AssertionError(
                    f"Group {gid} spans {seen_groups[gid]} and {split_name} (videos: {it['id']})"
                )
            seen_groups[gid] = split_name


def print_split_stats(out: Dict[str, List[dict]], vid_to_group: Dict[str, str]) -> None:
    for split_name, items in out.items():
        sludge = sum(1 for it in items if it["classification"])
        n = len(items)
        groups = {vid_to_group[it["id"]] for it in items}
        sludge_groups = {vid_to_group[it["id"]] for it in items if it["classification"]}
        ns_groups = {vid_to_group[it["id"]] for it in items if not it["classification"]}
        print(
            f"  {split_name:10s}  videos={n:5d}  sludge={sludge:5d} ({100*sludge/n:5.1f}%)  "
            f"groups={len(groups):5d} (sludge_g={len(sludge_groups)}, ns_g={len(ns_groups)})"
        )


def hash_split_set(items: List[dict]) -> str:
    """Stable hash of a split's video-id set; useful for log assertions."""
    h = hashlib.sha256()
    for it in sorted(items, key=lambda x: x["id"]):
        h.update(it["id"].encode())
    return h.hexdigest()[:16]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    p.add_argument("--out-map", type=Path, default=OUT_MAP_DEFAULT)
    p.add_argument("--fp-cache", type=Path, default=OUT_FP_CACHE_DEFAULT)
    p.add_argument("--ber-threshold", type=float, default=0.30,
                   help="Bit Error Rate below which two fingerprints are 'same audio source'. "
                        "0.30 ≈ Chromaprint canonical; raise to cluster more aggressively.")
    p.add_argument("--frac-test", type=float, default=0.15)
    p.add_argument("--frac-val", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    rows = [json.loads(l) for l in (args.root / "videos_index.jsonl").open()]
    print(f"[main] loaded {len(rows)} videos")

    fps = fingerprint_all(rows, args.root, args.fp_cache, args.workers)
    print(f"[main] fingerprinted {len(fps)} videos")

    print(f"[main] clustering with BER<{args.ber_threshold}...")
    vid_to_group = pairwise_ber_clusters(fps, args.ber_threshold)
    groups = defaultdict(list)
    for v, g in vid_to_group.items():
        groups[g].append(v)
    sizes = [len(g) for g in groups.values()]
    print(f"[cluster] {len(groups)} groups; singletons={sum(1 for s in sizes if s==1)}; "
          f"max={max(sizes)}; top-10 sizes={sorted(sizes, reverse=True)[:10]}")

    # Stats per class
    by_id = {r["video_id"]: r for r in rows}
    sludge_groups = {g for g, vs in groups.items() if any(by_id[v]["is_sludge"] for v in vs)}
    ns_groups = {g for g, vs in groups.items() if not any(by_id[v]["is_sludge"] for v in vs)}
    print(f"[cluster] sludge-bearing groups: {len(sludge_groups)}  |  pure non-sludge groups: {len(ns_groups)}")

    # Emit creator_map.jsonl
    args.out_map.parent.mkdir(parents=True, exist_ok=True)
    with args.out_map.open("w") as f:
        for vid, gid in vid_to_group.items():
            f.write(json.dumps({"video_id": vid, "group_id": gid}) + "\n")
    print(f"[main] wrote {args.out_map}")

    # Build split
    out = stratified_group_split(rows, vid_to_group, args.frac_test, args.frac_val, args.seed)
    assert_disjoint(out, vid_to_group)
    print(f"[split] group-disjoint OK")
    print_split_stats(out, vid_to_group)

    # Persist
    split_dir = args.root / "split_audio_grouped"
    split_dir.mkdir(exist_ok=True)
    for name, items in out.items():
        (split_dir / f"{name}.json").write_text(json.dumps(items, indent=2))
    print(f"[main] wrote {split_dir}/(train|validate|test).json")
    print(f"[main] split-set hashes: train={hash_split_set(out['train'])}  "
          f"validate={hash_split_set(out['validate'])}  test={hash_split_set(out['test'])}")


if __name__ == "__main__":
    main()

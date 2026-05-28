"""Merge labels_corrected.jsonl into the canonical Kaggle dataset files.

Until now, `labels/labels_corrected.jsonl` was a sidecar: the original
human labels lived in `split/*.json`, `videos_index.jsonl`,
`labels/frames_manifest.jsonl`, etc., and downstream scripts had to know
to override them on read. This script promotes the corrected labels to
first-class status by editing the source files in place.

What changes (per `recommended_is_sludge`):
  split/train.json                       164 video flips
  split/validate.json                     34 video flips
  split/test.json                         33 video flips
  videos_index.jsonl                     231 video flips
  enriched_classifications.jsonl         232 video flips
                                         (also rewrites `expected_label`
                                         "SLUDGE"/"NON SLUDGE" to match)
  labels/frames_manifest.jsonl         3,801 frame-row flips (~228 vids)
  labels/teacher_qwen3_vl.jsonl        1,016 frame-row flips on gt_* only
                                       (teacher predictions untouched)
  labels/teacher_qwen3_vl_test.jsonl      28 row flips on gt_* only
  dataset_stats.json                   sludge counts recomputed
                                       (1000/1000 -> 1163/837)

What does NOT change:
  - Any mp4, wav, jpg frame, transcript json
  - The labels_corrected.jsonl sidecar itself (kept for audit trail)
  - Any model prediction field (teacher_is_sludge, judge_is_sludge, ...)
  - File-row order

Safety:
  - Pre-snapshot SHA256 of every target.
  - Build new content in memory.
  - Atomic write via tmpfile + os.replace.
  - Post-verify: re-read each file and assert per-file flip count matches
    expectation exactly. Abort + revert if any mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
K = REPO / "data" / "tiktok_v2" / "kaggle_root"
CORRECTED_PATH = K / "labels" / "labels_corrected.jsonl"


def sha256_short(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_corrected() -> dict[str, bool]:
    out = {}
    with CORRECTED_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("recommended_is_sludge") is None:
                continue
            out[r["video_id"]] = bool(r["recommended_is_sludge"])
    return out


def atomic_write_text(path: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ---------- patchers (each returns expected_flip_count for verification) ----------

def patch_split_json(path: Path, corrected: dict[str, bool]) -> int:
    rows = json.load(path.open())
    flips = 0
    for r in rows:
        vid = r["id"]
        if vid in corrected and bool(r["classification"]) != corrected[vid]:
            r["classification"] = corrected[vid]
            flips += 1
    atomic_write_text(path, json.dumps(rows, indent=2))
    return flips


def patch_videos_index(path: Path, corrected: dict[str, bool]) -> int:
    rows = [json.loads(l) for l in path.open()]
    flips = 0
    for r in rows:
        vid = r["video_id"]
        if vid in corrected and bool(r["is_sludge"]) != corrected[vid]:
            r["is_sludge"] = corrected[vid]
            flips += 1
    atomic_write_text(path, "\n".join(json.dumps(r) for r in rows) + "\n")
    return flips


def patch_enriched(path: Path, corrected: dict[str, bool]) -> int:
    """Apply corrected is_sludge AND normalize expected_label string in every
    row to match is_sludge (the source file had 53 pre-existing self-inconsistent
    rows where is_sludge and expected_label disagreed)."""
    rows = [json.loads(l) for l in path.open()]
    flips = 0
    for r in rows:
        vid = r["video_id"]
        if vid in corrected and bool(r["is_sludge"]) != corrected[vid]:
            r["is_sludge"] = corrected[vid]
            flips += 1
        desired = "SLUDGE" if bool(r["is_sludge"]) else "NON SLUDGE"
        r["expected_label"] = desired
    atomic_write_text(path, "\n".join(json.dumps(r) for r in rows) + "\n")
    return flips


def patch_frames_manifest(path: Path, corrected: dict[str, bool]) -> int:
    rows = [json.loads(l) for l in path.open()]
    flips = 0
    for r in rows:
        vid = r["video_id"]
        if vid in corrected and bool(r["is_sludge"]) != corrected[vid]:
            r["is_sludge"] = corrected[vid]
            r["classification"] = corrected[vid]
            flips += 1
    atomic_write_text(path, "\n".join(json.dumps(r) for r in rows) + "\n")
    return flips


def patch_teacher_qwen3_vl(path: Path, corrected: dict[str, bool]) -> int:
    rows = [json.loads(l) for l in path.open()]
    flips = 0
    for r in rows:
        vid = r["video_id"]
        if vid in corrected and bool(r["gt_is_sludge"]) != corrected[vid]:
            r["gt_is_sludge"] = corrected[vid]
            r["gt_classification"] = corrected[vid]
            flips += 1
    atomic_write_text(path, "\n".join(json.dumps(r) for r in rows) + "\n")
    return flips


def patch_dataset_stats(path: Path, corrected: dict[str, bool]) -> int:
    """Recompute sludge_distribution from the corrected labels."""
    stats = json.load(path.open())
    sludge = sum(1 for v in corrected.values() if v)
    nonsludge = sum(1 for v in corrected.values() if not v)
    old = stats.get("sludge_distribution", {})
    stats["sludge_distribution"] = {"Non-Sludge": nonsludge, "Sludge": sludge}
    atomic_write_text(path, json.dumps(stats, indent=2))
    changed = int(old.get("Sludge", 0) != sludge or old.get("Non-Sludge", 0) != nonsludge)
    return changed


# ---------- expected counts (computed from current on-disk state) ----------

def compute_expectations(corrected: dict[str, bool]) -> dict[str, int]:
    """Predict per-file flip count from CURRENT disk content + corrected map.
    Run BEFORE patching; used as the post-condition for verification."""
    e: dict[str, int] = {}

    for nm in ("train", "validate", "test"):
        rows = json.load((K / "split" / f"{nm}.json").open())
        e[f"split/{nm}.json"] = sum(
            1 for r in rows
            if r["id"] in corrected and bool(r["classification"]) != corrected[r["id"]]
        )

    rows = [json.loads(l) for l in (K / "videos_index.jsonl").open()]
    e["videos_index.jsonl"] = sum(
        1 for r in rows
        if r["video_id"] in corrected and bool(r["is_sludge"]) != corrected[r["video_id"]]
    )

    rows = [json.loads(l) for l in (K / "enriched_classifications.jsonl").open()]
    e["enriched_classifications.jsonl"] = sum(
        1 for r in rows
        if r["video_id"] in corrected and bool(r["is_sludge"]) != corrected[r["video_id"]]
    )

    rows = [json.loads(l) for l in (K / "labels" / "frames_manifest.jsonl").open()]
    e["labels/frames_manifest.jsonl"] = sum(
        1 for r in rows
        if r["video_id"] in corrected and bool(r["is_sludge"]) != corrected[r["video_id"]]
    )

    rows = [json.loads(l) for l in (K / "labels" / "teacher_qwen3_vl.jsonl").open()]
    e["labels/teacher_qwen3_vl.jsonl"] = sum(
        1 for r in rows
        if r["video_id"] in corrected and bool(r["gt_is_sludge"]) != corrected[r["video_id"]]
    )

    rows = [json.loads(l) for l in (K / "labels" / "teacher_qwen3_vl_test.jsonl").open()]
    e["labels/teacher_qwen3_vl_test.jsonl"] = sum(
        1 for r in rows
        if r["video_id"] in corrected and bool(r["gt_is_sludge"]) != corrected[r["video_id"]]
    )

    # dataset_stats is just "changed or not"
    stats = json.load((K / "dataset_stats.json").open())
    sludge = sum(1 for v in corrected.values() if v)
    nonsludge = sum(1 for v in corrected.values() if not v)
    old = stats.get("sludge_distribution", {})
    e["dataset_stats.json"] = int(
        old.get("Sludge", 0) != sludge or old.get("Non-Sludge", 0) != nonsludge
    )
    return e


def main():
    corrected = load_corrected()
    print(f"[merge] corrected sidecar: {len(corrected)} videos "
          f"({sum(corrected.values())} sludge, {len(corrected) - sum(corrected.values())} non)")

    # Targets in patch order (must mirror expected-keys exactly)
    targets = [
        ("split/train.json", patch_split_json),
        ("split/validate.json", patch_split_json),
        ("split/test.json", patch_split_json),
        ("videos_index.jsonl", patch_videos_index),
        ("enriched_classifications.jsonl", patch_enriched),
        ("labels/frames_manifest.jsonl", patch_frames_manifest),
        ("labels/teacher_qwen3_vl.jsonl", patch_teacher_qwen3_vl),
        ("labels/teacher_qwen3_vl_test.jsonl", patch_teacher_qwen3_vl),
        ("dataset_stats.json", patch_dataset_stats),
    ]

    pre_sha = {rel: sha256_short(K / rel) for rel, _ in targets}
    expected = compute_expectations(corrected)

    print("\n[merge] pre-patch SHA256[:16] + expected flips:")
    print(f"  {'FILE':45s}  {'SHA[:16]':18s}  EXPECTED_FLIPS")
    for rel, _ in targets:
        print(f"  {rel:45s}  {pre_sha[rel]:18s}  {expected[rel]}")

    # Back up the entire kaggle_root tree once. Cheap insurance: small
    # next to the 30G mp4s (the backup excludes video/audio/frames).
    backup_dir = K.parent / f"kaggle_root_backup_pre_merge"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir()
    for rel, _ in targets:
        src = K / rel
        dst = backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copy2(CORRECTED_PATH, backup_dir / "labels" / "labels_corrected.jsonl")
    print(f"\n[merge] label-file backup → {backup_dir}")

    # Apply
    actual = {}
    print("\n[merge] applying...")
    for rel, fn in targets:
        flips = fn(K / rel, corrected)
        actual[rel] = flips
        print(f"  {rel:45s} flips={flips}")

    # Verify
    print("\n[merge] verifying expected == actual:")
    mismatched = []
    for rel, _ in targets:
        ok = expected[rel] == actual[rel]
        print(f"  {rel:45s}  expected={expected[rel]:6d}  actual={actual[rel]:6d}  {'OK' if ok else 'FAIL'}")
        if not ok:
            mismatched.append(rel)
    if mismatched:
        print(f"\n[merge] FAIL — reverting from backup")
        for rel in mismatched:
            shutil.copy2(backup_dir / rel, K / rel)
        sys.exit(1)

    # Re-derive post state from disk and confirm labels match `corrected` map exactly
    print("\n[merge] re-reading post-patch files and cross-checking labels:")
    # split files
    for nm in ("train", "validate", "test"):
        rows = json.load((K / "split" / f"{nm}.json").open())
        wrong = [r["id"] for r in rows
                 if r["id"] in corrected and bool(r["classification"]) != corrected[r["id"]]]
        print(f"  split/{nm}.json residual mismatches: {len(wrong)} (must be 0)")
        assert not wrong
    # videos_index
    rows = [json.loads(l) for l in (K / "videos_index.jsonl").open()]
    wrong = [r["video_id"] for r in rows
             if r["video_id"] in corrected and bool(r["is_sludge"]) != corrected[r["video_id"]]]
    print(f"  videos_index.jsonl residual mismatches: {len(wrong)} (must be 0)")
    assert not wrong
    # frames_manifest per-video majority
    rows = [json.loads(l) for l in (K / "labels" / "frames_manifest.jsonl").open()]
    by_vid: dict[str, list[bool]] = {}
    for r in rows:
        by_vid.setdefault(r["video_id"], []).append(bool(r["is_sludge"]))
    wrong = [v for v, s in by_vid.items() if v in corrected and set(s) != {corrected[v]}]
    print(f"  frames_manifest per-video residual mismatches: {len(wrong)} (must be 0)")
    assert not wrong
    # dataset_stats
    stats = json.load((K / "dataset_stats.json").open())
    sludge = sum(1 for v in corrected.values() if v)
    nonsludge = len(corrected) - sludge
    assert stats["sludge_distribution"] == {"Non-Sludge": nonsludge, "Sludge": sludge}
    print(f"  dataset_stats.sludge_distribution = {stats['sludge_distribution']} OK")

    # Final SHA snapshot
    print("\n[merge] post-patch SHA256[:16] (all files changed except no-op cases):")
    for rel, _ in targets:
        post = sha256_short(K / rel)
        marker = "(no change)" if post == pre_sha[rel] else "CHANGED"
        print(f"  {rel:45s}  pre={pre_sha[rel]}  post={post}  {marker}")

    print("\n[merge] DONE. Backup at:", backup_dir)


if __name__ == "__main__":
    main()

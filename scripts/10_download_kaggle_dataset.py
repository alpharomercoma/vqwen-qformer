"""Download the official TikTok sludge dataset from Kaggle.

Source:  https://www.kaggle.com/datasets/jobisaacong/tiktok-sludge-dataset-500
Target:  data/tiktok_v2/kaggle_root/
Layout after unzip:
    video/{Sludge,Non_Sludge}_Batch_{1,2}/*.mp4
    audio/{Sludge,Non_Sludge}_Batch_{1,2}/*.wav
    text/{Sludge,Non_Sludge}_Batch_{1,2}/*.json   (Whisper transcripts)
    split/{train,validate,test}.json
    enriched_classifications.jsonl

Requires `~/.kaggle/kaggle.json` (or KAGGLE_USERNAME / KAGGLE_KEY env vars).
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "data" / "tiktok_v2" / "kaggle_root"
DATASET_SLUG = "jobisaacong/tiktok-sludge-dataset-500"


def have_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def _kaggle_cmd() -> list[str] | None:
    """Resolve a usable kaggle CLI. Prefer the binary on PATH; fall back to
    invoking the installed kaggle package as a module via the current Python.
    Returns None when neither path works.
    """
    on_path = shutil.which("kaggle")
    if on_path:
        return [on_path]
    if importlib.util.find_spec("kaggle") is not None:
        return [sys.executable, "-m", "kaggle"]
    return None


def main() -> int:
    cmd = _kaggle_cmd()
    if cmd is None:
        print("[kaggle] `kaggle` not found on PATH and the `kaggle` Python "
              "package is not installed in this interpreter. "
              "Install it: pip install kaggle", file=sys.stderr)
        return 2
    if not have_credentials():
        print("[kaggle] missing credentials. Put kaggle.json in ~/.kaggle/ "
              "(chmod 600) or set KAGGLE_USERNAME / KAGGLE_KEY.", file=sys.stderr)
        return 2

    if (TARGET / "enriched_classifications.jsonl").exists():
        print(f"[kaggle] already downloaded at {TARGET}; skipping.")
        return 0

    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] downloading {DATASET_SLUG} -> {TARGET}  (~30 GB compressed)")
    r = subprocess.run(
        cmd + ["datasets", "download", "-d", DATASET_SLUG,
               "-p", str(TARGET), "--unzip"],
        check=False,
    )
    if r.returncode != 0:
        print(f"[kaggle] download failed (rc={r.returncode})", file=sys.stderr)
        return r.returncode

    splits = TARGET / "split"
    enriched = TARGET / "enriched_classifications.jsonl"
    if not splits.exists() or not enriched.exists():
        print(f"[kaggle] downloaded but expected files missing under {TARGET}", file=sys.stderr)
        return 3
    print(f"[kaggle] done. Split files: {sorted(p.name for p in splits.glob('*.json'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

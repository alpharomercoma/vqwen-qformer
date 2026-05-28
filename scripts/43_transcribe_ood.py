"""Whisper-base transcripts for the OOD TikTok set (matches deployment Space).

Reads data/tiktok_v2/ood_live/manifest.jsonl, runs openai/whisper-base over each
WAV, writes per-video text JSON in the same shape as kaggle_root/text/<batch>/<vid>.json
(a flat `{"text": "..."}` dict), and extends the manifest in place with the
`text_path` field that the eval scripts expect.

Whisper-base matches the deployed HF Space (CPU-budget choice) — testing under
the deployment ASR is the honest measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
OOD_ROOT = REPO_ROOT / "data" / "tiktok_v2" / "ood_live"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(OOD_ROOT / "manifest.jsonl"))
    p.add_argument("--model", default="openai/whisper-base")
    p.add_argument("--out-text-dir", default=str(OOD_ROOT / "text"))
    args = p.parse_args()

    from transformers import pipeline

    out_text_dir = Path(args.out_text_dir)
    out_text_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    records = [json.loads(l) for l in manifest_path.open()]
    print(f"[whisper] loading {args.model}...", file=sys.stderr)
    # chunk_length_s=30 lets the pipeline auto-chunk long audio (Whisper's mel
    # input is capped at ~30 s; without chunking, anything longer fails).
    asr = pipeline("automatic-speech-recognition", model=args.model, device=0,
                   chunk_length_s=30, return_timestamps=True)
    print(f"[whisper] {len(records)} videos to transcribe", file=sys.stderr)

    t0 = time.time()
    updated: list[dict] = []
    for i, r in enumerate(records, 1):
        vid = r["video_id"]
        text_path = out_text_dir / f"{vid}.json"
        # Re-transcribe if the cached transcript is empty (was likely a long-audio failure).
        force = text_path.exists() and len(json.load(text_path.open()).get("text", "")) == 0
        if text_path.exists() and not force:
            text = json.load(text_path.open()).get("text", "")
        else:
            wav = OOD_ROOT / r["wav_path"]
            try:
                out = asr(str(wav))
                text = (out.get("text") if isinstance(out, dict) else "") or ""
                text = text.strip()
            except Exception as e:  # noqa: BLE001
                print(f"[whisper] {vid}: {e}", file=sys.stderr)
                text = ""
            text_path.write_text(json.dumps({"text": text}))

        r["text_path"] = str(text_path.relative_to(OOD_ROOT))
        r["transcript_len"] = len(text)
        updated.append(r)

        if i % 10 == 0:
            print(f"[whisper] {i}/{len(records)}  {(time.time()-t0)/i:.2f}s/video", file=sys.stderr)

    with manifest_path.open("w") as f:
        for r in updated:
            f.write(json.dumps(r) + "\n")
    print(f"[whisper] updated {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

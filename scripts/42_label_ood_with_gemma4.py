"""Label the OOD TikTok set with google/gemma-4-31B-it as a stronger-VLM judge.

For each video in the OOD manifest, we read the sampled frames, prompt
Gemma-4-31B-IT with the same sludge definition the model was trained on, and
record its yes/no plus a brief reason. The per-frame predictions are then
majority-voted to a per-video label.

These are *auxiliary* labels — they substitute for human gold on a large
sample, but they carry the stronger VLM's own biases. The paper should
report results vs both:
  - Reddit weak prior (very noisy, just sanity-check)
  - Gemma-4 majority vote (much higher quality, paper headline for OOD)

Inputs:
  data/tiktok_v2/ood_live/manifest.jsonl

Outputs:
  data/tiktok_v2/ood_live/gemma4_labels.jsonl
  data/tiktok_v2/ood_live/gemma4_video_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
OOD_ROOT = REPO_ROOT / "data" / "tiktok_v2" / "ood_live"
DEFAULT_MODEL = REPO_ROOT / "models" / "gemma-4-31B-it"

PROMPT = (
    "You are an expert moderator labelling short videos.\n\n"
    "'Sludge content' is a TikTok-style video that stacks MULTIPLE UNRELATED visual "
    "feeds at the same time on screen — for example a Reddit story scrolling above "
    "a Subway Surfers gameplay clip, or a split-screen with two unrelated cartoons. "
    "A single coherent scene (even one with text overlays, captions, talking heads, "
    "B-roll, multiple cuts of the same subject, or picture-in-picture of related "
    "content) is NOT sludge.\n\n"
    "Look at the frame and decide whether this is sludge content.\n"
    "Answer in exactly this JSON format:\n"
    "{\"is_sludge\": <true|false>, \"reason\": \"<one sentence>\"}"
)

PARSE_RE = re.compile(r"\{[^{}]*\"is_sludge\"\s*:\s*(true|false)[^{}]*\}", re.I | re.S)
REASON_RE = re.compile(r"\"reason\"\s*:\s*\"([^\"]+)\"", re.I | re.S)


def load_model(model_path: Path, dtype=torch.bfloat16):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[gemma4] loading {model_path}...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(str(model_path))
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map="cuda",
    ).eval()
    print(f"[gemma4] loaded.", file=sys.stderr)
    return model, processor


def classify_frame(model, processor, image: Image.Image) -> tuple[bool | None, str]:
    """Single-frame inference. Returns (verdict, raw_reply)."""
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ]}
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    # bf16 image is set up via processor; ensure float dtype on image tensor.
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    in_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False, use_cache=True)
    gen = out[0][in_len:]
    reply = processor.decode(gen, skip_special_tokens=True).strip()
    m = PARSE_RE.search(reply)
    if not m:
        return None, reply
    return m.group(1).lower() == "true", reply


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(OOD_ROOT / "manifest.jsonl"))
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--frames-per-video", type=int, default=4,
                   help="Cap frames sent through Gemma per video (manifest may have fewer).")
    p.add_argument("--out-frames", default=str(OOD_ROOT / "gemma4_labels.jsonl"))
    p.add_argument("--out-videos", default=str(OOD_ROOT / "gemma4_video_labels.jsonl"))
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    records = [json.loads(l) for l in manifest_path.open()]
    print(f"[gemma4] {len(records)} videos to label", file=sys.stderr)

    model, processor = load_model(Path(args.model))

    frame_out = Path(args.out_frames).open("w")
    video_out = Path(args.out_videos).open("w")
    n_frames_total = 0
    unparseable_frames = 0
    t0 = time.time()

    for i, rec in enumerate(records, 1):
        frames = (rec.get("frames") or [])[:args.frames_per_video]
        votes: list[bool] = []
        unparseable_this = 0
        for frame in frames:
            img_path = OOD_ROOT / frame["frame_path"]
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:  # noqa: BLE001
                print(f"[gemma4] {rec['video_id']} {img_path.name}: open failed: {e}", file=sys.stderr)
                continue
            verdict, reply = classify_frame(model, processor, img)
            n_frames_total += 1
            frame_out.write(json.dumps({
                "video_id": rec["video_id"],
                "frame_index": frame["frame_index"],
                "time_s": frame["time_s"],
                "verdict": verdict,
                "reply": reply,
            }) + "\n")
            if verdict is None:
                unparseable_frames += 1
                unparseable_this += 1
            else:
                votes.append(verdict)

        # Video-level majority vote (skip if all unparseable)
        if not votes:
            video_label = None
            confidence = 0.0
        else:
            yes = sum(1 for v in votes if v)
            video_label = yes > len(votes) / 2
            confidence = max(yes, len(votes) - yes) / len(votes)

        video_out.write(json.dumps({
            "video_id": rec["video_id"],
            "url": rec["url"],
            "subreddit": rec.get("subreddit"),
            "candidate_label_sludge": rec.get("candidate_label_sludge"),
            "gemma4_is_sludge": video_label,
            "gemma4_confidence": confidence,
            "votes": votes,
            "n_frames_seen": len(frames),
            "n_unparseable_frames": unparseable_this,
        }) + "\n")

        if i % 5 == 0:
            elapsed = time.time() - t0
            print(f"[gemma4] {i}/{len(records)}  {elapsed/i:.2f}s/video  "
                  f"unparseable_frames={unparseable_frames}/{n_frames_total}",
                  file=sys.stderr)

    frame_out.close(); video_out.close()
    print(f"[gemma4] done. frames={n_frames_total} unparseable={unparseable_frames} ", file=sys.stderr)


if __name__ == "__main__":
    main()

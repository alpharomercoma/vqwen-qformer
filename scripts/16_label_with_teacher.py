"""Run Qwen3-VL-30B-A3B-Thinking as a teacher to label TikTok frames.

For each unique (video_id, frame) in our training manifest, ask the teacher
one structured prompt that yields:
  - description  : what is visibly present in THIS frame
  - explanation  : short structural reason (sludge / not sludge)

We keep the existing ground-truth `classification` + `layout_category` from
the dataset split files (more reliable than teacher self-classification) and
combine with teacher-generated description + explanation.

Output:
  data/tiktok_fps1/teacher_labels.jsonl   # one record per frame
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

# Structured prompt — asks the teacher to emit JSON we can parse.
# "Describe only what is visibly present in this frame" avoids speculating on
# specific shows/games/channels it can't literally read off the pixels.
TEACHER_PROMPT = """Analyze this single frame from a short video and respond with STRICT JSON only (no code fences, no extra prose). Fields:

{
  "sludge": <"yes"|"no">,
  "layout": <"single_scene"|"vertical"|"horizontal"|"picture-in-picture"|"collage"|"grid"|"layered"|"other">,
  "description": "<1-2 sentence description of what is VISIBLY present in this frame — describe panes, objects, scene types, on-screen UI elements. Do NOT speculate on TV show names, game titles, or creator identities unless text clearly reads them off the image.>",
  "explanation": "<One short sentence explaining why this is or is not sludge content, grounded in the visible structure.>"
}

Sludge = multiple unrelated video streams shown simultaneously (split-screen, PiP, collage). Single-scene content is NOT sludge even if visually busy."""


def build_label_lookup(tiktok_root: Path) -> dict:
    lookup: dict = {}
    for sp in ("train", "validate"):
        for r in json.load(open(tiktok_root / "split" / f"{sp}.json")):
            lookup[r["id"]] = {
                "classification": r["classification"],
                "batch_name": r["batch_name"],
                "split": sp,
            }
    for line in open(tiktok_root / "enriched_classifications.jsonl"):
        r = json.loads(line)
        vid = r["video_id"]
        if vid in lookup:
            lookup[vid]["layout_category"] = r.get("layout_category", "")
            lookup[vid]["is_sludge"] = r.get("is_sludge", None)
    return lookup


def parse_teacher_json(text: str) -> dict:
    """Best-effort parse of the teacher's reply. Qwen3-VL-Thinking emits a
    CoT block followed by `</think>` then the final answer. Sometimes the
    opening `<think>` is implicit and only `</think>` appears."""
    t = text.strip()
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1].strip()
    # Drop any code-fence markers
    t = t.replace("```json", "").replace("```", "").strip()
    lbrace = t.find("{")
    rbrace = t.rfind("}")
    if lbrace == -1 or rbrace == -1 or rbrace <= lbrace:
        return {}
    try:
        return json.loads(t[lbrace : rbrace + 1])
    except json.JSONDecodeError:
        return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_model", default=str(REPO_ROOT / "models" / "Qwen3-VL-30B-A3B-Instruct"))
    p.add_argument("--manifest", default=str(REPO_ROOT / "data" / "tiktok_fps1" / "frames_manifest.jsonl"))
    p.add_argument("--frames_root", default=str(REPO_ROOT / "data" / "tiktok_fps1" / "frames"))
    p.add_argument("--labels_src", default="/home/alpha/vqwen/data/tiktok_sludge")
    p.add_argument("--output", default=str(REPO_ROOT / "data" / "tiktok_fps1" / "teacher_labels.jsonl"))
    p.add_argument("--frames_per_video", type=int, default=5, help="match retraining cap")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="skip already-labeled frame_paths")
    args = p.parse_args()

    labels = build_label_lookup(Path(args.labels_src))
    with open(args.manifest) as f:
        recs = [json.loads(l) for l in f]

    # Cap 5 frames/video (same as v4 training)
    import random
    random.seed(42)
    by_vid = collections.defaultdict(list)
    for r in recs:
        if r.get("split") not in ("train", "validate"):
            continue
        by_vid[r["video_id"]].append(r)
    chosen = []
    for vid, frames in by_vid.items():
        random.shuffle(frames)
        chosen.extend(frames[: args.frames_per_video])
    if args.limit > 0:
        chosen = chosen[: args.limit]
    print(f"[teacher] labeling {len(chosen)} frames ({len(by_vid)} videos x up to {args.frames_per_video})",
          file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if args.resume and out_path.exists():
        for l in open(out_path):
            try:
                done_keys.add(json.loads(l)["frame_path"])
            except Exception: pass
        print(f"[teacher] resume: {len(done_keys)} already labeled", file=sys.stderr)
        chosen = [r for r in chosen if r["frame_path"] not in done_keys]
        print(f"[teacher] remaining: {len(chosen)}", file=sys.stderr)

    print(f"[teacher] loading {args.teacher_model} ...", file=sys.stderr)
    t0 = time.time()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        args.teacher_model, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.teacher_model)
    processor.tokenizer.padding_side = "left"   # required for batched generation
    print(f"[teacher] loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    mode = "a" if args.resume and out_path.exists() else "w"
    B = max(1, int(args.batch_size))
    with open(out_path, mode) as outf:
        for start in range(0, len(chosen), B):
            batch = chosen[start : start + B]
            messages_list = []
            images = []
            for r in batch:
                fp = Path(args.frames_root) / r["frame_path"]
                try:
                    img = Image.open(fp).convert("RGB")
                except Exception:
                    img = None
                images.append(img)
                messages_list.append([{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img or Image.new("RGB", (32, 32))},
                        {"type": "text", "text": TEACHER_PROMPT},
                    ],
                }])
            try:
                inputs = processor.apply_chat_template(
                    messages_list, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt", padding=True,
                ).to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                    )
                in_len = inputs["input_ids"].shape[1]
                replies = processor.batch_decode(out[:, in_len:], skip_special_tokens=True)
            except Exception as e:
                replies = [f"__ERROR__: {type(e).__name__}: {e}"] * len(batch)

            for r, img, reply in zip(batch, images, replies):
                if img is None:
                    reply = "__ERROR__: image_open_failed"
                parsed = parse_teacher_json(reply) if not reply.startswith("__ERROR__") else {}
                gt = labels.get(r["video_id"], {})
                rec = {
                    "frame_path": r["frame_path"],
                    "video_id": r["video_id"],
                    "split": r.get("split"),
                    "time_s": r.get("time_s"),
                    "gt_classification": gt.get("classification"),
                    "gt_is_sludge": gt.get("is_sludge"),
                    "gt_layout": gt.get("layout_category"),
                    "teacher_raw": reply,
                    "teacher": parsed,
                }
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            outf.flush()

            done = start + len(batch)
            dt = time.time() - t0
            if done % (B * 5) == 0 or done == len(chosen):
                eta = (len(chosen) - done) * dt / max(1, done) / 60
                print(f"[teacher] {done}/{len(chosen)}  {dt/max(1,done):.2f}s/frame  "
                      f"eta={eta:.1f} min", file=sys.stderr)

    print(f"[teacher] done. wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())

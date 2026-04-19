"""Label the held-out test split (300 videos × 1 mid-frame each) with the
Qwen3-VL-Instruct teacher. Uses the existing manifest at
/home/alpha/vqwen/data/tiktok_sludge_frames/frames_manifest.jsonl which covers
all splits (train/validate/test). We filter to split == 'test' and append to
teacher_labels_test.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Reuse the prompt + parser from the train/validate labeler.
from importlib.util import module_from_spec, spec_from_file_location  # noqa: E402

_spec = spec_from_file_location("lt", REPO_ROOT / "scripts" / "16_label_with_teacher.py")
_lt = module_from_spec(_spec); _spec.loader.exec_module(_lt)
TEACHER_PROMPT = _lt.TEACHER_PROMPT
parse_teacher_json = _lt.parse_teacher_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_model", default=str(REPO_ROOT / "models" / "Qwen3-VL-30B-A3B-Instruct"))
    p.add_argument("--manifest", default="/home/alpha/vqwen/data/tiktok_sludge_frames/frames_manifest.jsonl")
    p.add_argument("--frames_root", default="/home/alpha/vqwen/data/tiktok_sludge_frames")
    p.add_argument("--labels_src", default="/home/alpha/vqwen/data/tiktok_sludge")
    p.add_argument("--output", default=str(REPO_ROOT / "data" / "tiktok_fps1" / "teacher_labels_test.jsonl"))
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    # Load GT labels
    labels = {}
    for sp in ("train", "validate", "test"):
        for r in json.load(open(Path(args.labels_src) / "split" / f"{sp}.json")):
            labels[r["id"]] = {"classification": r["classification"], "split": sp}
    for line in open(Path(args.labels_src) / "enriched_classifications.jsonl"):
        r = json.loads(line)
        vid = r["video_id"]
        if vid in labels:
            labels[vid]["layout_category"] = r.get("layout_category", "")
            labels[vid]["is_sludge"] = r.get("is_sludge", None)

    # Filter manifest to test split only
    chosen = []
    with open(args.manifest) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test":
                chosen.append(r)
    if args.limit > 0:
        chosen = chosen[: args.limit]
    print(f"[test-teacher] labeling {len(chosen)} held-out test frames", file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        for l in open(out_path):
            try: done.add(json.loads(l)["frame_path"])
            except: pass
        chosen = [r for r in chosen if r["frame_path"] not in done]
        print(f"[test-teacher] resume: {len(done)} done, {len(chosen)} remaining", file=sys.stderr)

    print(f"[test-teacher] loading {args.teacher_model}", file=sys.stderr)
    t0 = time.time()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        args.teacher_model, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.teacher_model)
    print(f"[test-teacher] loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    mode = "a" if args.resume and out_path.exists() else "w"
    with open(out_path, mode) as outf:
        for i, r in enumerate(chosen):
            fp = Path(args.frames_root) / r["frame_path"]
            try:
                image = Image.open(fp).convert("RGB")
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": TEACHER_PROMPT},
                ]}]
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                          do_sample=False,
                                          pad_token_id=processor.tokenizer.pad_token_id,
                                          eos_token_id=processor.tokenizer.eos_token_id)
                reply = processor.batch_decode(
                    out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )[0]
                parsed = parse_teacher_json(reply)
            except Exception as e:
                reply = f"__ERROR__: {type(e).__name__}: {e}"
                parsed = {}

            gt = labels.get(r["video_id"], {})
            outf.write(json.dumps({
                "frame_path": r["frame_path"],
                "video_id": r["video_id"],
                "split": "test",
                "gt_classification": gt.get("classification"),
                "gt_is_sludge": gt.get("is_sludge"),
                "gt_layout": gt.get("layout_category"),
                "teacher_raw": reply,
                "teacher": parsed,
            }) + "\n")
            outf.flush()
            if (i + 1) % 25 == 0:
                dt = time.time() - t0
                eta = (len(chosen) - i - 1) * dt / (i + 1) / 60
                print(f"[test-teacher] {i+1}/{len(chosen)}  {dt/(i+1):.2f}s  eta={eta:.1f} min", file=sys.stderr)

    print(f"[test-teacher] done -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())

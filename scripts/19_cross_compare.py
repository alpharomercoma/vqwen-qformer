"""Use Gemma-4-31B as a tiebreaker judge to cross-compare teacher vs GT on
disputed TikTok frames. For each disagreement (teacher_labels.jsonl rows where
teacher.sludge != gt_is_sludge), ask Gemma to rule.

Outputs:
  data/tiktok_fps1/cross_compare.jsonl   # per-frame judgment
  data/tiktok_fps1/cross_compare_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

JUDGE_PROMPT = """You are a strict judge deciding whether a short-video frame is "sludge content."

DEFINITION: Sludge = multiple UNRELATED video streams shown simultaneously in the same frame (split-screen, picture-in-picture, or collage). A single scene with text captions / subtitles / "Part X/Y" overlays is NOT sludge — it's still one video. Two distinct video streams with different content at the same time IS sludge.

Analyze this frame and respond with STRICT JSON ONLY (no extra prose):

{
  "sludge": <"yes"|"no">,
  "reason": "<one short sentence grounded in what you visibly see in the frame>"
}"""


def yn(s): return (s or "").strip().lower().startswith("y")


def parse_json(text: str) -> dict:
    t = text.strip()
    if "</think>" in t: t = t.rsplit("</think>", 1)[1].strip()
    t = t.replace("```json", "").replace("```", "").strip()
    lbrace, rbrace = t.find("{"), t.rfind("}")
    if lbrace == -1 or rbrace == -1 or rbrace <= lbrace: return {}
    try: return json.loads(t[lbrace : rbrace + 1])
    except json.JSONDecodeError: return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge_model", default=str(REPO_ROOT / "models" / "gemma-3-27b-it"))
    p.add_argument("--teacher_labels", nargs="+", default=[
        str(REPO_ROOT / "data" / "tiktok_fps1" / "teacher_labels.jsonl"),        # train+validate (frames in tiktok_fps1)
        str(REPO_ROOT / "data" / "tiktok_fps1" / "teacher_labels_test.jsonl"),   # test (frames in vqwen tiktok_sludge_frames)
    ])
    p.add_argument("--frames_root_train", default=str(REPO_ROOT / "data" / "tiktok_fps1" / "frames"))
    p.add_argument("--frames_root_test", default="/home/alpha/vqwen/data/tiktok_sludge_frames")
    p.add_argument("--out_jsonl", default=str(REPO_ROOT / "data" / "tiktok_fps1" / "cross_compare.jsonl"))
    p.add_argument("--summary", default=str(REPO_ROOT / "data" / "tiktok_fps1" / "cross_compare_summary.json"))
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--include_agreements", action="store_true",
                   help="Also judge frames where teacher and GT agree (sanity check)")
    args = p.parse_args()

    # Collect from multiple label files (train/validate + test)
    all_recs = []
    for lf in args.teacher_labels:
        source = "test" if "test" in lf else "train_validate"
        for l in open(lf):
            r = json.loads(l)
            r["_label_source"] = source
            all_recs.append(r)
    disputes = []
    for r in all_recs:
        gt = bool(r.get("gt_is_sludge"))
        teacher = (r.get("teacher") or {}).get("sludge")
        if not teacher: continue
        t = yn(teacher)
        if t != gt or args.include_agreements:
            disputes.append(r)
    by_src = {}
    for r in disputes:
        by_src[r["_label_source"]] = by_src.get(r["_label_source"], 0) + 1
    print(f"[judge] {len(disputes)} disputes to judge  ({by_src})", file=sys.stderr)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        for l in open(out_path):
            try: done.add(json.loads(l)["frame_path"])
            except: pass
        disputes = [r for r in disputes if r["frame_path"] not in done]
        print(f"[judge] resume: {len(done)} done, {len(disputes)} remaining", file=sys.stderr)

    print(f"[judge] loading {args.judge_model} ...", file=sys.stderr)
    t0 = time.time()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        args.judge_model, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.judge_model)
    print(f"[judge] loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    mode = "a" if args.resume and out_path.exists() else "w"
    with open(out_path, mode) as outf:
        for i, r in enumerate(disputes):
            root = args.frames_root_test if r["_label_source"] == "test" else args.frames_root_train
            fp = Path(root) / r["frame_path"]
            try:
                image = Image.open(fp).convert("RGB")
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": JUDGE_PROMPT},
                    ],
                }]
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
                judged = parse_json(reply)
            except Exception as e:
                reply = f"__ERROR__: {type(e).__name__}: {e}"
                judged = {}

            outf.write(json.dumps({
                "frame_path": r["frame_path"],
                "video_id": r["video_id"],
                "split_source": r.get("_label_source"),
                "gt_is_sludge": r.get("gt_is_sludge"),
                "teacher_sludge": (r.get("teacher") or {}).get("sludge"),
                "teacher_layout": (r.get("teacher") or {}).get("layout"),
                "teacher_desc": (r.get("teacher") or {}).get("description"),
                "judge_sludge": judged.get("sludge"),
                "judge_reason": judged.get("reason"),
                "judge_raw": reply,
            }) + "\n")
            outf.flush()
            if (i + 1) % 10 == 0:
                dt = time.time() - t0
                eta = (len(disputes) - i - 1) * dt / (i + 1) / 60
                print(f"[judge] {i+1}/{len(disputes)}  {dt/(i+1):.2f}s  eta={eta:.1f} min", file=sys.stderr)

    # Aggregate
    recs = [json.loads(l) for l in open(out_path)]
    agree_teacher = agree_gt = tie = err = 0
    for rr in recs:
        j = rr.get("judge_sludge")
        if not j:
            err += 1; continue
        jy = yn(j)
        ty = yn(rr["teacher_sludge"]) if rr.get("teacher_sludge") else None
        gy = bool(rr["gt_is_sludge"])
        if ty is None: continue
        if jy == ty and jy != gy: agree_teacher += 1
        elif jy == gy and jy != ty: agree_gt += 1
        else: tie += 1

    summary = {
        "n_disputes_judged": len(recs), "errors": err,
        "judge_sides_with_teacher": agree_teacher,
        "judge_sides_with_gt": agree_gt,
        "judge_agrees_with_both_or_neither": tie,
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())

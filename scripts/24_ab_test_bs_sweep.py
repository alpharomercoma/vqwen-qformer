"""Sweep batch sizes {1, 4, 8, 16} on the same 40 frames.

Reports speedup + decision match per batch size vs serial baseline.
Identifies the sweet spot where speedup plateaus or quality drops.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from importlib.util import module_from_spec, spec_from_file_location

_spec = spec_from_file_location("lt", REPO_ROOT / "scripts" / "16_label_with_teacher.py")
_lt = module_from_spec(_spec); _spec.loader.exec_module(_lt)
TEACHER_PROMPT = _lt.TEACHER_PROMPT
parse_teacher_json = _lt.parse_teacher_json

_spec2 = spec_from_file_location("ab", REPO_ROOT / "scripts" / "23_ab_test_batching.py")
_ab = module_from_spec(_spec2); _spec2.loader.exec_module(_ab)
pick_40_frames = _ab.pick_40_frames


def label_with_bs(model, processor, frames, frames_root, bs, max_new_tokens=1024):
    """Label a list of frames with given batch size. Returns list of raw replies."""
    replies = []
    for i in range(0, len(frames), bs):
        batch = frames[i:i + bs]
        images = [Image.open(frames_root / f["frame_path"]).convert("RGB") for f in batch]
        if bs == 1:
            # single-sample path (no padding)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": images[0]},
                {"type": "text", "text": TEACHER_PROMPT},
            ]}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
        else:
            messages_list = [[{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": TEACHER_PROMPT},
            ]}] for img in images]
            inputs = processor.apply_chat_template(
                messages_list, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True,
            ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=processor.tokenizer.pad_token_id,
                                  eos_token_id=processor.tokenizer.eos_token_id)
        in_len = inputs["input_ids"].shape[1]
        r = processor.batch_decode(out[:, in_len:], skip_special_tokens=True)
        replies.extend(r)
    return replies


def compare(baseline, test, label):
    n = len(baseline)
    byte = parsed = sludge = layout = 0
    for bs_r, t_r in zip(baseline, test):
        if bs_r == t_r: byte += 1
        pb, pt = parse_teacher_json(bs_r), parse_teacher_json(t_r)
        if pb == pt: parsed += 1
        if pb.get("sludge") == pt.get("sludge"): sludge += 1
        if pb.get("layout") == pt.get("layout"): layout += 1
    return {
        "label": label,
        "byte_identical_pct": round(byte * 100 / n, 1),
        "json_equal_pct": round(parsed * 100 / n, 1),
        "sludge_match_pct": round(sludge * 100 / n, 1),
        "layout_match_pct": round(layout * 100 / n, 1),
    }


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    teacher = REPO_ROOT / "models" / "Qwen3-VL-30B-A3B-Instruct"
    frames_root = REPO_ROOT / "data" / "tiktok_fps1" / "frames"

    print("[sweep] picking 40 frames...", file=sys.stderr)
    frames = pick_40_frames()
    print(f"[sweep] got {len(frames)}", file=sys.stderr)

    print("[sweep] loading teacher...", file=sys.stderr)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        teacher, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(teacher)
    processor.tokenizer.padding_side = "left"
    print(f"[sweep] loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    results = {}
    runs = {}
    for bs in [1, 4, 8, 16]:
        print(f"[sweep] bs={bs} ...", file=sys.stderr)
        t0 = time.time()
        try:
            replies = label_with_bs(model, processor, frames, frames_root, bs)
            dt = time.time() - t0
            runs[bs] = replies
            results[bs] = {"time_s": round(dt, 1), "s_per_frame": round(dt / len(frames), 3)}
            print(f"[sweep] bs={bs} done: {dt:.1f}s ({dt/len(frames):.2f}s/frame)", file=sys.stderr)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[sweep] bs={bs} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            results[bs] = {"error": f"{type(e).__name__}: {e}"}
            break

    # Speedup + quality vs bs=1 baseline
    if 1 in runs:
        baseline = runs[1]
        t_serial = results[1]["time_s"]
        for bs in [4, 8, 16]:
            if bs in runs:
                results[bs]["speedup_vs_serial"] = round(t_serial / results[bs]["time_s"], 2)
                results[bs].update(compare(baseline, runs[bs], f"bs={bs}"))

    out = REPO_ROOT / "data" / "tiktok_fps1" / "bs_sweep_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

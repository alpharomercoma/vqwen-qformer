"""AB-test batched (bs=4) vs serial (bs=1) Qwen3-VL-Instruct labeling on the same 40 frames.

Reports:
  - % of frames where parsed JSON fields (sludge, layout) are identical
  - % where description is byte-identical
  - % where description differs but classify matches
  - raw differ samples for manual inspection
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

from importlib.util import module_from_spec, spec_from_file_location  # noqa: E402

_spec = spec_from_file_location("lt", REPO_ROOT / "scripts" / "16_label_with_teacher.py")
_lt = module_from_spec(_spec); _spec.loader.exec_module(_lt)
TEACHER_PROMPT = _lt.TEACHER_PROMPT
parse_teacher_json = _lt.parse_teacher_json


def pick_40_frames():
    import collections, random
    from importlib.util import module_from_spec, spec_from_file_location
    # Mirror script 16's frame selection to pick same deterministic set
    manifest = REPO_ROOT / "data" / "tiktok_fps1" / "frames_manifest.jsonl"
    labels = {}
    for sp in ("train", "validate", "test"):
        for r in json.load(open("/home/alpha/vqwen/data/tiktok_sludge/split/" + f"{sp}.json")):
            labels[r["id"]] = {"split": sp}
    recs = [json.loads(l) for l in open(manifest)]
    by_vid = collections.defaultdict(list)
    for r in recs:
        sp = labels.get(r["video_id"], {}).get("split")
        if sp not in ("train", "validate"): continue
        r["split"] = sp
        by_vid[r["video_id"]].append(r)

    rng = random.Random(42)
    already_done = set()
    for l in open(REPO_ROOT / "data" / "tiktok_fps1" / "teacher_labels.jsonl"):
        already_done.add(json.loads(l)["frame_path"])

    chosen = []
    for vid in sorted(by_vid):
        frames = by_vid[vid][:]
        rng.shuffle(frames)
        for f in frames[:5]:
            if f["frame_path"] not in already_done:
                chosen.append(f)
                if len(chosen) >= 40:
                    return chosen
    return chosen


def label_one(model, processor, image, max_new_tokens):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": TEACHER_PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=processor.tokenizer.pad_token_id,
                              eos_token_id=processor.tokenizer.eos_token_id)
    reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return reply


def label_batched(model, processor, images, max_new_tokens):
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
    return processor.batch_decode(out[:, in_len:], skip_special_tokens=True)


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    teacher = REPO_ROOT / "models" / "Qwen3-VL-30B-A3B-Instruct"
    frames_root = REPO_ROOT / "data" / "tiktok_fps1" / "frames"

    print("[ab] picking 40 frames...", file=sys.stderr)
    frames = pick_40_frames()
    print(f"[ab] got {len(frames)} frames", file=sys.stderr)

    print("[ab] loading teacher...", file=sys.stderr)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        teacher, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(teacher)
    processor.tokenizer.padding_side = "left"
    print(f"[ab] loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    # SERIAL
    print("[ab] serial (batch=1)...", file=sys.stderr)
    t0 = time.time()
    serial_replies = []
    for f in frames:
        img = Image.open(frames_root / f["frame_path"]).convert("RGB")
        serial_replies.append(label_one(model, processor, img, 1024))
    t_serial = time.time() - t0
    print(f"[ab] serial done: {t_serial:.1f}s = {t_serial/len(frames):.2f}s/frame", file=sys.stderr)

    # BATCHED bs=4
    print("[ab] batched (batch=4)...", file=sys.stderr)
    t0 = time.time()
    batched_replies = []
    for i in range(0, len(frames), 4):
        batch = frames[i:i + 4]
        images = [Image.open(frames_root / f["frame_path"]).convert("RGB") for f in batch]
        batched_replies.extend(label_batched(model, processor, images, 1024))
    t_batched = time.time() - t0
    print(f"[ab] batched done: {t_batched:.1f}s = {t_batched/len(frames):.2f}s/frame  "
          f"(speedup = {t_serial/t_batched:.2f}x)", file=sys.stderr)

    # COMPARE
    n = len(frames)
    byte_identical = 0
    parsed_json_equal = 0
    sludge_equal = 0
    layout_equal = 0
    diffs = []
    for i, f in enumerate(frames):
        s, b = serial_replies[i], batched_replies[i]
        ps, pb = parse_teacher_json(s), parse_teacher_json(b)
        if s == b: byte_identical += 1
        if ps == pb: parsed_json_equal += 1
        if ps.get("sludge") == pb.get("sludge"): sludge_equal += 1
        if ps.get("layout") == pb.get("layout"): layout_equal += 1
        if ps != pb:
            diffs.append({"frame_path": f["frame_path"], "serial": ps, "batched": pb})

    out = REPO_ROOT / "data" / "tiktok_fps1" / "ab_test_results.json"
    summary = {
        "n": n,
        "serial_time_s": round(t_serial, 1),
        "batched_time_s": round(t_batched, 1),
        "speedup": round(t_serial / t_batched, 2),
        "byte_identical": byte_identical,
        "parsed_json_equal": parsed_json_equal,
        "sludge_decisions_match": sludge_equal,
        "layout_decisions_match": layout_equal,
        "pct_byte_identical": round(byte_identical * 100 / n, 1),
        "pct_json_equal": round(parsed_json_equal * 100 / n, 1),
        "pct_sludge_match": round(sludge_equal * 100 / n, 1),
        "pct_layout_match": round(layout_equal * 100 / n, 1),
        "diffs": diffs[:5],
    }
    with open(out, "w") as fout:
        json.dump(summary, fout, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Build TikTok stage-2 conversations.

Inputs:
  data/tiktok_v2/frames_manifest.jsonl
  data/tiktok_v2/kaggle_root/text/{batch}/{video_id}.json
  data/tiktok_v2/kaggle_root/labels/labels_corrected.jsonl   (optional)

Outputs:
  data/tiktok_v2/tiktok_train.json
  data/tiktok_v2/tiktok_validate.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "tiktok_v2"
KAGGLE_ROOT = DATA_ROOT / "kaggle_root"
TEXT_ROOT = KAGGLE_ROOT / "text"

MAX_TRANSCRIPT_CHARS = 600  # ~150 Qwen tokens

CLS_Q = [
    "Is this sludge content?",
    "Does this image show two or more unrelated visual scenes displayed at the same time (split-screen, picture-in-picture, or collage)?",
    "Is this a sludge content layout?",
    "Does this frame contain more than one unrelated visual stream?",
    "Is this frame from a sludge-style video?",
    "Does the image simultaneously display unrelated videos side-by-side or stacked?",
    "Would you classify this as sludge content?",
    "Are there multiple panes showing different, unrelated content in this frame?",
    "Is the layout here sludge?",
    "Does this frame visually combine multiple unrelated clips?",
]
LAYOUT_Q = [
    "What layout type is shown?",
    "Identify the visual composition.",
    "How is this frame composed?",
    "What is the layout category?",
    "Classify the image layout.",
    "Which layout best describes this frame?",
]
DESC_Q = [
    "Describe this frame.",
    "What is visible in this frame?",
    "Describe the visual content.",
    "Describe the layout and content of this frame.",
    "What does this frame show?",
]
EXPLAIN_Q = ["Explain why.", "Justify your answer.", "Why?", "Explain briefly.", "What's your reasoning?"]
REFUSE_Q = [
    "What specific TV show is on the top pane?",
    "What game is being played in this frame?",
    "Who is the person in this video?",
    "Name the specific movie/show/game featured here.",
    "What is the channel or creator behind this video?",
]
REFUSE_A = (
    "I cannot reliably identify the specific show, game, or creator from this single frame. "
    "{grounded}"
)


def _load_transcript(text_json_path: Path) -> str:
    if not text_json_path.exists():
        return ""
    try:
        with open(text_json_path) as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    if isinstance(obj, dict):
        t = obj.get("text")
        if isinstance(t, str) and t.strip():
            return t.strip()
        segs = obj.get("segments")
        if isinstance(segs, list):
            parts = [s["text"].strip() for s in segs
                     if isinstance(s, dict) and isinstance(s.get("text"), str) and s["text"].strip()]
            if parts:
                return " ".join(parts)
    elif isinstance(obj, list):
        parts = [s["text"].strip() for s in obj
                 if isinstance(s, dict) and isinstance(s.get("text"), str) and s["text"].strip()]
        if parts:
            return " ".join(parts)
    return ""


def _truncate(t: str, n: int = MAX_TRANSCRIPT_CHARS) -> str:
    if len(t) <= n:
        return t
    cut = t.rfind(" ", 0, n)
    return (t[:cut] if cut > 0 else t[:n]).rstrip() + "…"


def _layout_str(rec) -> str:
    lc = (rec.get("layout_category") or "").strip()
    return lc or "other"


def _is_sludge(rec) -> bool:
    v = rec.get("is_sludge")
    if v is None:
        v = rec.get("classification")
    return bool(v)


def _human_value(transcript: str, image_q: str) -> str:
    head = f"Audio transcript: {transcript}\n" if transcript else ""
    return f"{head}<image>\n{image_q}"


def build_sample(rec, task: str, transcript: str, rng: random.Random,
                 p_drop_transcript: float):
    # Sample-level transcript dropout, train only.
    transcript_used = transcript
    if rec["split"] == "train" and transcript and rng.random() < p_drop_transcript:
        transcript_used = ""

    summary = (rec.get("summary") or "").strip()
    layout = _layout_str(rec)
    is_sludge = _is_sludge(rec)
    frame_path = rec["frame_path"].replace("frames/", "")
    sample_id = f"tiktok_{rec['video_id']}_f{int(rec.get('time_s') or 0)}_{task}"

    if task == "classify":
        q = rng.choice(CLS_Q)
        a = "Yes." if is_sludge else "No."
        conv = [{"from": "human", "value": _human_value(transcript_used, q)},
                {"from": "gpt",   "value": a}]
    elif task == "layout":
        q = rng.choice(LAYOUT_Q)
        conv = [{"from": "human", "value": _human_value(transcript_used, q)},
                {"from": "gpt",   "value": layout}]
    elif task == "describe":
        q = rng.choice(DESC_Q)
        a = summary or ("This frame shows a sludge layout." if is_sludge
                        else "This frame shows a single continuous scene.")
        conv = [{"from": "human", "value": _human_value(transcript_used, q)},
                {"from": "gpt",   "value": a}]
    elif task == "coupled":
        q1 = rng.choice(CLS_Q); a1 = "Yes." if is_sludge else "No."
        q2 = rng.choice(EXPLAIN_Q)
        prefix = "Yes, this is sludge. " if is_sludge else "No, this is not sludge. "
        a2 = prefix + (summary or "")
        conv = [
            {"from": "human", "value": _human_value(transcript_used, q1)},
            {"from": "gpt",   "value": a1},
            {"from": "human", "value": q2},
            {"from": "gpt",   "value": a2.rstrip()},
        ]
    elif task == "refuse":
        q = rng.choice(REFUSE_Q)
        grounded = summary or (
            "The frame shows a sludge layout." if is_sludge
            else "The frame shows a single continuous scene."
        )
        a = REFUSE_A.format(grounded=grounded)
        conv = [{"from": "human", "value": _human_value(transcript_used, q)},
                {"from": "gpt",   "value": a}]
    else:
        raise ValueError(f"unknown task {task}")

    return {
        "id": sample_id,
        "image": frame_path,
        "conversations": conv,
        "_meta": {
            "split": rec["split"], "task": task,
            "layout": layout, "is_sludge": is_sludge,
            "has_transcript": bool(transcript_used),
            "transcript_dropped": bool(transcript) and not transcript_used,
        },
    }


def load_corrected_labels(path: Optional[Path]) -> Dict[str, bool]:
    if path is None or not path.exists():
        return {}
    out: Dict[str, bool] = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            v = d.get("recommended_is_sludge")
            if v is None:
                continue
            out[d["video_id"]] = bool(v)
    return out


def main():
    p = argparse.ArgumentParser()
    # The canonical train+validate manifest lives under kaggle_root/labels/;
    # the file directly under data/tiktok_v2/ was rewritten as test-only at
    # some point and lacks train/val frames. Use the labels copy as source.
    p.add_argument("--manifest", default=str(KAGGLE_ROOT / "labels" / "frames_manifest.jsonl"))
    p.add_argument("--out_dir", default=str(DATA_ROOT))
    p.add_argument("--task_weights",
                   default="classify:25,layout:15,describe:20,coupled:35,refuse:5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--p_drop_transcript", type=float, default=0.4,
                   help="Probability of blanking the transcript on a train sample.")
    p.add_argument("--labels_corrected",
                   default=str(KAGGLE_ROOT / "labels" / "labels_corrected.jsonl"),
                   help="Path to labels_corrected.jsonl; set to '' to disable.")
    p.add_argument("--out_suffix", default="",
                   help="Suffix appended to tiktok_{split} output filenames.")
    args = p.parse_args()

    corrected_path = Path(args.labels_corrected) if args.labels_corrected else None
    corrected = load_corrected_labels(corrected_path)
    print(f"[build] corrected labels loaded: {len(corrected)}", file=sys.stderr)

    weights = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in args.task_weights.split(",")}
    pool: list[str] = []
    for k, w in weights.items():
        pool.extend([k] * w)

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    per_split: dict[str, list] = {"train": [], "validate": []}
    task_counts: collections.Counter = collections.Counter()
    transcript_cache: dict[str, str] = {}
    n_with_transcript = n_without = 0
    n_label_overridden = 0
    n_dropout_fired = 0

    with open(args.manifest) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") not in per_split:
                continue

            # Corrected-label override (per video_id).
            if r["video_id"] in corrected:
                new_v = corrected[r["video_id"]]
                if new_v != _is_sludge(r):
                    n_label_overridden += 1
                r["is_sludge"] = new_v
                r["classification"] = new_v

            vid = r["video_id"]
            if vid not in transcript_cache:
                # labels/frames_manifest.jsonl doesn't carry transcript_path;
                # derive it as {batch}/{video_id}.json under TEXT_ROOT.
                tp = r.get("transcript_path") or f"{r['batch']}/{vid}.json"
                transcript_cache[vid] = _truncate(_load_transcript(TEXT_ROOT / tp))
            transcript = transcript_cache[vid]
            if transcript:
                n_with_transcript += 1
            else:
                n_without += 1

            task = rng.choice(pool)
            task_counts[task] += 1
            sample = build_sample(r, task, transcript, rng, args.p_drop_transcript)
            if sample["_meta"]["transcript_dropped"]:
                n_dropout_fired += 1
            per_split[r["split"]].append(sample)

    for sp, samples in per_split.items():
        fp = out_dir / f"tiktok_{sp}{args.out_suffix}.json"
        with open(fp, "w") as f:
            json.dump(samples, f)
        n_train_with = sum(1 for s in samples if s["_meta"]["has_transcript"])
        n_train_without = len(samples) - n_train_with
        print(f"[build] {sp}: {len(samples)} -> {fp}  "
              f"(transcript_present={n_train_with}, blanked={n_train_without})",
              file=sys.stderr)

    print(f"[build] tasks: {dict(task_counts)}", file=sys.stderr)
    print(f"[build] raw transcript coverage (pre-dropout): with={n_with_transcript} without={n_without}",
          file=sys.stderr)
    print(f"[build] transcript dropout fired: {n_dropout_fired} samples (p={args.p_drop_transcript})",
          file=sys.stderr)
    print(f"[build] label overrides applied: {n_label_overridden}", file=sys.stderr)


if __name__ == "__main__":
    main()

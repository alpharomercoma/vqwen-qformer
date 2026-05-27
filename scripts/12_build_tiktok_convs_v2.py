"""v2 — build TikTok stage-2 conversations *with* the audio transcript modality.

Differences from the ablation builder (`12_build_tiktok_convs_ablation.py`):
  * No teacher / no distillation. Labels come from the Kaggle release directly:
      - `is_sludge`, `layout_category`, `summary` (from `enriched_classifications.jsonl`)
      - `split` (from `split/{train,validate,test}.json`)
  * Each human turn is prefixed with `Audio transcript: <text>\n`, where <text>
    is loaded from `kaggle_root/text/{batch}/{video_id}.json` (Whisper-V3-Turbo
    output, segments-style schema by default, with a fallback to a flat
    `{"text": ...}` shape).
  * The transcript is cached per video_id, not per frame.
  * Transcripts are truncated to ~600 characters to keep total prompt under the
    1024-token `model_max_length` budget after counting the 32 image tokens.

Inputs:
  data/tiktok_v2/frames_manifest.jsonl    (from 11_extract_tiktok_1fps_v2.py)
  data/tiktok_v2/kaggle_root/text/{batch}/{video_id}.json

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

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "tiktok_v2"
KAGGLE_ROOT = DATA_ROOT / "kaggle_root"
TEXT_ROOT = KAGGLE_ROOT / "text"

MAX_TRANSCRIPT_CHARS = 600  # ~150 Qwen tokens; leaves room for image + Q + A

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
    """Return a flat string for the transcript. Empty string if missing/unparseable.

    The Kaggle dataset ships per-video Whisper-V3-Turbo output. We probe for
    both common shapes:
        {"text": "the full transcript"}                          (flat dump)
        {"segments": [{"text": "...", "start_time": 0.0, ...}]}  (segments-style)
    """
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
            parts = []
            for s in segs:
                if isinstance(s, dict):
                    seg_t = s.get("text")
                    if isinstance(seg_t, str) and seg_t.strip():
                        parts.append(seg_t.strip())
            if parts:
                return " ".join(parts)
    elif isinstance(obj, list):
        # Edge case: top-level list of segments.
        parts = []
        for s in obj:
            if isinstance(s, dict):
                seg_t = s.get("text")
                if isinstance(seg_t, str) and seg_t.strip():
                    parts.append(seg_t.strip())
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


def build_sample(rec, task: str, transcript: str, rng: random.Random):
    summary = (rec.get("summary") or "").strip()
    layout = _layout_str(rec)
    is_sludge = _is_sludge(rec)
    frame_path = rec["frame_path"].replace("frames/", "")
    sample_id = f"tiktok_{rec['video_id']}_f{int(rec.get('time_s') or 0)}_{task}"

    if task == "classify":
        q = rng.choice(CLS_Q)
        a = "Yes." if is_sludge else "No."
        conv = [{"from": "human", "value": _human_value(transcript, q)},
                {"from": "gpt",   "value": a}]
    elif task == "layout":
        q = rng.choice(LAYOUT_Q)
        conv = [{"from": "human", "value": _human_value(transcript, q)},
                {"from": "gpt",   "value": layout}]
    elif task == "describe":
        q = rng.choice(DESC_Q)
        a = summary or ("This frame shows a sludge layout." if is_sludge
                        else "This frame shows a single continuous scene.")
        conv = [{"from": "human", "value": _human_value(transcript, q)},
                {"from": "gpt",   "value": a}]
    elif task == "coupled":
        q1 = rng.choice(CLS_Q); a1 = "Yes." if is_sludge else "No."
        q2 = rng.choice(EXPLAIN_Q)
        prefix = "Yes, this is sludge. " if is_sludge else "No, this is not sludge. "
        a2 = prefix + (summary or "")
        conv = [
            {"from": "human", "value": _human_value(transcript, q1)},
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
        conv = [{"from": "human", "value": _human_value(transcript, q)},
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
            "has_transcript": bool(transcript),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(DATA_ROOT / "frames_manifest.jsonl"))
    p.add_argument("--out_dir", default=str(DATA_ROOT))
    p.add_argument("--task_weights",
                   default="classify:25,layout:15,describe:20,coupled:35,refuse:5")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    weights = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in args.task_weights.split(",")}
    pool = []
    for k, w in weights.items(): pool.extend([k] * w)

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    per_split = {"train": [], "validate": []}
    task_counts = collections.Counter()
    transcript_cache: dict[str, str] = {}
    n_with_transcript = n_without = 0

    with open(args.manifest) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") not in per_split:
                continue
            vid = r["video_id"]
            if vid not in transcript_cache:
                tp = r.get("transcript_path")
                if tp:
                    transcript_cache[vid] = _truncate(_load_transcript(TEXT_ROOT / tp))
                else:
                    transcript_cache[vid] = ""
            transcript = transcript_cache[vid]
            if transcript: n_with_transcript += 1
            else:          n_without += 1
            task = rng.choice(pool); task_counts[task] += 1
            per_split[r["split"]].append(build_sample(r, task, transcript, rng))

    for sp, samples in per_split.items():
        fp = out_dir / f"tiktok_{sp}.json"
        with open(fp, "w") as f: json.dump(samples, f)
        print(f"[build-v2] {sp}: {len(samples)} -> {fp}", file=sys.stderr)
    print(f"[build-v2] tasks: {dict(task_counts)}", file=sys.stderr)
    print(f"[build-v2] transcript coverage: with={n_with_transcript} without={n_without}",
          file=sys.stderr)


if __name__ == "__main__":
    main()

"""v6 — same as v5 builder, but uses TEACHER'S sludge decision instead of GT
for both the classify task and the coupled-explain answer.

Rationale: Gemma-3 judged 250 disputed frames → sided with teacher 232 times
(92.8%), with GT only 18 times. Human GT has ~10-12% label noise by the strict
sludge definition; teacher (Qwen3-VL-Instruct) is closer to truth.

Signal alignment: since describe_teacher and coupled.explain already use
teacher's description/explanation, switching classify to teacher eliminates
the v5 signal conflict where classify:"Yes" clashed with describe:"single scene".
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

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


def teacher_is_sludge(teacher_dict) -> bool:
    s = (teacher_dict or {}).get("sludge") or ""
    return s.strip().lower().startswith("y")


def build_sample(rec, task: str, rng: random.Random):
    teacher = rec.get("teacher") or {}
    is_sludge = teacher_is_sludge(teacher)                      # ← teacher, not GT
    teacher_layout = (teacher.get("layout") or "").strip() or "other"
    teacher_desc = (teacher.get("description") or "").strip()
    teacher_expl = (teacher.get("explanation") or "").strip()
    frame_path = rec["frame_path"].replace("frames/", "")  # tolerate either root

    sample_id = f"tiktok_{rec['video_id']}_f{int(rec.get('time_s') or 0)}_{task}"
    if task == "classify":
        q = rng.choice(CLS_Q); a = "Yes." if is_sludge else "No."
        conv = [{"from": "human", "value": "<image>\n" + q}, {"from": "gpt", "value": a}]
    elif task == "layout":
        q = rng.choice(LAYOUT_Q); a = teacher_layout
        conv = [{"from": "human", "value": "<image>\n" + q}, {"from": "gpt", "value": a}]
    elif task == "describe_teacher":
        q = rng.choice(DESC_Q); a = teacher_desc
        conv = [{"from": "human", "value": "<image>\n" + q}, {"from": "gpt", "value": a}]
    elif task == "coupled":
        q1 = rng.choice(CLS_Q); a1 = "Yes." if is_sludge else "No."
        q2 = rng.choice(EXPLAIN_Q)
        prefix = "Yes, this is sludge. " if is_sludge else "No, this is not sludge. "
        a2 = prefix + (teacher_expl or teacher_desc)
        conv = [
            {"from": "human", "value": "<image>\n" + q1},
            {"from": "gpt",   "value": a1},
            {"from": "human", "value": q2},
            {"from": "gpt",   "value": a2},
        ]
    elif task == "refuse":
        q = rng.choice(REFUSE_Q)
        grounded = teacher_desc or \
            ("The frame shows a sludge layout." if is_sludge else "The frame shows a single continuous scene.")
        a = REFUSE_A.format(grounded=grounded)
        conv = [{"from": "human", "value": "<image>\n" + q}, {"from": "gpt", "value": a}]
    else:
        raise ValueError(f"unknown task {task}")

    return {
        "id": sample_id,
        "image": frame_path,
        "conversations": conv,
        "_meta": {"split": rec["split"], "task": task,
                  "layout": teacher_layout, "is_sludge": is_sludge},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_labels", default=str(REPO_ROOT / "data" / "tiktok_fps1" / "teacher_labels.jsonl"))
    p.add_argument("--out_dir", default=str(REPO_ROOT / "data" / "tiktok_fps1"))
    p.add_argument("--task_weights", default="classify:25,layout:15,describe_teacher:20,coupled:35,refuse:5")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    weights = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in args.task_weights.split(",")}
    pool = []
    for k, w in weights.items(): pool.extend([k] * w)

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)

    per_split = {"train": [], "validate": []}
    task_counts = collections.Counter()
    for line in open(args.teacher_labels):
        r = json.loads(line)
        teacher = r.get("teacher") or {}
        if not teacher.get("description"): continue
        if r.get("split") not in per_split: continue
        task = rng.choice(pool); task_counts[task] += 1
        per_split[r["split"]].append(build_sample(r, task, rng))

    for sp, samples in per_split.items():
        fp = out_dir / f"tiktok_{sp}.json"
        with open(fp, "w") as f: json.dump(samples, f)
        print(f"[build-v6] {sp}: {len(samples)} -> {fp}", file=sys.stderr)
    print(f"[build-v6] tasks: {dict(task_counts)}", file=sys.stderr)


if __name__ == "__main__":
    main()

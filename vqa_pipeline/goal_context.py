"""
Shared goal-context loader.

Every generation/eval script needs "the experimenter's stated goal for the
experiment" as text context. Normally that comes from GOAL-tagged
label_annotations.json entries (the experimenter states it on camera and an
annotator tags those lines GOAL). Some datasets (e.g. session04) have no such
lines at all, and instead ship a goal/ directory of reference material
supplied out of band — a lab protocol CSV, a separate interview transcript,
etc. — describing what the experiment is for.

load_goal_context(dataset_dir, label_annotations) is the single entry point:
if <dataset_dir>/goal/ exists and contains anything, its files are read
(recursively) and concatenated into one text block — CSV/text files verbatim,
transcript .json files reformatted as "SPEAKER: text" lines — in path order,
for a stable, reproducible result. Otherwise falls back to GOAL-tagged
label_annotations.json entries, joined in chronological order.
"""

import json
from pathlib import Path


def _categories_of(a: dict) -> list[str]:
    c = a.get("category")
    return c if isinstance(c, list) else [c]


def _goal_tagged_text(label_annotations: dict) -> str:
    """GOAL-tagged transcript_text lines only, joined in chronological order —
    the experimenter's stated goal(s) for the experiment, spoken on camera."""
    goals = [a for a in label_annotations.get("annotations", []) if "GOAL" in _categories_of(a)]
    goals.sort(key=lambda a: a["start"])
    return "\n".join(a.get("transcript_text", "") for a in goals)


def _format_transcript_json(path: Path) -> str:
    segments = json.loads(path.read_text())
    lines = []
    for seg in segments:
        speaker = seg.get("speaker") or seg.get("speaker_label")
        text = seg.get("text", "")
        lines.append(f"{speaker}: {text}" if speaker else text)
    return "\n".join(lines)


def _goal_folder_text(goal_dir: Path) -> str:
    parts = []
    for path in sorted(p for p in goal_dir.rglob("*") if p.is_file()):
        if any(part.startswith(".") for part in path.relative_to(goal_dir).parts):
            continue
        rel = path.relative_to(goal_dir)
        if path.suffix.lower() == ".json":
            parts.append(f"--- {rel} ---\n{_format_transcript_json(path)}")
        elif path.suffix.lower() in (".csv", ".txt"):
            parts.append(f"--- {rel} ---\n{path.read_text()}")
    return "\n\n".join(parts)


def load_goal_folder_text(dataset_dir: Path) -> str:
    """Just the goal/ folder's contents, or "" if the dataset has none. Used where a
    GOAL-tagged label_annotations.json fallback already exists elsewhere and would
    double up with it if load_goal_context's fallback ran too (see evaluate_vqa.py)."""
    goal_dir = dataset_dir / "goal"
    return _goal_folder_text(goal_dir) if goal_dir.is_dir() else ""


def load_goal_context(dataset_dir: Path, label_annotations: dict) -> str:
    """The experimenter's stated goal for the experiment, as one text block.
    Prefers a dataset's goal/ folder (external reference material) when
    present and non-empty; otherwise falls back to GOAL-tagged
    label_annotations.json entries."""
    text = load_goal_folder_text(dataset_dir)
    return text if text else _goal_tagged_text(label_annotations)

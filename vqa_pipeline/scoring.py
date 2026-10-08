"""
Shared scoring helpers: turn a judge model's free-text response into a letter, and pool the
blind (--no-video --no-text) runs that define the "hard" item subset.

Used by evaluate_vqa.py (--hard-only), hard_items.py and score_vqa.py, so a response is graded
identically everywhere.
"""

import json
import re
from pathlib import Path

BLIND_EVAL_FILENAME = "eval_results_no_video_no_text.json"


def parse_predicted_letter(response: str, valid_letters: set) -> str | None:
    text = response.strip().replace("*", "").replace("_", "")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1]

    cleaned = last.strip(" .:\"'")
    if cleaned in valid_letters:
        return cleaned

    m = re.search(r"\b([A-Z])\b\s*[.\"']*$", last)
    if m and m.group(1) in valid_letters:
        return m.group(1)

    m = re.search(r"\b([A-Z])\)", last)
    if m and m.group(1) in valid_letters:
        return m.group(1)

    return None


def blind_run_paths(dataset: Path, model_suffix: str = "") -> list[Path]:
    """Every blind (--no-video --no-text) run recorded for a dataset at one model:
    the temp=0 baseline plus any temp!=0 repeat passes run alongside it. A
    run-to-run variance check (see hard_items.py) found ~12-14% of
    items flip right/wrong between repeats even at temp=0, so a single blind
    run understates the true hard set by ~15-20% — callers should pool every
    run found here rather than trusting BLIND_EVAL_FILENAME alone.

    model_suffix defaults to "" (the default judge model, BLIND_EVAL_FILENAME).
    Pass e.g. "_model_qwen3_vl_32b" (matching evaluate_vqa.py's --model output
    suffix) to get that model's OWN blind runs instead — different judge
    models don't necessarily find the same items guessable, so "hard for
    Gemini" and "hard for Qwen" can differ; use each model's own blind runs
    when scoring that model, not the default model's."""
    paths = []
    if model_suffix:
        base = dataset / f"eval_results_no_video_no_text{model_suffix}.json"
    else:
        base = dataset / BLIND_EVAL_FILENAME
    if base.exists():
        paths.append(base)
    paths.extend(sorted(dataset.glob(f"eval_results_no_video_no_text{model_suffix}_temp*_run*.json")))
    return paths


def blind_wrongness(dataset: Path, model_suffix: str = "") -> tuple[dict[str, dict], str | None]:
    """key -> {"stage": str, "wrong": bool|None} pooled across every blind run
    found by blind_run_paths (same model_suffix meaning — see there). "wrong"
    is True if the item was answered incorrectly in AT LEAST ONE run (the hard
    threshold — see hard_items.py), False if every run that parsed it got it
    right, None if it never parsed in any run (excluded — neither hard nor
    easy, matching the old single-run convention). Returns (by_key,
    model_name)."""
    paths = blind_run_paths(dataset, model_suffix)
    if not paths:
        return {}, None
    by_key: dict[str, dict] = {}
    model_name = None
    for path in paths:
        data = json.loads(path.read_text())
        if model_name is None:
            model_name = data.get("model", "unknown model")
        for r in data.get("results", []):
            entry = by_key.setdefault(r["key"], {"stage": r.get("stage"), "wrong": None})
            predicted = parse_predicted_letter(r.get("response", ""), set(r.get("choices", {}).keys()))
            if predicted is None:
                continue
            if predicted != r.get("correct_letter"):
                entry["wrong"] = True
            elif entry["wrong"] is None:
                entry["wrong"] = False
    return by_key, model_name



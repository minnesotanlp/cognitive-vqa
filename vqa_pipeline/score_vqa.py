"""
Score evaluate_vqa.py output: per-stage accuracy for one or more eval_results*.json files.

Each result row already records the shuffled choices and correct_letter; the judge's free-text
response is parsed to a letter with scoring.parse_predicted_letter (the same parser
--hard-only and hard_items.py use). A response that doesn't parse to a valid letter counts as
WRONG in the accuracy denominator, and is also reported separately as "unparsed".

--hard-only restricts scoring to the blind-filtered "hard" subset: items that at least one
blind run (--no-video --no-text) of the SAME judge model got wrong. Blind runs are matched to
the results file by its _model_<slug> filename suffix (none = the default judge model), since
different judge models don't find the same items guessable.

Usage:
    python3 vqa_pipeline/score_vqa.py data/session01/eval_results_only_video.json
    python3 vqa_pipeline/score_vqa.py data/session0*/eval_results_no_video_no_text.json
    python3 vqa_pipeline/score_vqa.py data/session01/eval_results_only_video.json --hard-only
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import blind_wrongness, parse_predicted_letter  # noqa: E402

STAGES = ("stage1", "stage2", "stage3", "interview")
MODEL_SUFFIX_RE = re.compile(r"(_model_[a-z0-9_]+?)(?:_temp[0-9_]+)?\.json$")


def model_suffix_of(results_path: Path) -> str:
    m = MODEL_SUFFIX_RE.search(results_path.name)
    return m.group(1) if m else ""


def score_file(path: Path, hard_only: bool) -> dict[str, dict]:
    data = json.loads(path.read_text())
    hard_keys = None
    if hard_only:
        by_key, _ = blind_wrongness(path.parent, model_suffix_of(path))
        if not by_key:
            sys.exit(f"--hard-only: no blind runs found next to {path} "
                     f"(run evaluate_vqa.py --no-video --no-text first)")
        hard_keys = {k for k, e in by_key.items() if e["wrong"]}

    by_stage: dict[str, dict] = {}
    for r in data.get("results", []):
        if hard_keys is not None and r["key"] not in hard_keys:
            continue
        b = by_stage.setdefault(r["stage"], {"n": 0, "correct": 0, "unparsed": 0})
        b["n"] += 1
        predicted = parse_predicted_letter(r.get("response") or "", set(r.get("choices", {}).keys()))
        if predicted is None:
            b["unparsed"] += 1
        elif predicted == r.get("correct_letter"):
            b["correct"] += 1
    return by_stage


def fmt(b: dict) -> str:
    acc = b["correct"] / b["n"] if b["n"] else 0.0
    return f"n={b['n']:<4d} correct={b['correct']:<4d} acc={acc:6.1%}  unparsed={b['unparsed']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+", help="eval_results*.json file(s) to score")
    parser.add_argument("--hard-only", action="store_true",
                        help="Score only items wrong in >=1 blind run of the same judge model")
    args = parser.parse_args()

    grand = {s: {"n": 0, "correct": 0, "unparsed": 0} for s in STAGES}
    for path in args.results:
        by_stage = score_file(path, args.hard_only)
        print(f"=== {path.parent.name}/{path.name} ===")
        for s in STAGES:
            if s in by_stage:
                print(f"  {s:10s} {fmt(by_stage[s])}")
                for k in grand[s]:
                    grand[s][k] += by_stage[s][k]

    if len(args.results) > 1:
        print("=== Combined ===")
        total = {"n": 0, "correct": 0, "unparsed": 0}
        for s in STAGES:
            if grand[s]["n"]:
                print(f"  {s:10s} {fmt(grand[s])}")
                for k in total:
                    total[k] += grand[s][k]
        print(f"  {'TOTAL':10s} {fmt(total)}")


if __name__ == "__main__":
    main()

"""
Report the blind-filtered "hard" item counts for one or more VQA datasets.

A "hard" item is one that AT LEAST ONE of the dataset's blind runs (--no-video
--no-text: the temp=0 baseline plus any temp!=0 repeat passes sitting
alongside it, pooled by scoring.blind_run_paths) answered INCORRECTLY.
Items every blind run gets right are "guessable" from the question's phrasing
alone and don't test whether the video/context actually helps, so they're
excluded from the hard set. This is computed per dataset AND per stage (never
pooled first), then summed for combined totals.

Why "wrong in >=1 run" rather than the single baseline run: a 3-run repeat
check across all 5 datasets (temp=0 baseline + two temp=1.0 passes) found only
~33% of items were wrong in EVERY run, while another ~15% flipped right/wrong
depending on which single run you happened to use. A single-run filter
therefore both misses genuinely-hard items a given run got lucky on, and
undercounts by treating those flip-prone items as fully easy. Requiring wrong
in just one run is deliberately permissive (see the run count printed per
dataset below) rather than requiring agreement across all of them.

Covers stage1/stage2/stage3 AND interview — interview items use the same
lettered multiple-choice shape (choices + correct_letter) as stage1, so the
same blind/wrong scoring applies unchanged; grep this file for STAGES if a
future stage shape needs different handling.

Reuses blind_wrongness/parse_predicted_letter from scoring.py so a response is
graded identically here, in score_vqa.py, and in evaluate_vqa.py's --hard-only.

Usage:
    python3 vqa_pipeline/hard_items.py
    python3 vqa_pipeline/hard_items.py --dataset data/session04
    python3 vqa_pipeline/hard_items.py --stages stage1 stage2
"""

import argparse
import sys
from pathlib import Path

import os

VQA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VQA_DIR))
from scoring import blind_run_paths, blind_wrongness  # noqa: E402

DATA_DIR = Path(os.environ.get("COGVQA_DATA_DIR", VQA_DIR.parent / "data"))

STAGES = ("stage1", "stage2", "stage3", "interview")

DEFAULT_DATASETS = [
    DATA_DIR / "session01",
    DATA_DIR / "session02",
    DATA_DIR / "session03",
    DATA_DIR / "session04",
    DATA_DIR / "session05",
]


def score_dataset(dataset: Path, stages: tuple[str, ...]) -> tuple[dict, str | None, int]:
    """Returns ({stage: {n, blind_right, hard, unparsed}}, model_name, n_runs_pooled)."""
    n_runs = len(blind_run_paths(dataset))
    if n_runs == 0:
        return {}, None, 0
    by_key, model_name = blind_wrongness(dataset)
    by_stage = {s: {"n": 0, "blind_right": 0, "hard": 0, "unparsed": 0} for s in stages}
    for entry in by_key.values():
        stage = entry["stage"]
        if stage not in stages:
            continue
        b = by_stage[stage]
        b["n"] += 1
        if entry["wrong"] is None:
            b["unparsed"] += 1
        elif entry["wrong"]:
            b["hard"] += 1
        else:
            b["blind_right"] += 1
    return by_stage, model_name, n_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, nargs="+", default=DEFAULT_DATASETS,
                         help="Dataset directories to score (default: all 5 released sessions).")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES),
                         help="Which stages to include (default: all of stage1/stage2/stage3/interview).")
    args = parser.parse_args()
    stages = tuple(args.stages)

    grand = {s: {"n": 0, "blind_right": 0, "hard": 0, "unparsed": 0} for s in stages}
    models_seen = set()

    for dataset in args.dataset:
        by_stage, model_name, n_runs = score_dataset(dataset, stages)
        if not by_stage:
            print(f"=== {dataset.name} === (skipped: no blind runs found)")
            continue
        models_seen.add(model_name)
        run_word = "run" if n_runs == 1 else "runs"
        print(f"=== {dataset.name} === (blind model: {model_name}, {n_runs} {run_word} pooled)")
        for s in stages:
            b = by_stage[s]
            if b["n"] == 0:
                continue
            print(f"  {s:10s} n={b['n']:<4d} blind_right={b['blind_right']:<4d} "
                  f"hard={b['hard']:<4d} unparsed={b['unparsed']}")
            for k in grand[s]:
                grand[s][k] += b[k]

    print()
    if len(models_seen) > 1:
        print(f"WARNING: blind runs used different models across datasets: {sorted(models_seen)} "
              f"— combined totals below mix models, filter each model separately instead.")
    print("=== Combined ===")
    total_n = total_hard = 0
    for s in stages:
        g = grand[s]
        if g["n"] == 0:
            continue
        print(f"  {s:10s} n={g['n']:<4d} blind_right={g['blind_right']:<4d} "
              f"hard={g['hard']:<4d} unparsed={g['unparsed']}")
        total_n += g["n"]
        total_hard += g["hard"]
    print(f"  {'TOTAL':10s} n={total_n:<4d} hard={total_hard}")

    if "stage3" in stages:
        print("\nNote: the stage3 hard set is not a reliable difficulty control — blind's stage3 "
              "correctness looks driven by answer-letter position bias rather than guessability.")


if __name__ == "__main__":
    main()

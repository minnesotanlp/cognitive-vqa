"""
Structural self-checks for a ground_truth_actions.json (see ground_truth/README.md):

  - every entry has exactly the key set for its type
  - start < end for every entry
  - entries are sorted by start
  - scientist_action entries don't overlap each other; gaps between them are listed (gaps are
    allowed where nothing loggable happens, but should be deliberate)
  - every state entry overlaps at least one scientist_action entry

Exits non-zero if any hard error (schema, ordering, overlap, orphan state) is found.

Usage:
    python3 ground_truth/validate_ground_truth.py data/session01/annotations/ground_truth_actions.json
"""

import argparse
import json
import sys
from pathlib import Path

KEYS = {
    "scientist_action": {"type", "start", "end", "action"},
    "state": {"type", "start", "end", "subject", "state"},
}
GAP_TOLERANCE_SECONDS = 0.5


def validate(entries: list[dict]) -> tuple[list[str], list[str]]:
    errors, notes = [], []
    for i, e in enumerate(entries):
        expected = KEYS.get(e.get("type"))
        if expected is None:
            errors.append(f"[{i}] unknown type {e.get('type')!r}")
            continue
        if set(e) != expected:
            errors.append(f"[{i}] keys {sorted(e)} != {sorted(expected)}")
        if not e["start"] < e["end"]:
            errors.append(f"[{i}] start {e['start']} >= end {e['end']}")
        if i and e["start"] < entries[i - 1]["start"]:
            errors.append(f"[{i}] not sorted: start {e['start']} < previous {entries[i - 1]['start']}")

    actions = sorted((e for e in entries if e.get("type") == "scientist_action"), key=lambda e: e["start"])
    for prev, cur in zip(actions, actions[1:]):
        if cur["start"] < prev["end"] - GAP_TOLERANCE_SECONDS:
            errors.append(f"actions overlap: {prev['start']}-{prev['end']} and {cur['start']}-{cur['end']}")
        elif cur["start"] > prev["end"] + GAP_TOLERANCE_SECONDS:
            notes.append(f"gap between actions: {prev['end']}s -> {cur['start']}s")

    for e in entries:
        if e.get("type") == "state" and not any(a["start"] < e["end"] and e["start"] < a["end"] for a in actions):
            errors.append(f"state {e['start']}-{e['end']} ({e.get('subject')!r}) overlaps no action")
    return errors, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()

    failed = False
    for path in args.paths:
        entries = json.loads(path.read_text())
        errors, notes = validate(entries)
        n_actions = sum(e.get("type") == "scientist_action" for e in entries)
        span = f"{min(e['start'] for e in entries):g}-{max(e['end'] for e in entries):g}s" if entries else "empty"
        print(f"=== {path} === {len(entries)} entries ({n_actions} actions / {len(entries) - n_actions} states), {span}")
        for n in notes:
            print(f"  note: {n}")
        for err in errors:
            print(f"  ERROR: {err}")
        failed |= bool(errors)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

"""
Build vqa.json from <dataset>/gemini_queries.json: one multiple-choice item
per Gemini-generated question (stage1 always, stage2 for PLAN annotations,
stage3 for ASSESSMENT annotations). If <dataset>/interview_queries.json exists
(vqa_pipeline/query_gemini_annotations.py --interview's output), also adds one
"interview" stage item per entry there — same prose-answer shape as stage1,
distractors chosen the same BLIND+GROUNDED way, annotation_id carried through
unchanged ("entry_id:highlight_index", already unique, no merge/dedup needed
since interview highlights aren't split from a pre-merge annotation the way
label_annotations.json's are).

Stage1 and stage3 items get TWO incorrect answers, chosen from two sources:
  - the BLIND set: <dataset>/adversarial_items.json (vqa_pipeline/generate_adversarial.py's
    output, keyed "annotation_id:stage" -> [answer1, answer2]) — the model answers without
    seeing the video.
  - the GROUNDED set: <dataset>/adversarial_items_grounded.json
    (vqa_pipeline/generate_adversarial_grounded.py's output, one answer per item) — the model
    answers WITH the video, told the provided answer is a defective placeholder it must
    replace with the real, specific (but still wrong) account.
One blind answer is picked per item: if <dataset>/eval_results_only_video.json (the
video-only baseline eval) shows the judge model previously chose one of that item's two
blind answers over the correct one, that specific answer is kept (it's a *proven* hard
distractor); otherwise the first blind answer is used by default. The grounded answer is
always added as the second distractor. If either source is missing for an item, that item
is skipped (run both adversarial scripts against this vqa.json first, then re-run this
script to merge them in). This BLIND+GROUNDED scheme is now used for all three stages,
including stage2 (previously stage2 picked its own distractors here directly from
ground_truth_actions.json by vocabulary overlap; that path is gone).

For stage2/stage3 (bare-timespan answers), both the BLIND and GROUNDED sources are expected
to already keep every candidate at least MIN_GAP_SECONDS away from the correct answer's own
timespan (generate_adversarial.py and generate_adversarial_grounded.py enforce this by
filtering their candidate pool before ever showing it to the model) — choose_distractors
re-checks this gap defensively here and drops anything that slips through, rather than
trusting the upstream generation blindly.

Usage:
    python3 vqa_pipeline/build_vqa.py --dataset session01
"""

import argparse
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_VIDEOS_DIR = Path(os.environ.get("COGVQA_DATA_DIR", REPO_ROOT / "data"))

# Shared floor for stage2 AND stage3: a distractor's timespan must be at least this many
# seconds from the correct answer's own timespan to count as "clearly a different point in
# the experiment" — previously stage2 enforced 300s here while stage3 enforced nothing at all
# (0s), a 10x asymmetry that made stage3 dramatically harder for reasons unrelated to what the
# question was actually testing. Must match the MIN_GAP_SECONDS in generate_adversarial.py and
# generate_adversarial_grounded.py, which are what actually keep candidates this far away.
MIN_GAP_SECONDS = 100

STOPWORDS = {
    "a", "an", "and", "the", "to", "of", "in", "on", "with", "into", "is",
    "at", "for", "from", "it", "its", "this", "that", "then", "so", "as",
    "up", "out", "over", "again", "back", "scientist", "experimenter",
    "they", "their", "some", "any", "we", "our", "will", "be", "being",
    "which", "point", "before", "after", "while",
}


def parse_json_response(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def ground_truth_action_text(entry: dict) -> str:
    if entry["type"] == "scientist_action":
        return entry["action"]
    return f"{entry['subject']}: {entry['state']}"


def format_ground_truth_entry(entry: dict) -> str:
    return f"({entry['start']}s-{entry['end']}s) {ground_truth_action_text(entry)}"


def to_timespan(start: float, end: float) -> str:
    """Canonical answer form for the temporal (stage2/stage3) questions: a bare, integer
    time range like '140s-212s'. Integers (not the raw decimals) so the correct answer can't
    be singled out by having a different precision than the distractors."""
    return f"{int(round(start))}s-{int(round(end))}s"


def significant_words(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def parse_time_bracket(text: str) -> tuple[float, float] | None:
    m = re.search(r"\[?(\d+(?:\.\d+)?)s\s*-\s*(\d+(?:\.\d+)?)s\]?", text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def find_matching_ground_truth_entry(ground_truth_actions: list[dict], start: float, end: float,
                                       question_text: str) -> dict | None:
    """Among entries overlapping [start, end), prefer the one whose text best matches the
    question (multiple ground-truth actions can share a time window — picking by raw time
    overlap alone can favor an unrelated but longer-duration entry). Falls back to time-overlap
    duration as a tiebreaker, then to nearest by start-time distance if nothing overlaps."""
    overlapping = [
        e for e in ground_truth_actions
        if e["start"] < end and e["end"] > start
    ]
    if overlapping:
        q_words = significant_words(question_text)
        def score(e):
            word_overlap = len(q_words & significant_words(ground_truth_action_text(e)))
            time_overlap = min(e["end"], end) - max(e["start"], start)
            return (word_overlap, time_overlap)
        return max(overlapping, key=score)
    if not ground_truth_actions:
        return None
    return min(ground_truth_actions, key=lambda e: abs(e["start"] - start))


def timespan_gap_seconds(span_a: str, span_b: str) -> float | None:
    """Gap in seconds between two bare timespans like "140s-212s" (0 if they overlap or touch);
    None if either string doesn't parse as a timespan (e.g. a stage1 prose answer) — the caller
    treats None as "not applicable," not as violating the gap."""
    a = parse_time_bracket(span_a)
    b = parse_time_bracket(span_b)
    if a is None or b is None:
        return None
    return max(0.0, max(a[0] - b[1], b[0] - a[1]))


def parse_chosen_letter(response: str, choices: dict[str, str]) -> str | None:
    """The video-only eval's judge was asked to respond with just a letter, but sometimes
    reasons at length first — pull the LAST standalone A-Z token that's actually one of this
    item's valid choice letters, since that's the model's final answer."""
    valid = set(choices)
    matches = re.findall(r"\b([A-Z])\b", response)
    for letter in reversed(matches):
        if letter in valid:
            return letter
    return None


def match_span_width(correct_span: str, candidate: str) -> str:
    """Re-cut `candidate` to the same PRINTED duration as `correct_span`, keeping its midpoint
    and preserving at least MIN_GAP_SECONDS of separation.

    Correct answers are the experimenter's own utterance span (a sentence or two of speech:
    1-20s here, median 7s), while distractors are ground_truth_actions.json segments, which are
    whole activities and run far longer — 159s, 194s. That difference is a free signal: in the
    session01 stage3 set the correct answer was NEVER the widest of the three options (0 of 20),
    so "don't pick the long one" eliminates a distractor with no knowledge of the experiment at
    all. Matching the width makes duration carry no information either way.

    Two details that are easy to get wrong. Width is taken from the correct span AS PRINTED
    (integer seconds), not from the underlying floats, or rounding leaves the rendered options a
    second apart and the tell survives. And re-cutting moves a span's endpoints, which can pull a
    distractor inside MIN_GAP_SECONDS of the answer even though the candidate filter passed it
    earlier — so afterwards the span is pushed back out, away from the answer, keeping its width.

    Only the printed span moves; which moment the distractor points at is unchanged. Returns the
    candidate untouched if either span can't be parsed."""
    a = parse_time_bracket(correct_span)
    b = parse_time_bracket(candidate)
    if a is None or b is None:
        return candidate
    a_start, a_end = round(a[0]), round(a[1])
    width = a_end - a_start
    start = max(0.0, (b[0] + b[1]) / 2 - width / 2)
    start = round(start)

    # Restore the gap if re-cutting ate into it, moving away on whichever side it already sits.
    if start + width > a_start - MIN_GAP_SECONDS and start < a_end + MIN_GAP_SECONDS:
        if (b[0] + b[1]) / 2 < (a[0] + a[1]) / 2:
            start = a_start - MIN_GAP_SECONDS - width   # it was before the answer; push earlier
        else:
            start = a_end + MIN_GAP_SECONDS             # it was after; push later
        if start < 0:
            return candidate                            # no room before the answer — leave as-is
    return to_timespan(start, start + width)


def choose_distractors(key: str, blind_answers: list[str], grounded_answers: list[str],
                        only_video_by_key: dict[str, dict], correct_span: str | None = None,
                        match_width: bool = False) -> list[str]:
    """Two incorrect answers: one BLIND answer (whichever one previously fooled the
    video-only judge into picking it over the correct answer, or the first blind answer if
    the judge got this item right or it wasn't evaluated) plus the GROUNDED answer.

    correct_span, given only for stage2/stage3's bare-timespan answers, triggers a defensive
    MIN_GAP_SECONDS check: generation already filters both sources' candidates to this gap, so
    this should normally be a no-op, but it drops anything that slips through (e.g. stale data
    generated before that filter existed, or a malformed non-timespan answer) instead of
    trusting it blindly."""
    def far_enough(candidate: str) -> bool:
        gap = timespan_gap_seconds(correct_span, candidate)
        return gap is not None and gap >= MIN_GAP_SECONDS

    if correct_span is not None:
        blind_answers = [a for a in blind_answers if far_enough(a)]
        grounded_answers = [a for a in grounded_answers if far_enough(a)]

    chosen_blind = None
    ov = only_video_by_key.get(key)
    if ov is not None:
        letter = parse_chosen_letter(ov["response"], ov["choices"])
        if letter is not None and letter != ov["correct_letter"]:
            chosen_text = ov["choices"][letter]
            if chosen_text in blind_answers:
                chosen_blind = chosen_text
    if chosen_blind is None and blind_answers:
        chosen_blind = blind_answers[0]

    grounded = grounded_answers[0] if grounded_answers else None
    distractors = [a for a in (chosen_blind, grounded) if a is not None]

    # Guard against the two distractors coinciding (grounded answer happens to match the
    # chosen blind one) — fall back to the other blind answer so the item still gets two
    # genuinely different incorrect options.
    if len(distractors) == 2 and distractors[0] == distractors[1]:
        alt = next((a for a in blind_answers if a != distractors[0]), None)
        if alt is not None:
            distractors[0] = alt

    # Width-matching is opt-in per stage, not implied by correct_span. The duration leak exists
    # for every bare-timespan stage, but re-cutting changes the printed options, so turning it on
    # for a stage invalidates comparisons against that stage's earlier eval runs. stage3 opts in;
    # stage2 deliberately does not. Runs last, so it can't affect which candidates were selected
    # above — only how the chosen ones are printed.
    if match_width and correct_span is not None:
        distractors = [match_span_width(correct_span, d) for d in distractors]
        # Re-cutting can collapse two nearby distractors onto the same printed span.
        if len(distractors) == 2 and distractors[0] == distractors[1]:
            return distractors[:1]
    return distractors


def build_item(question: str, correct_answer: str, incorrect_answers: list[str],
                annotation_id: str, category: str, stage: str, rationale: str | None = None,
                current_point: str | None = None) -> dict:
    item = {"question": question, "correct_answer": correct_answer}
    for i, ans in enumerate(incorrect_answers, 1):
        item[f"incorrect_answer{i}"] = ans
    if rationale is not None:
        item["rationale"] = rationale
    item["annotation_id"] = annotation_id
    item["category"] = category
    item["stage"] = stage
    if current_point is not None:
        item["current_point"] = current_point
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset folder name, e.g. session01")
    args = parser.parse_args()

    dataset_dir = ANNOTATION_VIDEOS_DIR / args.dataset
    queries_file = dataset_dir / "gemini_queries.json"
    interview_queries_file = dataset_dir / "interview_queries.json"
    ground_truth_file = dataset_dir / "annotations" / "ground_truth_actions.json"
    label_annotations_file = dataset_dir / "annotations" / "label_annotations.json"
    adversarial_file = dataset_dir / "adversarial_items.json"
    grounded_adversarial_file = dataset_dir / "adversarial_items_grounded.json"
    only_video_eval_file = dataset_dir / "eval_results_only_video.json"
    output_file = dataset_dir / "vqa.json"

    data = json.loads(queries_file.read_text())
    ground_truth_actions = json.loads(ground_truth_file.read_text()) if ground_truth_file.exists() else []

    # stage1/stage3 BLIND incorrect answers, keyed "annotation_id:stage" -> [answer1, answer2].
    # Produced by vqa_pipeline/generate_adversarial.py; absent until that's been run against this
    # vqa.json at least once.
    adversarial_answers = {}
    if adversarial_file.exists():
        for r in json.loads(adversarial_file.read_text())["results"]:
            if r.get("adversarial_answers"):
                adversarial_answers[r["key"]] = r["adversarial_answers"]

    # stage1/stage3 GROUNDED incorrect answer (one per item), keyed the same way. Produced by
    # vqa_pipeline/generate_adversarial_grounded.py.
    grounded_answers = {}
    if grounded_adversarial_file.exists():
        for r in json.loads(grounded_adversarial_file.read_text())["results"]:
            if r.get("adversarial_answers"):
                grounded_answers[r["key"]] = r["adversarial_answers"]

    # Per-stage coverage, not a blanket file-exists check: a stage added to an already-mature
    # pipeline (its distractor files already populated by earlier stages) needs its OWN first,
    # distractor-less pass to seed generate_adversarial.py/generate_adversarial_grounded.py from
    # — an exists() check would instead treat every one of its items as a real "missing
    # distractors" problem from the very first run. Same fix "interview" already needed below.
    distractors_seeded = {
        stage: any(k.endswith(f":{stage}") for k in adversarial_answers)
        or any(k.endswith(f":{stage}") for k in grounded_answers)
        for stage in ("stage1", "stage2", "stage3")
    }

    # video-only baseline eval results, keyed the same way — used to detect which (if any) of
    # an item's blind answers already fooled a video-only judge into picking it over the
    # correct answer.
    only_video_by_key = {}
    if only_video_eval_file.exists():
        only_video_by_key = {r["key"]: r for r in json.loads(only_video_eval_file.read_text())["results"]}

    # label_annotations.json has been de-duplicated by merging annotations that share the same
    # transcript_line_ids (one long spoken line tagged at several points). gemini_queries.json,
    # however, still holds one result per pre-merge annotation, so multiple results can map to the
    # same merged annotation. Map each result to its surviving (merged) annotation via
    # transcript_line_ids, emit items under that survivor id + its merged category, and keep only
    # the FIRST result per (survivor, stage) — dropping the near-duplicate questions the pre-merge
    # sub-spans produced. Results whose line no longer maps to any annotation are dropped entirely.
    #
    # transcript_line_ids is normally a list of integer line indices, but a stray record can carry
    # a UUID string instead (same quirk query_gemini_annotations.py's own line_ids filtering
    # defends against) — sometimes the WHOLE list is non-integer, sometimes just one entry is mixed
    # in among real ints, and that mixed case makes plain sorted() crash comparing int to str. Drop
    # non-integer entries before keying so both sides of this lookup (label_annotations and
    # gemini_queries results) use the same, sortable key. But an entry with NO integer ids at all
    # can't fall back to a bare empty tuple either — every such entry would collapse onto the same
    # key and silently overwrite each other's survivor mapping in line_to_survivor. These entries
    # were never merge-dedup candidates anyway (dedup is keyed on REAL shared line ids), so key
    # them by their own id instead, which a pre-merge result's own annotation_id always matches
    # 1:1 when it was never a merge target.
    def _line_key(entry_id, ids):
        int_ids = tuple(sorted(i for i in (ids or []) if isinstance(i, int)))
        return int_ids if int_ids else ("_no_int_ids", entry_id)

    label_annotations = json.loads(label_annotations_file.read_text())["annotations"]
    line_to_survivor = {_line_key(a["id"], a["transcript_line_ids"]): a["id"] for a in label_annotations}
    id_to_category = {a["id"]: a["category"] for a in label_annotations}

    # Process results whose own id is the survivor first, so the merged annotation's own generated
    # question (when it exists) wins over a merged-away sibling's.
    results = sorted(
        data["results"],
        key=lambda r: 0 if line_to_survivor.get(_line_key(r["annotation_id"], r["transcript_line_ids"])) == r["annotation_id"] else 1,
    )

    items = []
    skipped = []
    n_stale = 0
    seen = {"stage1": set(), "stage2": set(), "stage3": set(), "interview": set()}  # (survivor_id) per stage, for dedup

    for r in results:
        survivor = line_to_survivor.get(_line_key(r["annotation_id"], r["transcript_line_ids"]))
        if survivor is None:
            n_stale += 1
            continue
        annotation_id = survivor
        category = id_to_category[survivor]

        # ── Stage 1: question_answer_pairs. Incorrect answers come later from
        # vqa_pipeline/generate_adversarial.py. ──
        if r.get("stage1") and survivor not in seen["stage1"]:
            try:
                stage1_json = parse_json_response(r["stage1"]["response1"])
                qa = stage1_json.get("question_answer_pairs", stage1_json.get("question_answer_pair"))
                if qa is None:
                    raise KeyError("question_answer_pairs")
                key = f"{annotation_id}:stage1"
                incorrect = choose_distractors(
                    key, adversarial_answers.get(key, []), grounded_answers.get(key, []), only_video_by_key
                )
                if len(incorrect) < 2 and distractors_seeded["stage1"]:
                    raise ValueError("fewer than 2 adversarial distractors found for this item "
                                      f"(blind={key in adversarial_answers}, grounded={key in grounded_answers})")
                items.append(build_item(
                    qa["question"], qa["answer"], incorrect,
                    annotation_id, category, "stage1", rationale=qa.get("rationale"),
                ))
                seen["stage1"].add(survivor)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                skipped.append((annotation_id, "stage1", str(e)))

        # ── Stage 2 (PLAN): dependency_question. Correct answer is the ground_truth_actions
        # entry matching Gemini's proposed time bracket; incorrect answers come from
        # vqa_pipeline/generate_adversarial.py + generate_adversarial_grounded.py, same BLIND+
        # GROUNDED scheme as stage1/stage3. The question refers to "the current point" in the
        # experiment (the future event the experimenter is preparing for) without stating it —
        # dq["future_event"] is Gemini's own identification of that moment, carried through as
        # current_point so evaluate_vqa.py can tell the judge model what "currently" means
        # (mirrors interview stage's reference_start/end disclosure; without it the full,
        # untruncated stage2 history gives the judge no way to know which moment is "now"). ──
        if r.get("stage2") and survivor not in seen["stage2"]:
            try:
                dq = parse_json_response(r["stage2"]["response"])["dependency_question"]
                bracket = parse_time_bracket(dq["answer"])
                if bracket is None or not ground_truth_actions:
                    raise ValueError(f"couldn't resolve answer time bracket: {dq['answer']!r}")
                correct_entry = find_matching_ground_truth_entry(ground_truth_actions, *bracket, dq["question"])
                correct_span = to_timespan(correct_entry["start"], correct_entry["end"])
                current_point_bracket = parse_time_bracket(dq.get("future_event", ""))
                if current_point_bracket is None:
                    raise ValueError(f"couldn't resolve future_event time bracket: {dq.get('future_event')!r}")
                current_point = to_timespan(*current_point_bracket)
                key = f"{annotation_id}:stage2"
                incorrect = choose_distractors(
                    key, adversarial_answers.get(key, []), grounded_answers.get(key, []), only_video_by_key,
                    correct_span=correct_span,
                )
                if len(incorrect) < 2 and distractors_seeded["stage2"]:
                    raise ValueError("fewer than 2 gap-compliant adversarial distractors found for this item "
                                      f"(blind={key in adversarial_answers}, grounded={key in grounded_answers})")
                items.append(build_item(
                    dq["question"], correct_span, incorrect,
                    annotation_id, category, "stage2", current_point=current_point,
                ))
                seen["stage2"].add(survivor)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                skipped.append((annotation_id, "stage2", str(e)))

        # ── Stage 3 (ASSESSMENT): assessment_question. Correct answer is the source
        # annotation's own [reference_start, reference_end] span (the moment the assessment
        # was made) rather than Gemini's quote+"[answer]" string, so it's the same
        # bare-timespan form the eval expects and carries no format tell. Incorrect answers
        # come later from vqa_pipeline/generate_adversarial.py. ──
        if r.get("stage3") and survivor not in seen["stage3"]:
            try:
                aq = parse_json_response(r["stage3"]["response1"])["assessment_question"]
                # The stage3 prompt asks Gemini to decline rather than invent a question when
                # the moment has no discrete visual trigger, or when the same observation holds
                # across a long stretch of footage so no single timestamp is defensibly the
                # answer. Both forms of decline land here: an explicit "unsuitable" verdict, or
                # a suitable-shaped response that failed to justify temporal uniqueness. These
                # are expected and common, not errors — most moments in an experiment genuinely
                # don't make well-posed "when did they think this?" items.
                if aq.get("unsuitable"):
                    raise ValueError(f"stage3 judged unsuitable: {aq['unsuitable']}")
                for field in ("visual_evidence", "why_not_elsewhere"):
                    if not (aq.get(field) or "").strip():
                        raise ValueError(f"stage3 response missing {field}")
                correct_span = to_timespan(r["reference_start"], r["reference_end"])
                key = f"{annotation_id}:stage3"
                incorrect = choose_distractors(
                    key, adversarial_answers.get(key, []), grounded_answers.get(key, []), only_video_by_key,
                    correct_span=correct_span, match_width=True,
                )
                if len(incorrect) < 2 and distractors_seeded["stage3"]:
                    raise ValueError("fewer than 2 gap-compliant adversarial distractors found for this item "
                                      f"(blind={key in adversarial_answers}, grounded={key in grounded_answers})")
                items.append(build_item(
                    aq["question"], correct_span, incorrect,
                    annotation_id, category, "stage3", rationale=aq.get("rationale"),
                ))
                seen["stage3"].add(survivor)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                skipped.append((annotation_id, "stage3", str(e)))

    # ── Interview: one "why"/intent question per query_gemini_annotations.py --interview
    # result (already unique per entry_id:highlight_index — no merge/dedup step needed here,
    # unlike stage1/2/3's label_annotations.json pre-merge duplicates). Distractors come from
    # the same BLIND+GROUNDED scheme, same as stage1. ──
    if interview_queries_file.exists():
        interview_data = json.loads(interview_queries_file.read_text())
        # Unlike stage1/2/3 (which have coexisted in adversarial_items.json /
        # adversarial_items_grounded.json since those files' first ever write), "interview" is a
        # stage added to an already-mature pipeline: both files already exist (populated by
        # stage1/2/3), so the plain .exists() check stage1/2/3 use would raise on EVERY interview
        # item before its distractors have ever been generated — there'd be no first,
        # distractor-less pass to seed generate_adversarial.py/generate_adversarial_grounded.py
        # from. Check coverage of this stage specifically instead.
        interview_distractors_seeded = (
            any(k.endswith(":interview") for k in adversarial_answers)
            or any(k.endswith(":interview") for k in grounded_answers)
        )
        for r in interview_data.get("results", []):
            annotation_id = r["annotation_id"]
            if annotation_id in seen["interview"]:
                continue
            try:
                parsed = parse_json_response(r["response"])
                qa = parsed.get("question_answer_pairs", parsed.get("question_answer_pair"))
                if qa is None:
                    raise KeyError("question_answer_pairs")
                key = f"{annotation_id}:interview"
                incorrect = choose_distractors(
                    key, adversarial_answers.get(key, []), grounded_answers.get(key, []), only_video_by_key
                )
                if len(incorrect) < 2 and interview_distractors_seeded:
                    raise ValueError("fewer than 2 adversarial distractors found for this item "
                                      f"(blind={key in adversarial_answers}, grounded={key in grounded_answers})")
                items.append(build_item(
                    qa["question"], qa["answer"], incorrect,
                    annotation_id, "INTERVIEW_INTENT", "interview", rationale=qa.get("rationale"),
                ))
                seen["interview"].add(annotation_id)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                skipped.append((annotation_id, "interview", str(e)))

    output_file.write_text(json.dumps(items, indent=2))
    print(f"Wrote {len(items)} items to {output_file}")
    if n_stale:
        print(f"Skipped {n_stale} result(s) whose transcript line no longer maps to any annotation.")
    print(f"(deduped to one item per merged-annotation per stage: "
          f"{len(seen['stage1'])} stage1, {len(seen['stage2'])} stage2, {len(seen['stage3'])} stage3, "
          f"{len(seen['interview'])} interview)")
    if skipped:
        print(f"Skipped {len(skipped)} (parse failures, or stage3 moments judged unsuitable):")
        for annotation_id, stage, err in skipped:
            print(f"  {annotation_id} ({stage}): {err}")


if __name__ == "__main__":
    main()

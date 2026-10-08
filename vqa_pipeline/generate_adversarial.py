"""
Generate adversarial answer pairs for <dataset>/vqa.json.

The model answers "blind" — it never sees the video or most of the experimenter's comments — so
these are adversarial: answers that sound reasonable without having watched the experiment. It is
always told the question's correct answer too, so it can generate genuinely wrong distractors
instead of accidentally rephrasing or duplicating the right one. What else it sees depends on
the question's answer type:
  - stage1 and interview (prose "why" answers — identical treatment, since an interview item is
    just a "why" question sourced from retrospective commentary instead of live narration): the
    question text + correct answer + the experiment's stated GOAL (label_annotations.json entries
    tagged GOAL, not time-filtered since the goal is stated up front and holds throughout) -> two
    different plausible but incorrect answers, matched to the correct answer's word count and
    phrase-vs-sentence format (a short-phrase correct answer shouldn't be paired with full-sentence
    distractors, since that alone gives it away). The goal gives just enough grounding to keep
    distractors on-topic without revealing the experimenter's other comments.
  - stage2/stage3 (timestamp-range answers): the question + correct answer PLUS the action sheet
    (ground_truth_actions.json, timespans + descriptions) -> it picks two plausible-but-incorrect
    timespans from that sheet. It needs the descriptions because, without the video, bare
    timestamps carry no information to reason about; the descriptions let it pick semantically
    tempting (but wrong) times. Output is the bare timespans, matching the eval answer format.
    The action sheet shown for each item excludes any entry within MIN_GAP_SECONDS of that
    item's own correct answer, so whichever two entries the model picks are guaranteed to
    already satisfy build_vqa.py's gap requirement — no reliance on the model reading and
    obeying a distance instruction.

Reuses the eval run's plumbing (Vertex client + auth, proactive per-minute throttle, and the
429/5xx-aware retry) to stay comfortably under the project's per-minute quota — but deliberately
NOT its video/full-text context: the whole point is that the model sees only the question (plus
the GOAL context for stage1, and the action sheet for stage2/3).

Output is written incrementally to <dataset>/adversarial_items.json, keyed by
"annotation_id:stage", so re-running skips questions already done.

Usage:
    python3 vqa_pipeline/generate_adversarial.py --dataset session01
    python3 vqa_pipeline/generate_adversarial.py --dataset session01 --limit 5
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception

from goal_context import load_goal_context

REPO_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_VIDEOS_DIR = Path(os.environ.get("COGVQA_DATA_DIR", REPO_ROOT / "data"))
DOTENV_FILE = REPO_ROOT / ".env"

MODEL = "gemini-3.1-pro-preview"
VERTEX_LOCATION = "global"

# A little warmth so the two answers are genuinely different rather than paraphrases.
ADVERSARIAL_TEMPERATURE = 0.8
MIN_SECONDS_BETWEEN_CALLS = 3.0  # same proactive throttle as the eval — paced to stay under quota

# Stage2/stage3 action-sheet entries within this many seconds of the correct answer's own
# timespan are excluded before the model ever sees them — must match build_vqa.py's
# MIN_GAP_SECONDS, which is what actually enforces "clearly a different point in the experiment".
MIN_GAP_SECONDS = 100

# stage1: prose answers, question + correct answer + the experiment's stated GOAL (not the full
# transcript — just the GOAL-tagged line(s), giving a little grounding without spoiling anything).
STAGE1_PROMPT = """
    You will be given a single quiz question about a scientific chemistry experiment, along with
    a possible answer and the experimenter's stated goal for the experiment. You do NOT have
    access to the video or any other of the experimenter's comments — only the question, one
    possible answer, and the stated goal of the experiment.

    Provide TWO additional plausible answers to the question. They must be genuinely different from both one another and from the provided answer. No answer should be a rephrasing or partial restatements of another answer.

    CRITICAL FORMATTING RULE: your two answers MUST match the provided answer's format and length
    almost exactly. The provided answer is {correct_answer_word_count} words long — keep each of
    your answers within plus or minus 10 words of that. If the provided answer is a short phrase (e.g. "to
    clear out the funnel"), write short phrases, NOT full sentences. If it's one or more full
    sentences, match that instead. An answer that is noticeably longer, grammatically different, or more
    explanatory than the provided answer is WRONG regardless of its content, because it makes one answer stick out too much — brevity and register must match, not just
    plausibility.

    Return ONLY a JSON object in this exact format:
    {{
      "answer1": "<first plausible answer, matching the provided answer's format and length>",
      "answer2": "<second, genuinely different plausible answer, matching the provided answer's format and length>"
    }}

    Experiment goal: {goal_context}
    Question: {question}
    Possible answer: {correct_answer}
"""

# stage2/stage3: timestamp answers chosen from the action sheet, question + correct answer + sheet only (no video).
STAGE_TIMESTAMP_PROMPT = """
    You will be given a quiz question about a scientific chemistry experiment, its correct answer,
    and a list of timestamped actions/observations from that experiment (the "action sheet"). The
    answer to the question is one point in time. You do NOT have access to the video or the
    experimenter's comments — only the question, one possible answer, and the action sheet.

    From the action sheet, choose the TWO timespans that are the most plausible INCORRECT answers to
    the question. They must be two DIFFERENT entries, and neither may be the correct answer's
    timespan. Pick tempting, on-topic times — not obviously unrelated ones.

    Return ONLY a JSON object in this exact format, copying each timespan exactly as it appears in
    the sheet (e.g. "246s-279s"):
    {{
      "answer1": "<timespan of the first plausible incorrect entry>",
      "answer2": "<timespan of the second, different plausible incorrect entry>"
    }}

    Question: {question}
    Possible answer: {correct_answer}

    Action sheet:
    {action_sheet}
"""


def ground_truth_action_text(entry: dict) -> str:
    if entry["type"] == "scientist_action":
        return entry["action"]
    return f"{entry['subject']}: {entry['state']}"


def to_timespan(start: float, end: float) -> str:
    return f"{int(round(start))}s-{int(round(end))}s"


def parse_timespan(text: str) -> tuple[float, float] | None:
    m = re.search(r"(\d+(?:\.\d+)?)s?\s*-\s*(\d+(?:\.\d+)?)s", text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def gap_seconds(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, max(a[0] - b[1], b[0] - a[1]))


def _is_retryable(exc: BaseException) -> bool:
    """httpx.TransportError covers ReadError/ConnectError/*Timeout — the underlying httpx client
    raises these directly (e.g. "Connection reset by peer"), bypassing google.genai's
    ClientError/ServerError wrapping, so they need their own check rather than the status-code path."""
    if isinstance(exc, (ConnectionError, TimeoutError, httpx.TransportError)):
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code in (408, 429, 500, 502, 503, 504)


def _response_metadata(response) -> dict:
    """Everything worth keeping from a GenerateContentResponse besides the text itself: full
    token accounting (prompt/output/cached/thinking tokens) plus per-call diagnostics. Captured
    for every call so token spend is auditable after the fact instead of discarded."""
    usage = response.usage_metadata
    candidate = response.candidates[0] if response.candidates else None
    return {
        "model_version": response.model_version,
        "response_id": response.response_id,
        "create_time": response.create_time.isoformat() if response.create_time else None,
        "usage": usage.model_dump(mode="json", exclude_none=True) if usage else None,
        "finish_reason": candidate.finish_reason.value if candidate and candidate.finish_reason else None,
        "finish_message": candidate.finish_message if candidate else None,
    }


def parse_json_response(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def load_output(output_file: Path, dataset: str) -> dict:
    if output_file.exists():
        return json.loads(output_file.read_text())
    return {"dataset": dataset, "model": MODEL, "generated_at": None, "results": []}


def save_output(output_file: Path, output: dict) -> None:
    output["generated_at"] = datetime.now(timezone.utc).isoformat()
    output_file.write_text(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset folder name, e.g. session01")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N remaining questions")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3", "interview"], default=None,
                         help="Restrict to one stage before applying --limit")
    args = parser.parse_args()

    dataset_dir = ANNOTATION_VIDEOS_DIR / args.dataset
    vqa_file = dataset_dir / "vqa.json"
    ground_truth_file = dataset_dir / "annotations" / "ground_truth_actions.json"
    label_annotations_file = dataset_dir / "annotations" / "label_annotations.json"
    output_file = dataset_dir / "adversarial_items.json"
    if not vqa_file.exists():
        sys.exit(f"vqa.json not found: {vqa_file} (run vqa_pipeline/build_vqa.py first)")

    ground_truth_actions = json.loads(ground_truth_file.read_text()) if ground_truth_file.exists() else []

    def action_sheet_for(correct_answer: str) -> str:
        """Excludes entries within MIN_GAP_SECONDS of the item's own correct answer, so
        whichever two the model picks are already gap-compliant by construction."""
        correct_span = parse_timespan(correct_answer)
        eligible = ground_truth_actions if correct_span is None else [
            e for e in ground_truth_actions
            if gap_seconds(correct_span, (e["start"], e["end"])) >= MIN_GAP_SECONDS
        ]
        return "\n".join(f"{to_timespan(e['start'], e['end'])}: {ground_truth_action_text(e)}" for e in eligible)

    goal_context = load_goal_context(dataset_dir, json.loads(label_annotations_file.read_text())) \
        if label_annotations_file.exists() else ""

    from dotenv import load_dotenv
    load_dotenv(DOTENV_FILE)
    _creds_path = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    if not _creds_path.is_absolute():
        _creds_path = (DOTENV_FILE.parent / _creds_path).resolve()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_creds_path)
    vertex_project = json.loads(_creds_path.read_text())["project_id"]

    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=vertex_project, location=VERTEX_LOCATION)
    last_call = [0.0]

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=2, min=4, max=60),
           retry=retry_if_exception(_is_retryable), reraise=True)
    def call_gemini(prompt: str) -> tuple[str, dict]:
        wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - last_call[0])
        if wait > 0:
            time.sleep(wait)
        last_call[0] = time.monotonic()
        response = client.models.generate_content(
            model=MODEL, contents=[prompt],
            config=types.GenerateContentConfig(temperature=ADVERSARIAL_TEMPERATURE),
        )
        return response.text, _response_metadata(response)

    items = json.loads(vqa_file.read_text())
    if args.stage is not None:
        items = [i for i in items if i["stage"] == args.stage]

    output = load_output(output_file, args.dataset)
    done_keys = {r["key"] for r in output["results"]}
    todo = [i for i in items if f"{i['annotation_id']}:{i['stage']}" not in done_keys]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"{len(items)} vqa questions selected, {len(todo)} remaining ({len(done_keys)} already done).")
    if not todo:
        print("Nothing to do.")
        return

    for i, item in enumerate(todo, 1):
        key = f"{item['annotation_id']}:{item['stage']}"
        print(f"[{i}/{len(todo)}] {key} ({item['category']!r})")

        if item["stage"] in ("stage1", "interview"):  # both are prose "why" answers, same shape
            prompt = STAGE1_PROMPT.format(
                question=item["question"], correct_answer=item["correct_answer"], goal_context=goal_context,
                correct_answer_word_count=len(item["correct_answer"].split()),
            )
        else:
            prompt = STAGE_TIMESTAMP_PROMPT.format(
                question=item["question"], correct_answer=item["correct_answer"],
                action_sheet=action_sheet_for(item["correct_answer"]),
            )

        response, usage = call_gemini(prompt)
        try:
            parsed = parse_json_response(response)
            answers = [parsed["answer1"], parsed["answer2"]]
            # normalize timestamp-type answers to the bare "Xs-Ys" form the eval uses
            if item["stage"] not in ("stage1", "interview"):
                normed = [parse_timespan(a) for a in answers]
                if any(span is None for span in normed):
                    raise ValueError(f"answer not a timespan: {answers!r}")
                answers = [to_timespan(*span) for span in normed]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  parse failure ({e}); storing raw response")
            answers = None

        output["results"].append({
            "key": key,
            "annotation_id": item["annotation_id"],
            "stage": item["stage"],
            "category": item["category"],
            "question": item["question"],
            "adversarial_answers": answers,
            "raw_response": response if answers is None else None,
            "usage": usage,
        })
        save_output(output_file, output)  # incremental — safe to interrupt/resume

    print(f"Wrote {len(output['results'])} total results to {output_file}")


if __name__ == "__main__":
    main()

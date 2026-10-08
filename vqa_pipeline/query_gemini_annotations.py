"""
Query Gemini for each label-tool annotation in a dataset, in three stages.

Reads annotations/label_annotations.json for the given dataset, filters down
to the categories we care about (the union of every stage's target
categories — see ALL_TARGET_CATEGORIES). Each stage independently decides,
via its own *_TARGET_CATEGORIES, whether it runs for a given annotation:

Stage 1 (one call):
  - category + transcript excerpt + text context (clip_summary entries from
    annotations/inferred_states.json, start of video through the
    annotation's reference_end) + a 30-second silent video clip centered on
    reference_start, at media_resolution=low, 1 fps.
  Only runs for STAGE1_TARGET_CATEGORIES (PLAN, ASSESSMENT,
  INTENT_REASONING, RISK_MANAGEMENT).

Stage 2 (one call):
  - Same context as stage 1 (text context + same video window), with one
    addition: the text context also includes each segment's
    inferred_local_state.current_action alongside its clip_summary, again
    only up through reference_end.
  Only runs for STAGE2_TARGET_CATEGORIES (PLAN).

Stage 3 (one call, same shape as stage 1):
  - dialogue point (transcript excerpt of an ASSESSMENT-category
    annotation) + text context (the full contents of
    annotations/ground_truth_actions.json for the dataset — the whole file,
    not filtered to reference_end, since it's the ground-truth action/state
    log for the entire experiment) + the same video window as stages 1/2.
  Only runs for STAGE3_TARGET_CATEGORIES (ASSESSMENT). Datasets
  without ground_truth_actions.json (only session01 has it so far) print a
  warning and skip stage 3 for every annotation.

Distractor answers are no longer generated here — see the adversarial
distractor script instead.

Calls Vertex AI (not the Gemini Developer API — the Files API used by the
old upload-once approach isn't available on Vertex at all), authenticating
via the service account key that GOOGLE_APPLICATION_CREDENTIALS in
the repo-root .env points to. Since there's no Files API, video
context is sent as inline bytes per call: a silent ~30s clip is cut with
ffmpeg for each annotation and reused across all of its stage 1, stage 2,
and stage 3 calls (never uploaded anywhere — just base64'd into the request).

Output is written incrementally (after every annotation) to
<dataset>/gemini_queries.json, so an interrupted run doesn't lose progress —
re-running skips annotation ids already present in the output file.

--interview switches to an entirely separate mode: instead of label_annotations.json,
reads annotations/interview_annotations.json and generates one "why"/intent question
(same shape as stage 1) per core_highlights span marked on any entry — category is
ignored, since the ACTION/PLAN/etc. taxonomy doesn't fit retrospective interview
commentary well. An entry with N highlights produces N separate prompts, not one blended
one: each call gets the entry's FULL transcript_text with just its own highlight wrapped
in <focus>...</focus> tags (see tagged_transcripts_for) and is told to build the question
only around the tagged part, using the rest as surrounding context. Also included:
the same context_up_to(...) prior-experiment summary stage 1 uses. Video is the WHOLE
reference video (the experiment footage, not the interview clip) — a Vertex CachedContent
built ONCE per dataset and reused identically for every call (same trick evaluate_vqa.py
uses for its stage2/3 shared clip). There's no per-call way to slice a cached video to
just one entry's [reference_start, reference_end] window (a cached Part's video_metadata
is fixed at cache-creation time, not overridable per generate_content call), so instead —
matching how evaluate_vqa.py's stage2/3 already works — the prompt itself states that
window in text and the model locates it within the full video it can already see. Written
to <dataset>/interview_queries.json, keyed "entry_id:highlight_index" (same "id:index"
composite-key convention evaluate_vqa.py uses), incrementally, resumable the same way.

Usage:
    python3 vqa_pipeline/query_gemini_annotations.py --dataset session01
    python3 vqa_pipeline/query_gemini_annotations.py --dataset session02 --limit 5
    python3 vqa_pipeline/query_gemini_annotations.py --dataset session01 --interview
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

import vertex_cache
from goal_context import load_goal_context
from dataset_config import video_floor_seconds

REPO_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_VIDEOS_DIR = Path(os.environ.get("COGVQA_DATA_DIR", REPO_ROOT / "data"))
DOTENV_FILE = REPO_ROOT / ".env"

# Vertex AI, not the Gemini Developer API. gemini-3.1-pro-preview is only
# available via location="global"
# (404s in us-central1 despite showing up in that region's model catalog).
MODEL = "gemini-3.1-pro-preview"
VERTEX_LOCATION = "global"

STAGE1_TARGET_CATEGORIES = ["PLAN", "ASSESSMENT", "INTENT_REASONING", "RISK_MANAGEMENT"]
STAGE2_TARGET_CATEGORIES = ["PLAN"]
STAGE3_TARGET_CATEGORIES = ["ASSESSMENT"]
# Union of every stage's target categories — an annotation is selected for
# processing at all if it falls in here, then each stage independently
# decides (via its own *_TARGET_CATEGORIES) whether to run for it. Every
# stage-3 category is also a stage-1 category, so stage 3 currently adds
# nothing to this union.
ALL_TARGET_CATEGORIES = sorted(set(STAGE1_TARGET_CATEGORIES) | set(STAGE2_TARGET_CATEGORIES) | set(STAGE3_TARGET_CATEGORIES))
VIDEO_WINDOW_SECONDS = 30
PRECEDING_TRANSCRIPT_LINES = 2

# --interview mode: encoding params for the ONE whole-reference-video CachedContent shared
# across every interview entry (same idea as evaluate_vqa.py's stage2/3 shared clip).
INTERVIEW_FPS = 0.5
INTERVIEW_VIDEO_SCALE_WIDTH = 480
INTERVIEW_CRF = 32
INTERVIEW_CACHE_TTL_SECONDS = 21600  # 6h — long enough to outlast a full run's thinking latency

# Stage 3 sees the SAME video the eval will: one whole-recording CachedContent rather than the
# narrow per-annotation clip stages 1 and 2 use. These constants must stay equal to
# evaluate_vqa.py's STAGE23_FPS / STAGE_VIDEO_SCALE_WIDTH / STAGE23_CRF / VIDEO_FLOOR_SECONDS —
# the cache key below is byte-identical to the eval's, so when the models match, generation and
# evaluation share one cache instead of each encoding and uploading the same video.
#
# Matching the eval's ENCODING is the point, not an accident of reuse: a subtle color or texture
# cue that survives a 1fps close clip but not 0.5fps@480px is a cue the test taker will never
# see, so an item resting on it is unanswerable. Generating at higher fidelity than the eval is
# exactly how those unanswerable items got made.
STAGE3_FPS = 0.5
STAGE3_VIDEO_SCALE_WIDTH = 480
STAGE3_CRF = 32
STAGE3_CACHE_TTL_SECONDS = 21600

STAGE1_PROMPT1_TEMPLATE = """
    You are writing a multiple choice test for advanced, graduate-level scientists.
    You will be given a scientific experiment and some context.
    You will also be given the experimenter's real comments about a certain part of the experiment.

    Your task is to create a multiple choice question based on that comment and the surrounding context. The question should test the student's ability to anticipate and understand what a real experimenter is thinking during this specific experimenter. The students will not have access to the experimenters' comments, only the video of the experiment itself. Therefore, the question should not reference the comments at all, only the video. The goal is to test whether the test takers have developed a seasoned scientific understanding of the experiment that matches the experimenters'.

    Do not include any context at all in the question. For instance, it is bad to include in the question a phrase like "Given that this is a nitration."

    You will also be given background reference material (a lab protocol, or other written or spoken
    description of the experiment's goal) in addition to the video. This material often contains specific
    facts — reagent names, volumes, step order, what happens next. Do NOT write a question that merely asks
    the test-taker to recall or predict one of these facts (e.g. "What volume are they using?", "What reagent
    comes next?", "What material is X, and what is inside it?", "What do these numbers represent?") — that
    tests memorization of the reference material, not scientific understanding. Every question must still
    test insight into the experimenter's thinking: WHY they are doing what they're doing, what risk or
    precaution they're managing, or what they anticipate will happen next and why — never a bare fact.

    Bad example (tests recall of the reference material, not reasoning):
    Comment: "This bottle has silicon nanoparticles in it, so I'm wrapping the cap with Parafilm just to be safe."
    Question: "What material is the experimenter wrapping around the bottle, and what is likely inside?"
    Why this is bad: it only asks the test-taker to name the material and contents. It doesn't test whether
    they understand WHY wrapping it matters — the precaution against the specific risk those nanoparticles pose.
    Better: "Why is the experimenter taking the precaution of wrapping this bottle's cap?"

    For example:
    Comment: "So the reaction hasn't started yet though, because we still need to add our nitric acid." 
    Question:"Why hasn't the reaction started yet?"
    Question: "Has the reaction started? Why or why not?"
    Rationale: The experimenter notes that the reaction hasn't started. Test whether the student knows this. The experimenter notes that the reaction hasn't started because the acid hasn't been added yet. Test whether the student knows this. 

    Comment: "Usually with the addition of the tin and the HCL, things will start to loosen up and dissolve a bit" 
    Question: "What is the likely effect of the addition of tin and HCL on the solution?" 
    Rationale: The experimenter knows that adding tin and HCL will loosen up the solution. Test whether the student also knows this.
    Question: "Why is the stirring likely to improve in the future?"
    Rationale: The context of the experimenter's comments makes it clear that this is in reference to anticipating better stirring in the future. Test whether the student knows this.

    Comment: "And as we're doing that, we should notice that the solution should kind of go to a more transparent yellowy color." 
    Question: "What kind of color change should we expect to see in the solution?"
    Rationale: The experimenter is anticipating that in the future the solution will be more transparent and yellowy. Test whether the student knows this.

    Return ONLY a JSON object in this exact format:
{{
  "question_answer_pairs": {{
    "question": "<The multiple choice question that tests understanding of the experimenter's internal thoughts and comment>",
    "answer": "<The correct answer. The incorrect options will be created later.>",
    "rationale": "<The part of the experimenter's comments that the question is testing.>"
  }}

    Context (The experimenter's stated goal for the experiment): {goal_block}
    Context (Summary of what has happened in the experiment up to this point): {context_block}
    Context (Video at this point): {window}
    Comment from the experimenter at this point in the experimenter: {transcript_text}
"""

STAGE2_PROMPT_TEMPLATE = """
    You are preparing to create an advanced quiz that tests understanding of dependencies in a scientific experiment.

    You will be given a summary of the whole experiment, along with the experimenter's thoughts at a certain point in the experiment. These thoughts will be related to something that will happen in the future -- for instance, how the experimenter is currently preparing for something to happen to in the future, or else how the experimenter is currently preparing to ensure that something negative does not happen in the future.

    Your task is to find which point(s) in the experiment corresponds to that future scenario that the exerimenter is currently anticipating.

    Then, you will create a question for a student. Imagine the student is watching a video of the experiment, and has reached the future point you identified. The question should ask the student to identify at what point in the past the experimenter was preparing for this outcome (or lack of negative outcome). Note that the student will not have access to the experimenter's comments, only the video of their actions. Your question should assess how aligned the student is to the experiment's planning.

    Do not include any context at all in the question. For instance, it is bad to include in the question a phrase like "Given that this is a nitration."

    Return ONLY a JSON object in this exact format:
{{
  "dependency_question": {{
    "future_event": "<Timestamp of the future scenario you identified>",
    "question": "<Asking about when the experimenter acted to prepare for the future event>",
    "answer": "<Timestamp of the point in the video that answers the question>",
  }}

    Context (The experimenter's stated goal for the experiment): {goal_block}
    Description of the whole experiment: {context_block}

    Experimenter's thoughts at a certain point in the experiment: {transcript_text}

"""

# {ground_truth_actions} is the entire contents of ground_truth_actions.json
# (pretty-printed JSON), unfiltered — the full ground-truth action/state log
# for the whole experiment.
STAGE3_PROMPT1_TEMPLATE = """You are preparing to create an advanced quiz that tests PhD candidates understanding of an ongoing scientific experiment.

    You have the COMPLETE video of a real experiment, along with the experimenter's thoughts at one particular point in it. The comment in question is SPOKEN from {reference_start:.0f}s to {reference_end:.0f}s in that video; locate it yourself and study what is on screen there, and in the minute or so before and after -- the cue being discussed often sits outside the span of the speech itself. These thoughts will be related to an assessment of the current experimental state -- for instance, whether the reaction is proceeding normally, or whether the experimenter is noticing a possible risk that should be addressed.

    Judge both properties below against the VIDEO, not against the timestamp list. You can see the
    whole experiment, so check the neighbouring footage directly rather than reasoning about what
    the list implies. The test taker will see this same video at this same quality: if you cannot
    make out a cue yourself, neither can they, and the item is unusable.

    Your task is to create a question that asks the test taker to identify the timestamp at which the experimenter made this assessment. The test takers will not have access to the experimenter's thoughts; they will only see the video of the experiment itself. We want to test whether they have developed a seasoned scientific understanding such that their reading of the experiment matches the experimenter's.

    A good item for this test requires TWO properties. Check both before writing anything.

    (1) A DISCRETE VISUAL TRIGGER. There must be something concretely observable on screen -- a
        change in the apparatus, the mixture, an instrument readout, or the experimenter's own
        action -- that a careful viewer could point to as the reason the assessment was made.

        The timestamps you are given mark when the experimenter SPOKE, which is often NOT when
        the cue was on screen. It can fall either side. Sometimes they narrate something they
        have already seen, so the trigger comes first -- by a few seconds, occasionally by half a
        minute -- and the camera has moved on by the time the words are said. Other times they
        announce first and then show, tilting the flask toward the lens or panning to a readout
        just after speaking. So search roughly the 45 seconds before and the 30 seconds after the
        comment, and NEVER conclude a cue is absent merely because it is off screen during the
        speech itself. Report the moment you actually found it in trigger_timestamp.

        The trigger also only has to be PART of what prompted the assessment, not the whole of
        it. Experimenters constantly combine what they see with arithmetic, prior knowledge, or
        memory of earlier steps: reading a level off a graduated vial and subtracting to get
        what remains is still triggered by something on screen. Do not reject an item because
        the conclusion required a calculation or expertise to reach -- that is what makes it
        worth testing.

        The widened window is for finding THIS comment's cue, not for finding a better comment.
        Whatever you settle on must be the thing THIS comment is about: if the remark is about
        the mixture's color, the trigger is that color; if it is about a temperature reading, the
        trigger is that readout. A vivid, easily-described event happening nearby is NOT a
        licence to build the question around it instead. Ask yourself: would the experimenter
        have said these exact words because of the thing I picked? If not, you have the wrong
        trigger, however photogenic it is.

        It follows that when the comment carries no assessment at all -- logistics ("I need new
        gloves"), narration of one's own action, a statement of what comes next -- the answer is
        to DECLINE. Do not rescue such a moment by substituting a nearby event that does carry an
        assessment; that produces a question the source comment cannot answer. Declining is the
        correct, expected outcome here, and it is much better than a fabricated item.

        "Logistics" here means a remark with NO observable basis -- housekeeping the experimenter
        would have said regardless of what the apparatus was doing. It does NOT mean any remark
        that mentions a quantity or a step. "6.3 mils, we probably added like 1.3, so we probably
        still have about 5 mils left" is read off the graduations of a vial the camera is looking
        at; it is an assessment of how far the addition has got, and it is exactly the kind of
        quantitative read-plus-inference worth testing. Before calling something logistics, ask
        whether the experimenter had to LOOK at anything to say it. If they did, it is not
        logistics.

        Property (1) fails only if nothing anywhere in that surrounding window is observable --
        an assessment the experimenter merely thought, anticipated, or remembered, with no
        on-screen correlate at all.

    (2) LOCAL DISTINGUISHABILITY. The test taker will choose between timestamps that are at
        least 100 seconds apart, so the trigger only needs to be distinguishable from its own
        neighbourhood -- not from the entire experiment. A slow or continuous change is
        perfectly acceptable: a mixture that yellows steadily over five minutes, a temperature
        that holds flat for a long stretch, or a solid that stays visible throughout are all
        fine, so long as this is the point at which the experimenter would actually remark on
        it. Reject only when a distinctly separate, comparably good moment sits within roughly
        two minutes and would prompt the same inference just as well.

    If either property fails, DO NOT invent a question. Return the unsuitable form described
    below instead. It is much better to decline than to produce an item whose answer cannot be
    defended; many candidate moments in an experiment are legitimately unsuitable.

    If both properties hold, write the question. The question must name the INFERENCE the
    experimenter drew, not the percept they saw. A question that restates the observation turns
    the task into visual search and tests nothing about scientific judgment.

    Be careful not to over-apply this. Most assessments in a lab are phrased AS an observation
    with a verdict attached, and the verdict is the inference -- it does not have to be spelled
    out as a separate sentence. "This orangier state is common" says the color is EXPECTED;
    "that's a really good sign" says the reaction is on track; "I'm not too worried about that"
    says a visible anomaly is benign. All of those are assessments, and all are fair game. You
    are also given the couple of transcript lines preceding the comment: use them. A short line
    like "Getting more yellow" may look like a bare percept alone while plainly continuing a
    running series of progress checks in context. Reject for this reason only when the comment
    carries no evaluative content at all, in its context -- a pure naming of what is on screen.

    BAD  (restates the percept): "At what timestamp does the reaction mixture visibly become
         more transparent?"
    GOOD (names the inference):  "At what moment does the experimenter first have visual
         confirmation that the reaction is proceeding as intended?"

    BAD  (restates the percept): "At what timestamp does the experimenter put on a fresh pair
         of gloves?"
    GOOD (states the reason behind the observation): "At what point does the experimenter realize that there may be a risk of chemical transfer?"

    Worked example of a suitable item:

    Experimenter comment: It means we probably are getting a little bit of product in there.
    Comment spoken at: 2218s
    trigger_timestamp: 2204s
    visual_evidence: The contents of the flask shift to a lighter color, visible through the glass. This happens at 2204s, about fourteen seconds before the experimenter remarks on it -- by 2218s the camera has already drifted toward the bench.
    why_not_elsewhere: No other point within a couple of minutes either side would prompt this inference -- the mixture is a steady darker color beforehand, and nothing else in that neighbourhood suggests product is forming. The color continues to deepen later on, but this is where the experimenter would first call it.
    Question: What moment in the video gives the experimenter the first evidence that product is forming?
    Answer: 2218s
    Rationale: The experimenter connects the color change to product formation. Test whether the students can draw the same inference from the same visual cue.

    Do not include any context at all in the question. For instance, it is bad to include in the question a phrase like "Given that this is a nitration."

    Context (The experimenter's stated goal for the experiment): {goal_block}

    Segment boundaries for the whole experiment, as bare timestamp spans. These tell you how the
    experiment is carved up, NOT what happens in each span -- the descriptions are withheld on
    purpose, because the test taker doesn't get them either. Anything you claim is observable must
    come from watching the video:
{ground_truth_actions}

    Experimenter's comment at this point ({reference_start:.0f}s-{reference_end:.0f}s). The line
    inside <focus>...</focus> is THE ONE you are assessing, and the timestamps above are its
    timestamps. Any lines before it are earlier narration, included only so you can tell what the
    focused line means -- they are NOT what you are assessing. Build the question around the
    focused line alone. If the focused line carries no assessment, DECLINE: do not reach back to
    a context line that does, because the answer would then point at the focused line's
    timestamp while the question described a different moment.
{transcript_text}

    If the moment is suitable, return ONLY a JSON object in this exact format:
{{
  "assessment_question": {{
    "trigger_timestamp": "<When the cue is actually ON SCREEN, in seconds. This need not fall inside the spoken comment's span -- say where you really found it.>",
    "visual_evidence": "<What is concretely observable on screen at that timestamp that triggered the assessment>",
    "why_not_elsewhere": "<Why no nearby moment -- roughly two minutes either side -- fits this question as well as the one you chose. A gradual change is fine; say why this is the point the experimenter would remark on it. Refer to the surrounding footage and the ground truth actions.>",
    "question": "<Asking when the experimenter drew this inference about the experimental state>",
    "answer": "<Timestamp of the point in the video>",
    "rationale": "<Explanation of what the question is testing>"
  }}
}}

    If the moment fails either property, return ONLY:
{{
  "assessment_question": {{
    "unsuitable": "<Which property fails and why>"
  }}
}}
"""

# --interview mode: same "why" question shape as stage 1, but the comment comes from a
# LATER interview reflecting back on the experiment (highlighted excerpt only) rather than
# live narration, and the video window is that entry's own reference span (via cache +
# start_offset/end_offset) instead of a fixed ~30s clip.
INTERVIEW_INTENT_PROMPT_TEMPLATE = """
    You are writing a multiple choice test for advanced, graduate-level scientists.
    You will be given a scientific experiment and some context. You have access to the FULL
    video of the experiment. You will also be given an excerpt from a LATER interview in which
    the experimenter reflects back on this experiment, and the point in the video (in seconds
    from the start) that excerpt is about.

    Part of that excerpt is wrapped in <focus>...</focus> tags. The rest of the excerpt is only
    there to help you understand the tagged part in context — your question and answer must be
    built entirely around what is inside the <focus> tags, not around the surrounding sentences.

    Your task is to create a multiple choice question based on the <focus>-tagged reflective
    comment and the surrounding context. The question should test the student's ability to
    anticipate and understand what the experimenter was thinking during this specific moment of
    the experiment. The students will not have access to the experimenter's interview comments,
    only the video of the experiment itself. Therefore, the question should not reference the
    comments at all, only the video. The goal is to test whether the test takers have developed a
    seasoned scientific understanding of the experiment that matches the experimenter's own
    retrospective account of their intentions and thought process.

    The student has good general chemistry knowledge, but you are specifically testing whether they are ready to apply it in an experimental setting. So, identify something interesting or possibly surprising to a student in the scientist's comments. Then, target that knowledge in your question.

    Do not include any context at all in the question. For instance, it is bad to include in the
    question a phrase like "Given that this is a nitration."

    Return ONLY a JSON object in this exact format:
{{
  "question_answer_pairs": {{
    "question": "<The multiple choice question that tests understanding of the experimenter's intentions at this point>",
    "answer": "<The correct answer. The incorrect options will be created later.>",
    "rationale": "<What part of the scientist's comments is probably surprising or challenging to students, and what part of that concept does the question target>"
  }}
}}

    Point in the video this excerpt is about: {reference_start:.1f}s to {reference_end:.1f}s
    Context (Summary of what has happened in the experiment up to this point): {context_block}
    Excerpt from a later interview, reflecting on this point in the experiment
    (build the question around the <focus> tagged part only): {transcript_text}
"""


def _parse_activity_timestamp(ts: str) -> tuple[float, float]:
    """"activity_1_0s_140s" -> (0.0, 140.0)"""
    parts = ts.split("_")
    return float(parts[-2].rstrip("s")), float(parts[-1].rstrip("s"))


def _video_duration_seconds(video_path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def _extract_clip(video_path: Path, start: float, end: float, out_path: Path) -> None:
    """Cut a silent [start, end) clip out of video_path into out_path (re-encoded, since
    a short clip like this is cheap to encode and we want an exact-length standalone file)."""
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{end - start:.3f}",
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(out_path)],
        check=True, capture_output=True,
    )


def _extract_downscaled_clip(video_path: Path, start: float, end: float, out_path: Path,
                              fps: float, scale_width: int, crf: int) -> None:
    """Like _extract_clip, but downsampled in fps/resolution — for --interview mode's whole-
    reference-video shared clip, where native quality would be far too large to cache."""
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{end - start:.3f}",
         "-an", "-vf", f"fps={fps},scale={scale_width}:-2",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), str(out_path)],
        check=True, capture_output=True,
    )


def load_interview_annotations(dataset_dir: Path) -> dict:
    path = dataset_dir / "annotations" / "interview_annotations.json"
    if not path.exists():
        sys.exit(f"interview_annotations.json not found: {path}")
    return json.loads(path.read_text())


HIGHLIGHT_TAG_OPEN = "<focus>"
HIGHLIGHT_TAG_CLOSE = "</focus>"


def tagged_transcripts_for(entry: dict) -> list[str]:
    """One item per entry['core_highlights'] span (character offsets into transcript_text),
    in span order — NOT joined into one blended comment. Each item is the entry's FULL
    transcript_text with just that one highlight wrapped in <focus>...</focus>, so the model
    sees the surrounding context but is pointed at exactly one thing to build the question
    around; a multi-highlight entry therefore produces multiple separate prompts, one per
    highlight, each with only its own span tagged. Empty list if the entry has no highlights,
    so callers can filter those out entirely."""
    highlights = entry.get("core_highlights") or []
    if not highlights:
        return []
    text = entry.get("transcript_text", "")
    ranges = sorted(highlights, key=lambda h: h["start"])
    return [
        text[:h["start"]] + HIGHLIGHT_TAG_OPEN + text[h["start"]:h["end"]] + HIGHLIGHT_TAG_CLOSE + text[h["end"]:]
        for h in ranges
    ]


# ── CLI ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", required=True, help="Dataset folder name, e.g. session01 or session02")
parser.add_argument("--limit", type=int, default=None, help="Only process the first N matching annotations (for testing)")
parser.add_argument("--category", choices=ALL_TARGET_CATEGORIES, default=None,
                     help="Restrict to a single category before applying --limit (for controlled test runs)")
parser.add_argument("--redo-stage2", action="store_true",
                     help="Instead of processing new annotations, re-run stage 2 for annotations already "
                          "present in the output file (whose category is in STAGE2_TARGET_CATEGORIES), "
                          "overwriting just their stage2 result. Stage 1 is left untouched.")
parser.add_argument("--skip-stage2", action="store_true",
                     help="Don't make stage 2 calls at all this run (stage2 is written as null for every "
                          "annotation, regardless of category) — for a stage1-only pass where you don't want "
                          "to pay for stage2 calls you won't use.")
parser.add_argument("--redo-stage3", action="store_true",
                     help="Instead of processing new annotations, re-run stage 3 for annotations already "
                          "present in the output file (whose category is in STAGE3_TARGET_CATEGORIES), "
                          "overwriting just their stage3 result. Stages 1 and 2 are left untouched. "
                          "Combine with --category/--limit/--only-unsuitable to regenerate only a subset.")
parser.add_argument("--only-unsuitable", action="store_true",
                     help="Modifier for --redo-stage3: restrict the redo to annotations whose EXISTING "
                          "stage3 result declined to produce a question (an \"unsuitable\" verdict). Use "
                          "after relaxing the prompt's suitability rules, to retry just the moments the "
                          "previous, stricter wording rejected without spending calls on the ones it kept.")
parser.add_argument("--only-stale", action="store_true",
                     help="Modifier for --redo-stage3: restrict the redo to annotations whose EXISTING "
                          "stage3 result predates the current prompt schema (no visual_evidence and no "
                          "unsuitable verdict). For resuming a partially-completed regeneration without "
                          "re-spending calls on the annotations already redone.")
parser.add_argument("--annotation-id", default=None,
                     help="Modifier for --redo-stage3: restrict the redo to the single annotation whose id "
                          "starts with this prefix. For re-running one item after a targeted prompt change, "
                          "without spending calls on the rest.")
parser.add_argument("--interview", action="store_true",
                     help="Switch to interview mode: generate one intent question per "
                          "annotations/interview_annotations.json entry that has core_highlights marked "
                          "(category is ignored). Writes to <dataset>/interview_queries.json instead of "
                          "gemini_queries.json. Mutually exclusive with --category/--redo-stage2/--redo-stage3, "
                          "which only apply to the label_annotations.json-based stage1/2/3 flow.")
args = parser.parse_args()
if args.interview and (args.category or args.redo_stage2 or args.redo_stage3):
    parser.error("--interview is mutually exclusive with --category/--redo-stage2/--redo-stage3")

DATASET_DIR = ANNOTATION_VIDEOS_DIR / args.dataset
STAGE3_VIDEO_FLOOR_SECONDS = video_floor_seconds(args.dataset)
ANNOTATIONS_FILE = DATASET_DIR / "annotations" / "label_annotations.json"
INFERRED_STATES_FILE = DATASET_DIR / "annotations" / "inferred_states.json"
GROUND_TRUTH_ACTIONS_FILE = DATASET_DIR / "annotations" / "ground_truth_actions.json"
TRANSCRIPTS_DIR = DATASET_DIR / "transcripts"
OUTPUT_FILE = DATASET_DIR / "gemini_queries.json"

if not DATASET_DIR.exists():
    sys.exit(f"Dataset directory not found: {DATASET_DIR}")
if not ANNOTATIONS_FILE.exists():
    sys.exit(f"label_annotations.json not found: {ANNOTATIONS_FILE}")

from dotenv import load_dotenv
load_dotenv(DOTENV_FILE)

# GOOGLE_APPLICATION_CREDENTIALS in the .env is a relative path (key.json),
# which resolves against the process's cwd, not the .env file's location —
# fix it up to an absolute path so this works regardless of where we're run from.
_creds_path = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
if not _creds_path.is_absolute():
    _creds_path = (DOTENV_FILE.parent / _creds_path).resolve()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_creds_path)
VERTEX_PROJECT = json.loads(_creds_path.read_text())["project_id"]

from google import genai
from google.genai import types

INFERENCE_TEMPERATURE = 0.2
MIN_SECONDS_BETWEEN_CALLS = 3.0  # proactive throttle, same as the other vqa_pipeline scripts —
                                 # paces calls to mostly avoid 429s rather than eat long backoffs

client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
_last_call = [0.0]  # wall-clock of the last request start, for the proactive throttle


def _is_retryable(exc: BaseException) -> bool:
    """httpx.TransportError covers ReadError/ConnectError/*Timeout — the underlying httpx client
    raises these directly (e.g. "Connection reset by peer"), bypassing google.genai's
    ClientError/ServerError wrapping, so they need their own check rather than the status-code
    path below. 429 RESOURCE_EXHAUSTED is common enough on a full run that it needs retrying
    like any other transient error, not just connection failures."""
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


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=4, max=60),
       retry=retry_if_exception(_is_retryable), reraise=True)
def _call_gemini(prompt: str, clip_bytes: bytes = None, cached_content: str = None) -> tuple[str, dict]:
    """clip_bytes, if given, is sent inline (Vertex AI has no Files API to upload to).
    cached_content, if given, references a Vertex CachedContent by name instead (--interview
    mode's shared whole-reference-video clip) — mutually exclusive with clip_bytes in practice,
    though nothing here enforces that since no caller currently passes both.

    The @retry belongs here, on the actual network call — it used to sit on _response_metadata
    instead, which only ever runs on an already-successful response and can't meaningfully raise
    a retryable error, so a 429 previously crashed the whole run outright instead of backing off."""
    wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()
    contents = [prompt]
    config_kwargs = {"temperature": INFERENCE_TEMPERATURE}
    if clip_bytes is not None:
        contents.append(types.Part(
            inline_data=types.Blob(data=clip_bytes, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=1.0),
        ))
        # media_resolution must be set on the request config, not the Part —
        # Vertex rejects per-Part media_resolution ("not supported for this Model").
        config_kwargs["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_LOW
    if cached_content is not None:
        config_kwargs["cached_content"] = cached_content
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return response.text, _response_metadata(response)


def load_segment_context() -> list[dict]:
    """Per-activity-segment context: start/end, clip_summary, current_action."""
    if not INFERRED_STATES_FILE.exists():
        print(f"Warning: {INFERRED_STATES_FILE} not found — proceeding with no text context.")
        return []
    states = json.loads(INFERRED_STATES_FILE.read_text())
    segments = []
    for s in states:
        start, end = _parse_activity_timestamp(s["timestamp"])
        local = s.get("inferred_local_state", {})
        segments.append({
            "start": start,
            "end": end,
            "summary": local.get("clip_summary", ""),
            "current_action": local.get("current_action", ""),
        })
    segments.sort(key=lambda c: c["start"])
    return segments


def ground_truth_timestamps_only(actions: list[dict]) -> str:
    """The bare timestamp spans from ground_truth_actions.json, descriptions stripped.

    evaluate_vqa.py sends stage2/stage3 the ground-truth actions as TIMESTAMPS ONLY, so the model
    can't answer a timestamp question by reading the answer off the sheet. Generation used to get
    the full log, descriptions and all — so a question could be built (and its uniqueness
    "verified") against text the test taker is deliberately denied, leaving the visual evidence
    the item asserts entirely unchecked. Stage 3 now sees the same withheld form the eval uses."""
    return "\n".join(f"({a['start']}s-{a['end']}s)" for a in actions)


def load_ground_truth_actions() -> list[dict] | None:
    """Full contents of annotations/ground_truth_actions.json, or None if the dataset doesn't have one."""
    if not GROUND_TRUTH_ACTIONS_FILE.exists():
        print(f"Warning: {GROUND_TRUTH_ACTIONS_FILE} not found — stage 3 will be skipped for this dataset.")
        return None
    return json.loads(GROUND_TRUTH_ACTIONS_FILE.read_text())


def categories_of(ann: dict) -> list[str]:
    """Normalize an annotation's 'category' field to a list. It's a list when the label tool
    had emitted duplicate entries for the same [start, end) span under different categories
    (e.g. PLAN + INTENT_REASONING) — those get merged into one annotation with a list of
    categories; everything else still has a single string."""
    cat = ann["category"]
    return cat if isinstance(cat, list) else [cat]


def load_transcript(transcript_file: str | None) -> dict[int, str]:
    """id -> text for every line in the dataset's transcript file. Used to prepend a
    couple lines of preceding context ahead of each annotation's own transcript_text."""
    if not transcript_file:
        return {}
    transcript_path = TRANSCRIPTS_DIR / transcript_file
    if not transcript_path.exists():
        print(f"Warning: {transcript_path} not found — no preceding transcript context will be added.")
        return {}
    raw = transcript_path.read_text()
    # Two formats in the wild: session01 ships JSON ([{id, text}, ...]), session02 a plain .txt of
    # "[H:MM:SS.mmm] text" lines whose ORDER is the id space label_annotations.json indexes into
    # (its transcript_line_ids are 0-based and sequential). Fall back to the line-oriented reader
    # rather than raising, so a .txt dataset still gets preceding context instead of crashing.
    try:
        lines = json.loads(raw)
        return {line["id"]: line["text"] for line in lines}
    except json.JSONDecodeError:
        pass
    out = {}
    for i, line in enumerate(l for l in raw.splitlines() if l.strip()):
        out[i] = re.sub(r"^\[[^\]]*\]\s*", "", line).strip()
    return out


def transcript_text_with_preceding_context(ann: dict, transcript_lines: dict[int, str]) -> str:
    """ann['transcript_text'] prefixed with up to PRECEDING_TRANSCRIPT_LINES transcript
    lines immediately before the annotation's own transcript_line_ids span."""
    own_text = ann.get("transcript_text", "")
    line_ids = [i for i in (ann.get("transcript_line_ids") or []) if isinstance(i, int)]
    if not line_ids or not transcript_lines:
        return own_text
    first_id = min(line_ids)
    preceding_ids = [i for i in range(first_id - PRECEDING_TRANSCRIPT_LINES, first_id) if i in transcript_lines]
    if not preceding_ids:
        return own_text
    preceding_text = "\n".join(f"SCIENTIST: {transcript_lines[i]}" for i in preceding_ids)
    return f"{preceding_text}\n{own_text}"


def focus_tagged_with_context(ann: dict, transcript_lines: dict[int, str]) -> str:
    """Preceding context lines plain, the annotation's OWN line wrapped in <focus>...</focus>.

    Same trick --interview mode uses (see tagged_transcripts_for): the model sees the surrounding
    narration but is pointed at exactly one sentence to assess. Without the tag the context and
    the annotated line arrive as one undifferentiated blob, and the model cannot tell which is
    which — it read a context line ("the solution should kind of go to a more transparent yellowy
    color") as the comment under assessment and built a question around that, while the answer
    key stayed pinned to the annotation's own span. Tagging is what keeps question and answer
    describing the same moment."""
    own_text = ann.get("transcript_text", "")
    tagged = f"{HIGHLIGHT_TAG_OPEN}{own_text}{HIGHLIGHT_TAG_CLOSE}"
    # transcript_line_ids is normally a list of integer line indices, but a stray record in each
    # of session01/session02 carries a UUID string instead. Ignore anything non-integer rather
    # than crashing the run partway through: that annotation simply gets no preceding context.
    line_ids = [i for i in (ann.get("transcript_line_ids") or []) if isinstance(i, int)]
    if not line_ids or not transcript_lines:
        return tagged
    first_id = min(line_ids)
    preceding_ids = [i for i in range(first_id - PRECEDING_TRANSCRIPT_LINES, first_id) if i in transcript_lines]
    if not preceding_ids:
        return tagged
    preceding_text = "\n".join(f"SCIENTIST: {transcript_lines[i]}" for i in preceding_ids)
    return f"{preceding_text}\n{tagged}"


def format_transcript_excerpt(start: float, end: float, transcript_text: str) -> str:
    return f"[{start:.1f}s-{end:.1f}s] {transcript_text}"


def context_up_to(segments: list[dict], reference_end: float | None, include_current_action: bool) -> str:
    """reference_end=None means no cutoff — the whole experiment, start to end."""
    relevant = segments if reference_end is None else [c for c in segments if c["start"] < reference_end]
    lines = []
    for c in relevant:
        line = f"[{c['start']:.0f}s-{c['end']:.0f}s] {c['summary']}"
        if include_current_action:
            line += f"\n  Current action: {c['current_action']}"
        lines.append(line)
    return "\n".join(lines)


def load_output() -> dict:
    if OUTPUT_FILE.exists():
        return json.loads(OUTPUT_FILE.read_text())
    return {
        "dataset": args.dataset,
        "video": None,
        "transcript_file": None,
        "model": MODEL,
        "stage1_categories": STAGE1_TARGET_CATEGORIES,
        "stage2_categories": STAGE2_TARGET_CATEGORIES,
        "stage3_categories": STAGE3_TARGET_CATEGORIES,
        "video_window_seconds": VIDEO_WINDOW_SECONDS,
        "generated_at": None,
        "results": [],
    }


def save_output(output: dict) -> None:
    output["generated_at"] = datetime.now(timezone.utc).isoformat()
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))


def build_stage2_result(ann: dict, segments: list[dict], clip_bytes: bytes,
                         transcript_lines: dict[int, str], goal_context: str) -> dict:
    # Bare transcript_text, deliberately: STAGE2_PROMPT_TEMPLATE has no <focus> instructions, so
    # handing it tagged-or-context-prefixed text would change stage 2's inputs as a side effect of
    # a stage-3 fix. transcript_lines is accepted so both builders share one call shape.
    transcript_excerpt = format_transcript_excerpt(ann["start"], ann["end"], ann.get("transcript_text", ""))
    # Full experiment, not just up to reference_end — stage 2 has to look
    # into the future relative to the experimenter's comment.
    context2 = context_up_to(segments, None, include_current_action=True)
    prompt3 = STAGE2_PROMPT_TEMPLATE.format(
        transcript_text=transcript_excerpt,
        context_block=context2 or "(no prior clip summaries found)",
        goal_block=goal_context or "(no stated goal found)",
        window=VIDEO_WINDOW_SECONDS,
    )
    response3, usage3 = _call_gemini(prompt3, clip_bytes)
    return {"prompt": prompt3, "response": response3, "usage": usage3}


def build_stage3_cache(video_path: Path) -> str:
    """One whole-recording CachedContent for stage 3, shared across every annotation in the run.

    The cache key is byte-identical to evaluate_vqa.py's stage2/3 key, so when generation and
    evaluation run against the same model this reuses the eval's existing cache (and vice versa)
    rather than re-encoding and re-uploading the same video — the same sharing --interview mode
    and the eval already do for the interview video.

    start_offset is baked into the cached Part at creation time: a cached Part's video_metadata
    is fixed once the cache exists and can't be overridden per call, which is also why the
    annotation's own [start, end] window is stated in the prompt text rather than sliced here."""
    duration = _video_duration_seconds(video_path)

    def build_video_only_contents():
        with tempfile.TemporaryDirectory() as tmp_dir:
            clip_path = Path(tmp_dir) / "stage3_whole.mp4"

            def extract_at(crf):
                # Encode from 0 (NOT from the floor) so the video keeps ORIGINAL experiment time —
                # the intro is excluded at request time via start_offset. Cutting instead would
                # shift every timestamp the model reports, and stage 3's answers are timestamps.
                _extract_downscaled_clip(video_path, 0.0, duration, clip_path,
                                          STAGE3_FPS, STAGE3_VIDEO_SCALE_WIDTH, crf)
                return clip_path.read_bytes()

            data, _ = vertex_cache.encode_under_size_limit(extract_at, STAGE3_CRF)
        print(f"  cut whole-video clip ({len(data) / 1e6:.1f} MB)")
        return [types.Part(
            inline_data=types.Blob(data=data, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(
                fps=STAGE3_FPS, start_offset=f"{int(STAGE3_VIDEO_FLOOR_SECONDS)}s"),
        )]

    video_sig = (lambda st: f"{video_path.name}:{st.st_mtime_ns}:{st.st_size}:{duration:.1f}"
                 )(video_path.stat())
    cache_key = (f"eval_stage23_video|model={MODEL}|fps={STAGE3_FPS}|scale={STAGE3_VIDEO_SCALE_WIDTH}"
                 f"|crf={STAGE3_CRF}|floor={STAGE3_VIDEO_FLOOR_SECONDS}|video={video_sig}")
    print(f"Stage 3 video context: whole video [0s-{duration:.1f}s] at {STAGE3_FPS}fps "
          f"@{STAGE3_VIDEO_SCALE_WIDTH}px, start_offset={int(STAGE3_VIDEO_FLOOR_SECONDS)}s (cache)...")
    cache_name, reused = vertex_cache.get_or_create(
        client, types, DATASET_DIR, cache_key, MODEL, STAGE3_CACHE_TTL_SECONDS, build_video_only_contents,
    )
    print(f"  {'reusing live cache' if reused else 'cache created'}: {cache_name} "
          f"(ttl={STAGE3_CACHE_TTL_SECONDS}s)")
    return cache_name


def build_stage3_result(ann: dict, ground_truth_actions: list[dict], cache_name: str,
                         transcript_lines: dict[int, str], goal_context: str) -> dict:
    # The annotation's own line, <focus>-tagged, preceded by up to PRECEDING_TRANSCRIPT_LINES of
    # context. Judging a bare sentence like "Getting more yellow." in isolation makes it look like
    # a pure percept; the preceding narration is what shows it's one in a running series of
    # progress assessments. The tag is what stops the model assessing the context instead.
    transcript_excerpt = format_transcript_excerpt(
        ann["start"], ann["end"], focus_tagged_with_context(ann, transcript_lines))
    # Dialogue point + bare segment timestamps + the WHOLE video (cached).
    prompt1 = STAGE3_PROMPT1_TEMPLATE.format(
        transcript_text=transcript_excerpt,
        ground_truth_actions=ground_truth_timestamps_only(ground_truth_actions),
        goal_block=goal_context or "(no stated goal found)",
        reference_start=ann["start"],
        reference_end=ann["end"],
    )
    response1, usage1 = _call_gemini(prompt1, cached_content=cache_name)
    return {"prompt1": prompt1, "response1": response1, "usage1": usage1}


def redo_stage2(output: dict, video_path: Path, segments: list[dict], goal_context: str) -> None:
    """Re-run stage 2 only, for annotations already present in the output file."""
    targets = [r for r in output["results"] if set(categories_of(r)) & set(STAGE2_TARGET_CATEGORIES)]
    if args.category is not None:
        targets = [r for r in targets if args.category in categories_of(r)]
    if args.limit is not None:
        targets = targets[: args.limit]

    transcript_lines = load_transcript(output.get("transcript_file"))
    print(f"Redoing stage 2 for {len(targets)} existing result(s) in {STAGE2_TARGET_CATEGORIES}. "
          "Stage 1 is untouched.")
    if not targets:
        print("Nothing to do.")
        return

    for i, r in enumerate(targets, 1):
        vw = r["video_window"]
        print(f"[{i}/{len(targets)}] annotation {r['annotation_id']} ({r['category']}), "
              f"video window {vw['start']:.1f}s-{vw['end']:.1f}s")
        with tempfile.TemporaryDirectory() as tmp_dir:
            clip_path = Path(tmp_dir) / f"{r['annotation_id']}.mp4"
            _extract_clip(video_path, vw["start"], vw["end"], clip_path)
            clip_bytes = clip_path.read_bytes()
        r["stage2"] = build_stage2_result(r["annotation"], segments, clip_bytes, transcript_lines, goal_context)
        save_output(output)  # incremental write — safe to interrupt/resume

    print(f"Redid stage 2 for {len(targets)} result(s) in {OUTPUT_FILE}")


def stage3_declined(result: dict) -> bool:
    """True if this result's existing stage3 response is an "unsuitable" verdict — the prompt's
    way of declining a moment that has no discrete visual trigger, or whose trigger isn't
    distinguishable from its neighbours. A result with no stage3 at all, or whose stage3 response
    doesn't parse, counts as declined too: both are states --only-unsuitable should retry rather
    than leave stuck."""
    stage3 = result.get("stage3")
    if not stage3:
        return True
    text = stage3["response1"].strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        aq = json.loads(text)["assessment_question"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return True
    return bool(aq.get("unsuitable"))


def stage3_is_legacy(result: dict) -> bool:
    """True if this result's stage3 response predates the current prompt's schema.

    The current prompt returns either a suitability verdict ("unsuitable") or a question carrying
    "visual_evidence"/"why_not_elsewhere"; the older one returned a bare question/answer/rationale.
    Anything with neither marker was produced by the old prompt and still needs regenerating."""
    stage3 = result.get("stage3")
    if not stage3:
        return True
    text = stage3["response1"].strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        aq = json.loads(text)["assessment_question"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return True
    return not (aq.get("unsuitable") or aq.get("visual_evidence"))


def redo_stage3(output: dict, video_path: Path, ground_truth_actions: list[dict] | None,
                 goal_context: str) -> None:
    """Re-run stage 3 only, for annotations already present in the output file."""
    if ground_truth_actions is None:
        print(f"No ground_truth_actions.json for this dataset — stage 3 can't be redone.")
        return

    targets = [r for r in output["results"] if set(categories_of(r)) & set(STAGE3_TARGET_CATEGORIES)]
    if args.category is not None:
        targets = [r for r in targets if args.category in categories_of(r)]
    if args.only_unsuitable:
        targets = [r for r in targets if stage3_declined(r)]
    if args.only_stale:
        targets = [r for r in targets if stage3_is_legacy(r)]
    if args.annotation_id:
        targets = [r for r in targets if r["annotation_id"].startswith(args.annotation_id)]
    if args.limit is not None:
        targets = targets[: args.limit]

    transcript_lines = load_transcript(output.get("transcript_file"))
    print(f"Redoing stage 3 for {len(targets)} existing result(s) in {STAGE3_TARGET_CATEGORIES}"
          f"{' whose previous result was unsuitable' if args.only_unsuitable else ''}. "
          "Stages 1 and 2 are untouched.")
    if not targets:
        print("Nothing to do.")
        return

    cache_name = build_stage3_cache(video_path)

    for i, r in enumerate(targets, 1):
        ann = r["annotation"]
        print(f"[{i}/{len(targets)}] annotation {r['annotation_id']} ({r['category']}), "
              f"moment {ann['start']:.1f}s-{ann['end']:.1f}s")
        r["stage3"] = build_stage3_result(ann, ground_truth_actions, cache_name, transcript_lines, goal_context)
        save_output(output)  # incremental write — safe to interrupt/resume

    print(f"Redid stage 3 for {len(targets)} result(s) in {OUTPUT_FILE}")


def load_interview_output(dataset_dir: Path) -> tuple[Path, dict]:
    path = dataset_dir / "interview_queries.json"
    if path.exists():
        return path, json.loads(path.read_text())
    return path, {
        "dataset": args.dataset,
        "video": None,
        "model": MODEL,
        "generated_at": None,
        "results": [],
    }


def save_interview_output(path: Path, output: dict) -> None:
    output["generated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(output, indent=2))


def run_interview_mode() -> None:
    interview_data = load_interview_annotations(DATASET_DIR)
    video_path = DATASET_DIR / "videos" / interview_data["reference_video"]
    if not video_path.exists():
        sys.exit(f"reference video not found: {video_path}")
    duration = _video_duration_seconds(video_path)
    segments = load_segment_context()

    # One item per highlight, not per entry — a multi-highlight entry produces one prompt for
    # each of its highlights, tagged separately (see tagged_transcripts_for). item_id follows
    # the "id:index" composite-key convention used elsewhere in this pipeline (e.g.
    # evaluate_vqa.py's "annotation_id:stage"), so each highlight is independently resumable.
    items = [
        {"item_id": f"{entry['id']}:{idx}", "entry": entry, "tagged_text": tagged_text}
        for entry in interview_data.get("annotations", [])
        for idx, tagged_text in enumerate(tagged_transcripts_for(entry))
    ]
    if args.limit is not None:
        items = items[: args.limit]

    output_path, output = load_interview_output(DATASET_DIR)
    output["video"] = interview_data["reference_video"]
    done_ids = {r["annotation_id"] for r in output["results"]}
    todo = [it for it in items if it["item_id"] not in done_ids]

    print(f"{len(items)} interview highlights found, {len(todo)} remaining to query "
          f"({len(done_ids)} already done).")
    if not todo:
        print("Nothing to do.")
        return

    def build_cache_contents():
        with tempfile.TemporaryDirectory() as tmp_dir:
            clip_path = Path(tmp_dir) / "interview_reference_whole.mp4"

            def extract_at(crf):
                _extract_downscaled_clip(video_path, 0.0, duration, clip_path,
                                          INTERVIEW_FPS, INTERVIEW_VIDEO_SCALE_WIDTH, crf)
                return clip_path.read_bytes()

            data, _ = vertex_cache.encode_under_size_limit(extract_at, INTERVIEW_CRF)
        print(f"  cut whole-reference-video clip ({len(data) / 1e6:.1f} MB)")
        return [types.Part(
            inline_data=types.Blob(data=data, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=INTERVIEW_FPS),
        )]

    video_stat = video_path.stat()
    cache_key = (f"interview_whole|model={MODEL}|fps={INTERVIEW_FPS}|scale={INTERVIEW_VIDEO_SCALE_WIDTH}"
                 f"|crf={INTERVIEW_CRF}|video={video_path.name}:{video_stat.st_mtime_ns}:{video_stat.st_size}:{duration:.1f}")
    print(f"Shared reference-video context: whole video [0s-{duration:.1f}s] at {INTERVIEW_FPS}fps "
          f"@{INTERVIEW_VIDEO_SCALE_WIDTH}px (cache)...")
    cache_name, reused = vertex_cache.get_or_create(
        client, types, DATASET_DIR, cache_key, MODEL, INTERVIEW_CACHE_TTL_SECONDS, build_cache_contents,
    )
    print(f"  {'reusing live cache' if reused else 'cache created'}: {cache_name} (ttl={INTERVIEW_CACHE_TTL_SECONDS}s)")

    for i, item in enumerate(todo, 1):
        entry = item["entry"]
        tagged_text = item["tagged_text"]
        reference_start = entry["reference_start"]
        reference_end = entry["reference_end"]
        print(f"[{i}/{len(todo)}] interview highlight {item['item_id']}, "
              f"reference {reference_start:.1f}s-{reference_end:.1f}s")

        context_block = context_up_to(segments, reference_end, include_current_action=False)
        prompt = INTERVIEW_INTENT_PROMPT_TEMPLATE.format(
            transcript_text=tagged_text,
            context_block=context_block or "(no prior clip summaries found)",
            reference_start=reference_start,
            reference_end=reference_end,
        )
        response, usage = _call_gemini(prompt, cached_content=cache_name)

        output["results"].append({
            "annotation_id": item["item_id"],
            "entry_id": entry["id"],
            "annotation": entry,  # the exact item used to build the prompt, for traceability
            "tagged_text": tagged_text,
            "transcript_line_ids": entry.get("transcript_line_ids"),
            "reference_start": reference_start,
            "reference_end": reference_end,
            "prompt": prompt,
            "response": response,
            "usage": usage,
        })
        save_interview_output(output_path, output)  # incremental write — safe to interrupt/resume

    print(f"Wrote {len(output['results'])} total results to {output_path}")


def main():
    if args.interview:
        run_interview_mode()
        return

    data = json.loads(ANNOTATIONS_FILE.read_text())
    video_path = DATASET_DIR / "videos" / data["video"]
    duration = _video_duration_seconds(video_path)
    segments = load_segment_context()
    ground_truth_actions = load_ground_truth_actions()
    goal_context = load_goal_context(DATASET_DIR, data)

    output = load_output()
    output["video"] = data.get("video")
    output["transcript_file"] = data.get("transcript_file")

    if args.redo_stage2:
        redo_stage2(output, video_path, segments, goal_context)
        return

    if args.redo_stage3:
        redo_stage3(output, video_path, ground_truth_actions, goal_context)
        return

    annotations = [a for a in data.get("annotations", []) if set(categories_of(a)) & set(ALL_TARGET_CATEGORIES)]
    if args.category is not None:
        annotations = [a for a in annotations if args.category in categories_of(a)]
    if args.limit is not None:
        annotations = annotations[: args.limit]

    done_ids = {r["annotation_id"] for r in output["results"]}
    todo = [a for a in annotations if a["id"] not in done_ids]

    transcript_lines = load_transcript(data.get("transcript_file"))
    stage3_cache_name = [None]  # lazily built on the first stage-3 annotation; see below
    n_with_stage1 = sum(1 for a in todo if set(categories_of(a)) & set(STAGE1_TARGET_CATEGORIES))
    n_with_stage2 = 0 if args.skip_stage2 else sum(1 for a in todo if set(categories_of(a)) & set(STAGE2_TARGET_CATEGORIES))
    n_with_stage3 = sum(1 for a in todo if set(categories_of(a)) & set(STAGE3_TARGET_CATEGORIES)) if ground_truth_actions else 0
    print(f"{len(annotations)} annotations selected, {len(todo)} remaining to query "
          f"({len(done_ids)} already done). Stage 1 (1 call) runs for the {n_with_stage1} of those in {STAGE1_TARGET_CATEGORIES}; "
          f"stage 2 (1 call) runs for the {n_with_stage2} of those in {STAGE2_TARGET_CATEGORIES}"
          f"{' (skipped — --skip-stage2)' if args.skip_stage2 else ''}; "
          f"stage 3 (1 call) runs for the {n_with_stage3} of those in {STAGE3_TARGET_CATEGORIES}"
          f"{'' if ground_truth_actions else ' (skipped — no ground_truth_actions.json)'}.")
    if not todo:
        print("Nothing to do.")
        return

    half_window = VIDEO_WINDOW_SECONDS / 2
    n_beyond_duration = 0

    for i, ann in enumerate(todo, 1):
        reference_start = ann["start"]
        reference_end = ann["end"]
        category = ann["category"]
        categories = categories_of(ann)
        transcript_text = ann.get("transcript_text", "")
        transcript_excerpt = format_transcript_excerpt(reference_start, reference_end, transcript_text)

        window_start = max(0.0, reference_start - half_window)
        window_end = min(duration, window_start + VIDEO_WINDOW_SECONDS)

        if window_end <= window_start:
            # reference_start is beyond the video's actual duration (a label/video mismatch —
            # e.g. annotations made against a longer recording than the video file on disk).
            # Skip rather than crash ffmpeg with a negative clip duration; not written to the
            # output file, so it's picked back up automatically once the mismatch is fixed.
            print(f"[{i}/{len(todo)}] annotation {ann['id']} ({category}) skipped — "
                  f"reference_start {reference_start:.1f}s is beyond the video's {duration:.1f}s duration")
            n_beyond_duration += 1
            continue

        print(f"[{i}/{len(todo)}] annotation {ann['id']} ({category}), "
              f"video window {window_start:.1f}s-{window_end:.1f}s")

        with tempfile.TemporaryDirectory() as tmp_dir:
            clip_path = Path(tmp_dir) / f"{ann['id']}.mp4"
            _extract_clip(video_path, window_start, window_end, clip_path)
            print(f"  cut clip ({clip_path.stat().st_size / 1e6:.1f} MB)")
            clip_bytes = clip_path.read_bytes()

        # ── Stage 1: full context + video, one call.
        # Only runs for categories in STAGE1_TARGET_CATEGORIES. ──
        if set(categories) & set(STAGE1_TARGET_CATEGORIES):
            context1 = context_up_to(segments, reference_end, include_current_action=False)
            prompt1 = STAGE1_PROMPT1_TEMPLATE.format(
                transcript_text=transcript_excerpt,
                context_block=context1 or "(no prior clip summaries found)",
                goal_block=goal_context or "(no stated goal found)",
                window=VIDEO_WINDOW_SECONDS,
            )
            print("  stage 1 (context + video)...")
            response1, usage1 = _call_gemini(prompt1, clip_bytes)
            stage1_result = {"prompt1": prompt1, "response1": response1, "usage1": usage1}
        else:
            print(f"  stage 1 skipped (categories {categories!r} not in {STAGE1_TARGET_CATEGORIES})")
            stage1_result = None

        # ── Stage 2: whole-experiment context + current_action, one call.
        # Only runs for categories in STAGE2_TARGET_CATEGORIES. ──
        if args.skip_stage2:
            print("  stage 2 skipped (--skip-stage2)")
            stage2_result = None
        elif set(categories) & set(STAGE2_TARGET_CATEGORIES):
            print("  stage 2 (context + current_action + video)...")
            stage2_result = build_stage2_result(ann, segments, clip_bytes, transcript_lines, goal_context)
        else:
            print(f"  stage 2 skipped (categories {categories!r} not in {STAGE2_TARGET_CATEGORIES})")
            stage2_result = None

        # ── Stage 3: bare segment timestamps + the WHOLE video (cached), one call.
        # Only runs for categories in STAGE3_TARGET_CATEGORIES, and only if
        # the dataset has a ground_truth_actions.json. Unlike stages 1 and 2 it does NOT use
        # this annotation's narrow clip_bytes — it shares one whole-recording cache, built
        # lazily here so a run with no stage-3 annotation never pays to encode it. ──
        if set(categories) & set(STAGE3_TARGET_CATEGORIES) and ground_truth_actions is not None:
            if stage3_cache_name[0] is None:
                stage3_cache_name[0] = build_stage3_cache(video_path)
            print("  stage 3 (segment timestamps + whole video)...")
            stage3_result = build_stage3_result(ann, ground_truth_actions, stage3_cache_name[0],
                                                 transcript_lines, goal_context)
        else:
            reason = "no ground_truth_actions.json" if ground_truth_actions is None \
                else f"categories {categories!r} not in {STAGE3_TARGET_CATEGORIES}"
            print(f"  stage 3 skipped ({reason})")
            stage3_result = None

        output["results"].append({
            "annotation_id": ann["id"],
            "annotation": ann,  # the exact item used to build the prompts, for traceability
            "category": category,
            "transcript_line_ids": ann.get("transcript_line_ids"),
            "transcript_text": transcript_text,
            "reference_start": reference_start,
            "reference_end": reference_end,
            "video_window": {"start": window_start, "end": window_end},
            "stage1": stage1_result,
            "stage2": stage2_result,
            "stage3": stage3_result,
        })
        save_output(output)  # incremental write — safe to interrupt/resume

    print(f"Wrote {len(output['results'])} total results to {OUTPUT_FILE}")
    if n_beyond_duration:
        print(f"Skipped {n_beyond_duration} annotation(s) beyond the video's duration (label/video mismatch).")


if __name__ == "__main__":
    main()

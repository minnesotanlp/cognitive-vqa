"""
Generate GROUNDED adversarial answers for <dataset>/vqa.json.

Unlike vqa_pipeline/generate_adversarial.py — where the model answers blind, knowing only the
question, the correct answer and (for stage1) the experiment's stated goal — here the model
gets the SAME context the judge gets in evaluate_vqa.py: the video plus the experiment's
stated GOAL. The framing in both cases is the adversarial trick: the model is never told to
be wrong, it is told the real answer is defective and asked to do its honest best with the
context it now has. What comes back is a concrete, video-grounded, confident-sounding answer
that competes with the true answer on specificity instead of losing to it on vagueness.

  - stage1 (prose "why" answers): the real correct answer is presented as a GENERIC,
    placeholder answer — vague, written without watching this experiment, could apply to many
    experiments — and the model is asked to say what the answer REALLY is for this specific
    situation. Context is that item's own ~30s clip (gemini_queries.json's video_window,
    floored to VIDEO_FLOOR_SECONDS, 1fps) + the GOAL lines, i.e. exactly the stage1 judge
    context. Sent inline per item (nothing is shared across stage1 items).
  - stage2/stage3 (bare-timespan answers): the real correct answer is presented as a timespan
    that has already been verified WRONG, and the model is asked to find where in the video the
    asked-about moment ACTUALLY takes place. It answers by picking from the experiment's event
    boundaries with the DESCRIPTIONS WITHHELD (the same timestamps-only sheet the stage2/3 judge
    gets, minus every entry within MIN_GAP_SECONDS of the correct span — not just the correct
    span itself) — so it can't look the moment up in text, must watch the video, and whichever
    candidate it picks is already gap-compliant by construction (matches build_vqa.py's
    MIN_GAP_SECONDS requirement). Context is one shared whole-recording clip (STAGE23_FPS,
    downscaled, pre-floor intro excluded via VideoMetadata.start_offset so the model keeps
    ORIGINAL video time) + the GOAL lines; identical across every stage2/3 item, so it goes into
    a single Vertex CachedContent built once (--no-cache resends it inline instead).
  - interview (prose "why" answers, same generic-answer trick as stage1): context is the WHOLE
    reference video — a single Vertex CachedContent shared across every interview item, reusing
    the exact cache query_gemini_annotations.py --interview already built (same key format,
    model/fps/scale/crf/video identity) — plus the GOAL lines sent inline alongside it, since a
    cached video's own metadata can't be overridden per call; the item's [reference_start,
    reference_end] (read from interview_queries.json) is instead stated as text in the prompt.

Each call asks for ONE answer, and is told to match the provided answer's length and register —
a distractor that is noticeably longer or more explanatory gives itself away regardless of
content.

Reuses the eval run's plumbing (Vertex client + auth, ffmpeg clip extraction, context caching,
proactive per-minute throttle, 429/5xx-aware retry) so it stays under the project's per-minute
quota.

Output is written incrementally to <dataset>/adversarial_items_grounded.json, keyed by
"annotation_id:stage", so re-running skips questions already done. To feed these into vqa.json,
point build_vqa.py's adversarial_file at this file (or swap the filenames).

Usage:
    python3 vqa_pipeline/generate_adversarial_grounded.py --dataset session01
    python3 vqa_pipeline/generate_adversarial_grounded.py --dataset session01 --stage stage1 --limit 5
    python3 vqa_pipeline/generate_adversarial_grounded.py --dataset session01 --stage stage2 --limit 2 --no-cache --max-clip-seconds 300
"""

import argparse
import hashlib
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
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception

import vertex_cache
from goal_context import load_goal_context
from dataset_config import video_floor_seconds

REPO_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_VIDEOS_DIR = Path(os.environ.get("COGVQA_DATA_DIR", REPO_ROOT / "data"))
DOTENV_FILE = REPO_ROOT / ".env"

MODEL = "gemini-3.1-pro-preview"
VERTEX_LOCATION = "global"

# Clip settings — identical to evaluate_vqa.py, so the distractor writer sees exactly the frames
# the judge will later see.
STAGE1_FPS = 1.0
STAGE23_FPS = 0.5               # 1 frame per 2 seconds
VIDEO_SCALE_WIDTH = 480
STAGE1_CRF = 28
STAGE23_CRF = 32                # the shared whole-video clip must stay under the CachedContent
                                # inline size limit (~10MB); crf 32 @480p/0.5fps lands ~8.5MB
STAGE23_CACHE_TTL_SECONDS = 21600  # 6h — long enough to outlast a full run's thinking latency

# interview-stage: MUST match query_gemini_annotations.py's INTERVIEW_* constants exactly — the
# cache key is built the same way here, so this reuses that script's whole-reference-video cache
# instead of re-encoding it.
INTERVIEW_FPS = 0.5
INTERVIEW_VIDEO_SCALE_WIDTH = 480
INTERVIEW_CRF = 32
INTERVIEW_CACHE_TTL_SECONDS = 21600

# A little warmth so the answer is a real alternative reading rather than a bland paraphrase.
ADVERSARIAL_TEMPERATURE = 0.8
MIN_SECONDS_BETWEEN_CALLS = 3.0  # same proactive throttle as the eval — paced to stay under quota

# Stage2/stage3 candidate timespans within this many seconds of the correct answer are excluded
# before the model ever sees them — must match build_vqa.py's MIN_GAP_SECONDS, which is what
# actually enforces "clearly a different point in the experiment".
MIN_GAP_SECONDS = 100

# stage1. The "generic answer" framing: the provided answer IS the correct one, but presenting it
# as a vague placeholder is what makes the model reach for concrete, video-grounded specifics
# instead of inventing something implausible. CRITICAL caveat (added after review): the model
# tended to just bolt correct, goal-derived specifics onto the SAME underlying claim as the
# generic answer (e.g. correct="yes, it's a good sign, product is forming"; "wrong" answer="yes,
# the orange solid is the nitrated product forming" — same claim, still true, just more specific,
# making it arguably a BETTER answer than the one it's supposed to be a distractor for). The
# prompt now says outright: a more specific version of the provided answer is ALWAYS wrong for
# this task, no matter how accurate — the answer must rest on a different, incompatible claim.
STAGE1_PROMPT = """
    You are helping write a detailed answer key for a quiz about a scientific chemistry
    experiment. You have been given a video clip of the relevant moment of the experiment and
    the experimenter's stated goal for it.

    Below is a quiz question and a GENERIC answer to it — a vague placeholder that was written
    without watching this particular experiment, and that could apply to many experiments.
    Unlike whoever wrote it, you can see this specific moment.

    IMPORTANT: a more specific, more detailed, or more concrete version of the generic answer is
    ALWAYS WRONG for this task, no matter how accurate it is. Simply naming the specific
    equipment, reagent, or quantity involved in the SAME underlying claim as the generic answer
    does not count as a different answer — it is still the same answer, just dressed up, and
    must not be submitted. Your answer's core claim (the actual reason, mechanism, or judgment it
    gives) must be a DIFFERENT one from the generic answer's core claim — one that could not both
    be true at once. For example, if the generic answer's claim is "yes, this is desirable", your
    answer must not also claim "yes, this is desirable" with extra detail added; it must argue
    for a different verdict or a different underlying reason.

    Your job is to invent that different, incorrect claim, and then ground IT — not the generic
    answer — in specifics from what you can actually observe in the clip and the stated goal, so
    that it reads with the same real, concrete confidence a true answer would have: naming
    particular equipment, materials, quantities, or conditions you see, but in service of a claim
    that is actually false for this moment.

    Write ONE such answer. Before finalizing it, check yourself: if someone believed only your
    answer, would they conclude something different from, and incompatible with, the generic
    answer? If your answer could be true at the same time as the generic answer, it has failed —
    revise it so the two genuinely conflict.

    CRITICAL FORMATTING RULE: be specific in CONTENT, not in length. Your answer MUST match the
    generic answer's format and length almost exactly. The generic answer is
    {correct_answer_word_count} words long — keep your answer within plus or minus 10 words of
    that. If the generic answer is a short phrase (e.g. "to clear out the funnel"), write a short
    phrase, NOT a full sentence. If it is one or more full sentences, match that instead. An
    answer that is noticeably longer, grammatically different, or more explanatory than the
    generic answer is WRONG regardless of its content, because it makes one answer stick out too
    much — brevity and register must match, not just plausibility.

    Return ONLY a JSON object in this exact format:
    {{
      "answer": "<the specific, incorrect answer, matching the generic answer's format and length>"
    }}

    Question: {question}
    Generic answer: {correct_answer}
"""

# interview: same generic-answer trick as stage 1, but the video is the WHOLE reference video
# (via the shared cache) rather than a narrow clip already centered on the relevant moment, so
# the model needs to be told in text where in the video to look.
INTERVIEW_PROMPT = """
    You are helping write a detailed answer key for a quiz about a scientific chemistry
    experiment. You have been given the FULL video of the experiment and the experimenter's
    stated goal for it. The question below is about the moment in the video from
    {reference_start:.1f}s to {reference_end:.1f}s.

    Below is a quiz question and a GENERIC answer to it — a vague placeholder that was written
    without watching this particular experiment, and that could apply to many experiments.
    Unlike whoever wrote it, you can see this specific moment.

    IMPORTANT: a more specific, more detailed, or more concrete version of the generic answer is
    ALWAYS WRONG for this task, no matter how accurate it is. Simply naming the specific
    equipment, reagent, or quantity involved in the SAME underlying claim as the generic answer
    does not count as a different answer — it is still the same answer, just dressed up, and
    must not be submitted. Your answer's core claim (the actual reason, mechanism, or judgment it
    gives) must be a DIFFERENT one from the generic answer's core claim — one that could not both
    be true at once. For example, if the generic answer's claim is "yes, this is desirable", your
    answer must not also claim "yes, this is desirable" with extra detail added; it must argue
    for a different verdict or a different underlying reason.

    Your job is to invent that different, incorrect claim, and then ground IT — not the generic
    answer — in specifics from what you can actually observe at that moment in the video and the
    stated goal, so that it reads with the same real, concrete confidence a true answer would
    have: naming particular equipment, materials, quantities, or conditions you see, but in
    service of a claim that is actually false for this moment.

    Write ONE such answer. Before finalizing it, check yourself: if someone believed only your
    answer, would they conclude something different from, and incompatible with, the generic
    answer? If your answer could be true at the same time as the generic answer, it has failed —
    revise it so the two genuinely conflict.

    CRITICAL FORMATTING RULE: be specific in CONTENT, not in length. Your answer MUST match the
    generic answer's format and length almost exactly. The generic answer is
    {correct_answer_word_count} words long — keep your answer within plus or minus 10 words of
    that. If the generic answer is a short phrase (e.g. "to clear out the funnel"), write a short
    phrase, NOT a full sentence. If it is one or more full sentences, match that instead. An
    answer that is noticeably longer, grammatically different, or more explanatory than the
    generic answer is WRONG regardless of its content, because it makes one answer stick out too
    much — brevity and register must match, not just plausibility.

    Return ONLY a JSON object in this exact format:
    {{
      "answer": "<the specific, incorrect answer, matching the generic answer's format and length>"
    }}

    Question: {question}
    Generic answer: {correct_answer}
"""

# stage2/stage3. Same trick in the temporal register: the real answer's timespan is presented as
# one that has already been ruled out, so the model's honest best attempt to locate the moment in
# the video lands on its runner-up — a video-grounded, semantically tempting, but wrong time. The
# candidate list carries timestamps ONLY (no descriptions, correct span removed), so it cannot be
# answered by text lookup: the model has to watch.
STAGE23_PROMPT = """
    You are helping repair the answer key for a quiz about a scientific chemistry experiment. You
    have been given the video of the experiment and the experimenter's stated goal for it. Times
    in the video and in every timespan below are in the same coordinates: seconds from the start
    of the original recording.

    Below is a quiz question whose answer is a single point in time, and the timespan the answer
    key currently gives for it. That timespan has already been checked and is WRONG — the moment
    the question asks about does NOT happen there. Unlike whoever wrote the key, you can watch
    the video. Your job is to find where in the video this ACTUALLY takes place.

    Watch the video, work out which moment the question is really pointing at, and report the ONE
    timespan that covers it. Your answer must be copied exactly from the candidate list below —
    these are the experiment's event boundaries (the known-wrong timespan has already been
    removed from the list). Pick the single best one, and pick it from what you SEE happening at
    that time, not from a guess about where in the experiment it ought to fall.

    Return ONLY a JSON object in this exact format, copying the timespan exactly as it appears in
    the list (e.g. "246s-279s"):
    {{
      "answer": "<the timespan where this actually takes place>"
    }}

    Question: {question}
    Timespan the answer key gives (already verified WRONG): {correct_answer}

    Candidate timespans:
    {candidate_timespans}
"""


def to_timespan(start: float, end: float) -> str:
    return f"{int(round(start))}s-{int(round(end))}s"


def parse_timespan(text: str) -> tuple[float, float] | None:
    m = re.search(r"(\d+(?:\.\d+)?)s?\s*-\s*(\d+(?:\.\d+)?)s", text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def gap_seconds(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, max(a[0] - b[1], b[0] - a[1]))


def _extract_clip(video_path: Path, start: float, end: float, out_path: Path, fps: float,
                   crf: int) -> None:
    """Encodes at `fps` and downscaled — the fps passed to Gemini's video_metadata only controls
    how it *samples* whatever bytes it receives, so we downsample both frame rate and resolution
    here (native 1080p at native rate would be multi-GB). Same settings as evaluate_vqa.py.

    NOTE: `ffmpeg -ss` re-zeroes the output's timeline (the frame at original `start` becomes
    0s). For the stage2/3 shared clip we therefore pass start=0 and exclude the intro via
    Gemini's VideoMetadata.start_offset instead, so the model keeps ORIGINAL video time."""
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{end - start:.3f}",
         "-an", "-vf", f"fps={fps},scale={VIDEO_SCALE_WIDTH}:-2",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), str(out_path)],
        check=True, capture_output=True,
    )


def _video_duration_seconds(video_path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


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
    parser.add_argument("--no-cache", action="store_true",
                         help="Send the shared stage2/3 context (whole-video clip + GOAL text) inline on "
                              "every call instead of via a Vertex CachedContent.")
    parser.add_argument("--max-clip-seconds", type=float, default=None,
                         help="Cap the stage2/3 shared clip length (from VIDEO_FLOOR_SECONDS) for testing, "
                              "so it stays small instead of spanning the whole video.")
    args = parser.parse_args()

    dataset_dir = ANNOTATION_VIDEOS_DIR / args.dataset
    VIDEO_FLOOR_SECONDS = video_floor_seconds(args.dataset)
    vqa_file = dataset_dir / "vqa.json"
    queries_file = dataset_dir / "gemini_queries.json"
    interview_queries_file = dataset_dir / "interview_queries.json"
    ground_truth_file = dataset_dir / "annotations" / "ground_truth_actions.json"
    label_annotations_file = dataset_dir / "annotations" / "label_annotations.json"
    output_file = dataset_dir / "adversarial_items_grounded.json"
    if not vqa_file.exists():
        sys.exit(f"vqa.json not found: {vqa_file} (run vqa_pipeline/build_vqa.py first)")
    if not queries_file.exists():
        sys.exit(f"gemini_queries.json not found: {queries_file} (needed for each item's video window)")

    queries_data = json.loads(queries_file.read_text())
    queries_by_id = {r["annotation_id"]: r for r in queries_data["results"]}
    interview_queries_by_id = {}
    interview_reference_video_name = None
    if interview_queries_file.exists():
        interview_queries_data = json.loads(interview_queries_file.read_text())
        interview_queries_by_id = {r["annotation_id"]: r for r in interview_queries_data["results"]}
        interview_reference_video_name = interview_queries_data.get("video")
    label_annotations = json.loads(label_annotations_file.read_text())
    goal_context = load_goal_context(dataset_dir, label_annotations)
    goal_text = f"The experimenter's stated goal for the experiment:\n{goal_context}"

    # Timestamps only — descriptions withheld, exactly as the stage2/3 judge sees them, so the
    # model must locate the moment in the video rather than look it up in text.
    ground_truth_actions = json.loads(ground_truth_file.read_text()) if ground_truth_file.exists() else []
    all_candidate_spans = [to_timespan(e["start"], e["end"]) for e in ground_truth_actions]

    video_name = queries_data.get("video") or label_annotations.get("video")
    video_path = dataset_dir / "videos" / video_name
    if not video_path.exists():
        sys.exit(f"video not found: {video_path}")

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

    def video_part(clip_bytes: bytes, fps: float, start_offset: str | None = None):
        vm = types.VideoMetadata(fps=fps, start_offset=start_offset) if start_offset \
            else types.VideoMetadata(fps=fps)
        return types.Part(inline_data=types.Blob(data=clip_bytes, mime_type="video/mp4"),
                           video_metadata=vm)

    last_call = [0.0]

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=2, min=4, max=60),
           retry=retry_if_exception(_is_retryable), reraise=True)
    def call_gemini(prompt: str, context_parts: list | None = None,
                     cached_content: str | None = None) -> tuple[str, dict]:
        """context_parts are the video clip + GOAL text, always sent as separate Parts (never
        embedded in the prompt string) — same shape as evaluate_vqa.py's calls. cached_content is
        the stage2/3 alternative: the identical context lives server-side and is referenced by
        handle instead of resent."""
        wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - last_call[0])
        if wait > 0:
            time.sleep(wait)
        last_call[0] = time.monotonic()
        config_kwargs = {"temperature": ADVERSARIAL_TEMPERATURE}
        if any(p.inline_data is not None for p in (context_parts or [])):
            config_kwargs["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_LOW
        if cached_content is not None:
            config_kwargs["cached_content"] = cached_content
        response = client.models.generate_content(
            model=MODEL, contents=[prompt, *(context_parts or [])],
            config=types.GenerateContentConfig(**config_kwargs),
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
    if any(i["stage"] in ("stage2", "stage3") for i in todo) and not all_candidate_spans:
        sys.exit(f"ground_truth_actions.json not found or empty: {ground_truth_file} "
                  "(stage2/3 needs it for the candidate timespans)")

    # ── Stage2/3 shared context: the whole-recording clip + GOAL text, identical across all
    # stage2/3 items. Built once, then either cached (default) or resent inline (--no-cache).
    # Only built if there is stage2/3 work to do. ──
    stage23_cache_name = None
    stage23_clip_bytes = None
    stage23_start_offset = f"{int(VIDEO_FLOOR_SECONDS)}s"
    if any(i["stage"] in ("stage2", "stage3") for i in todo):
        duration = _video_duration_seconds(video_path)
        # Encode from 0 (NOT from VIDEO_FLOOR) so the clip keeps original video time; the intro
        # before VIDEO_FLOOR is excluded at request time via start_offset, not by cutting.
        clip_end = duration if args.max_clip_seconds is None \
            else min(duration, VIDEO_FLOOR_SECONDS + args.max_clip_seconds)
        print(f"Shared stage2/3 context: video [0s-{clip_end:.1f}s] at {STAGE23_FPS}fps "
              f"@{VIDEO_SCALE_WIDTH}px, start_offset={stage23_start_offset} + GOAL text"
              f"{' (INLINE, --no-cache)' if args.no_cache else ' (cache)'}...")

        def build_stage23_clip_bytes():
            def extract_at(crf):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    clip_path = Path(tmp_dir) / "stage23_shared.mp4"
                    _extract_clip(video_path, 0.0, clip_end, clip_path, STAGE23_FPS, crf)
                    return clip_path.read_bytes()
            data, _ = vertex_cache.encode_under_size_limit(extract_at, STAGE23_CRF)
            print(f"  cut shared clip ({len(data) / 1e6:.1f} MB)")
            return data

        if args.no_cache:
            stage23_clip_bytes = build_stage23_clip_bytes()
        else:
            # Deferred into build_contents() so a cache HIT (see vertex_cache.get_or_create)
            # skips the ffmpeg encode entirely instead of redoing it just to discard the bytes.
            def build_contents():
                return [video_part(build_stage23_clip_bytes(), STAGE23_FPS, start_offset=stage23_start_offset),
                        types.Part(text=goal_text)]

            st = video_path.stat()
            video_sig = f"{video_path.name}:{st.st_mtime_ns}:{st.st_size}:{clip_end:.1f}"
            goal_sig = hashlib.sha256(goal_text.encode()).hexdigest()[:16]
            cache_key = (f"grounded_stage23|model={MODEL}|fps={STAGE23_FPS}|scale={VIDEO_SCALE_WIDTH}"
                         f"|crf={STAGE23_CRF}|floor={VIDEO_FLOOR_SECONDS}|video={video_sig}|goal={goal_sig}")
            stage23_cache_name, reused = vertex_cache.get_or_create(
                client, types, dataset_dir, cache_key, MODEL, STAGE23_CACHE_TTL_SECONDS, build_contents,
            )
            print(f"  {'reusing live cache' if reused else 'cache created'}: {stage23_cache_name} "
                  f"(ttl={STAGE23_CACHE_TTL_SECONDS}s)")

    # ── Interview: one whole-reference-video CachedContent, identical across every interview
    # item — reuses the exact cache query_gemini_annotations.py --interview already built (same
    # key format, so a hit here means no re-encode), since interview items need the same context
    # that generation step used. ──
    interview_cache_name = None
    if any(i["stage"] == "interview" for i in todo):
        if not interview_reference_video_name:
            sys.exit("interview_queries.json not found or has no 'video' field — needed for the "
                      "reference video (run query_gemini_annotations.py --interview first)")
        interview_video_path = dataset_dir / "videos" / interview_reference_video_name
        if not interview_video_path.exists():
            sys.exit(f"interview reference video not found: {interview_video_path}")
        interview_duration = _video_duration_seconds(interview_video_path)
        print(f"Shared interview context: whole video [0s-{interview_duration:.1f}s] at "
              f"{INTERVIEW_FPS}fps @{INTERVIEW_VIDEO_SCALE_WIDTH}px (cache)...")

        def build_interview_cache_contents():
            def extract_at(crf):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    clip_path = Path(tmp_dir) / "interview_reference_whole.mp4"
                    _extract_clip(interview_video_path, 0.0, interview_duration, clip_path, INTERVIEW_FPS, crf)
                    return clip_path.read_bytes()
            data, _ = vertex_cache.encode_under_size_limit(extract_at, INTERVIEW_CRF)
            print(f"  cut whole-reference-video clip ({len(data) / 1e6:.1f} MB)")
            return [video_part(data, INTERVIEW_FPS)]

        ist = interview_video_path.stat()
        interview_cache_key = (
            f"interview_whole|model={MODEL}|fps={INTERVIEW_FPS}|scale={INTERVIEW_VIDEO_SCALE_WIDTH}"
            f"|crf={INTERVIEW_CRF}|video={interview_video_path.name}:{ist.st_mtime_ns}:{ist.st_size}"
            f":{interview_duration:.1f}"
        )
        interview_cache_name, reused = vertex_cache.get_or_create(
            client, types, dataset_dir, interview_cache_key, MODEL, INTERVIEW_CACHE_TTL_SECONDS,
            build_interview_cache_contents,
        )
        print(f"  {'reusing live cache' if reused else 'cache created'}: {interview_cache_name} "
              f"(ttl={INTERVIEW_CACHE_TTL_SECONDS}s)")

    for i, item in enumerate(todo, 1):
        key = f"{item['annotation_id']}:{item['stage']}"
        if item["stage"] == "interview":
            iref = interview_queries_by_id.get(item["annotation_id"])
            if iref is None:
                print(f"[{i}/{len(todo)}] {key}: no matching interview_queries.json entry, skipping")
                continue
        else:
            ref = queries_by_id.get(item["annotation_id"])
            if ref is None:
                print(f"[{i}/{len(todo)}] {key}: no matching gemini_queries.json entry, skipping")
                continue

        print(f"[{i}/{len(todo)}] {key} ({item['category']!r})")
        video_window = None

        if item["stage"] == "stage1":
            # Per-item context, inline: this item's own ~30s clip (matching how the stage1
            # question was generated and how the judge is later shown it) + the GOAL text.
            vw = ref["video_window"]
            window_start = max(VIDEO_FLOOR_SECONDS, vw["start"])
            window_end = max(window_start + 1.0, vw["end"])
            video_window = {"start": window_start, "end": window_end}
            with tempfile.TemporaryDirectory() as tmp_dir:
                clip_path = Path(tmp_dir) / f"{item['annotation_id']}.mp4"
                _extract_clip(video_path, window_start, window_end, clip_path, STAGE1_FPS, STAGE1_CRF)
                clip_bytes = clip_path.read_bytes()
            prompt = STAGE1_PROMPT.format(
                question=item["question"], correct_answer=item["correct_answer"],
                correct_answer_word_count=len(item["correct_answer"].split()),
            )
            response, usage = call_gemini(
                prompt, context_parts=[video_part(clip_bytes, STAGE1_FPS), types.Part(text=goal_text)]
            )
        elif item["stage"] == "interview":
            # Video comes from the shared whole-reference-video cache; only the GOAL text is
            # sent inline alongside it (same combination pattern as stage2/3's cached calls).
            prompt = INTERVIEW_PROMPT.format(
                question=item["question"], correct_answer=item["correct_answer"],
                correct_answer_word_count=len(item["correct_answer"].split()),
                reference_start=iref["reference_start"], reference_end=iref["reference_end"],
            )
            response, usage = call_gemini(
                prompt, context_parts=[types.Part(text=goal_text)], cached_content=interview_cache_name
            )
        else:
            # Anything within MIN_GAP_SECONDS of the correct span is dropped from the candidate
            # list — not just the exact span — so the model can't return the answer it's just
            # been told is wrong, AND whichever candidate it does pick is already gap-compliant
            # by construction (matches build_vqa.py's MIN_GAP_SECONDS requirement).
            correct_span = parse_timespan(item["correct_answer"])
            if correct_span is not None:
                candidates = [s for s in all_candidate_spans
                              if gap_seconds(correct_span, parse_timespan(s)) >= MIN_GAP_SECONDS]
            else:
                candidates = [s for s in all_candidate_spans if s != item["correct_answer"]]
            prompt = STAGE23_PROMPT.format(
                question=item["question"], correct_answer=item["correct_answer"],
                candidate_timespans="\n".join(candidates),
            )
            if args.no_cache:
                response, usage = call_gemini(prompt, context_parts=[
                    video_part(stage23_clip_bytes, STAGE23_FPS, start_offset=stage23_start_offset),
                    types.Part(text=goal_text),
                ])
            else:
                response, usage = call_gemini(prompt, cached_content=stage23_cache_name)

        try:
            answer = parse_json_response(response)["answer"]
            if item["stage"] not in ("stage1", "interview"):
                # normalize to the bare "Xs-Ys" form the eval uses, and refuse an answer that
                # landed back on the correct span (it would make the item unanswerable)
                span = parse_timespan(answer)
                if span is None:
                    raise ValueError(f"answer not a timespan: {answer!r}")
                answer = to_timespan(*span)
                if correct_span is not None and gap_seconds(correct_span, span) < MIN_GAP_SECONDS:
                    raise ValueError(f"answer is within MIN_GAP_SECONDS of the correct span: {answer!r}")
            answers = [answer]  # a list of one, matching build_vqa.py's distractor shape
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  parse failure ({e}); storing raw response")
            answers = None

        output["results"].append({
            "key": key,
            "annotation_id": item["annotation_id"],
            "stage": item["stage"],
            "category": item["category"],
            "question": item["question"],
            "correct_answer": item["correct_answer"],
            "video_window": video_window,
            "adversarial_answers": answers,
            "raw_response": response if answers is None else None,
            "usage": usage,
        })
        save_output(output_file, output)  # incremental — safe to interrupt/resume

    print(f"Wrote {len(output['results'])} total results to {output_file}")


if __name__ == "__main__":
    main()

"""
clip_summaries.py — Per-clip inferred_states for an activity-segmented recording.

One Gemini call per activity clip (video only, no memory threaded between clips), asking for
just two fields: clip_summary and current_action. These are exactly the two fields
vqa_pipeline/query_gemini_annotations.py's load_segment_context() reads from
inferred_states.json (see that function and context_up_to() for how they're used), and the
ground-truth reconciliation step (ground_truth/README.md) uses as its visual evidence.

(Sessions 01-04 in the paper were processed with an earlier, heavier engine that additionally
produced intent/cognitive-state fields and threaded a memory across clips. Nothing downstream
reads those extra fields, so this lean engine — used for session05 — is the one released.)

Reads activity clips from inferred_states/sessions/{session}/clips/ (activity_N_Xs_Ys.mp4, from
cut_video.py) and writes
inferred_states/sessions/{session}/output/inferred_states/{run_id}/inferred_states.json as a list
of {timestamp, video, inferred_local_state: {clip_summary, current_action}}. Copy the finished
file to data/{session}/annotations/inferred_states.json for the downstream steps.

Usage:
    python3 inferred_states/clip_summaries.py --session session05
    python3 inferred_states/clip_summaries.py --session session05 --resume
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing_extensions import TypedDict

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
DOTENV_FILE = REPO_ROOT / ".env"

from dotenv import load_dotenv
load_dotenv(DOTENV_FILE)

# Vertex AI (same service-account setup as vqa_pipeline).
_creds_path = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
if not _creds_path.is_absolute():
    _creds_path = (DOTENV_FILE.parent / _creds_path).resolve()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_creds_path)
VERTEX_PROJECT = json.loads(_creds_path.read_text())["project_id"]
VERTEX_LOCATION = "global"

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

INFERENCE_MODEL = "gemini-2.5-flash-lite"
INFERENCE_TEMPERATURE = 0.2

client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)

CLIP_SUMMARY_PROMPT = """You are captioning one clip from a longer recording.

Return ONLY a JSON object with two fields:
- clip_summary: a factual, grounded description of this clip — what is on screen, what happens,
  what is said. Describe in detail, as though captioning. It is a "summary" only because it is
  less detailed than the full clip, not because it should be vague.
- current_action: as specific as the evidence allows (e.g. "Rephrasing the opening paragraph of
  the Methods section" rather than "Editing a document"). Only assert what is directly
  observable."""


class ClipSummarySchema(TypedDict):
    clip_summary: str
    current_action: str


def _is_retryable(exc: BaseException) -> bool:
    """Covers connection failures, 429/500/502/503/504, the intermittent empty-candidate response
    observed in scene_split/query_frames.py, and an intermittent truncated/malformed JSON body
    (finish_reason=STOP but the string cuts off mid-field) — both reproduced as one-off flakes
    that succeed cleanly on retry, not a real prompt/schema problem."""
    if isinstance(exc, (ConnectionError, TimeoutError, json.JSONDecodeError)):
        return True
    if "Empty response from Gemini" in str(exc):
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code in (408, 429, 500, 502, 503, 504)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _call_gemini(video_path: Path) -> dict:
    part = types.Part(inline_data=types.Blob(data=video_path.read_bytes(), mime_type="video/mp4"))
    response = client.models.generate_content(
        model=INFERENCE_MODEL,
        contents=[part, CLIP_SUMMARY_PROMPT],
        config=types.GenerateContentConfig(
            temperature=INFERENCE_TEMPERATURE,
            response_mime_type="application/json",
            response_schema=ClipSummarySchema,
        ),
    )
    if response.text is None:
        finish_reason = response.candidates[0].finish_reason if response.candidates else None
        raise RuntimeError(f"Empty response from Gemini (finish_reason={finish_reason})")
    return json.loads(response.text)


def _activity_number(p: Path) -> int:
    match = re.search(r"activity_(\d+)", p.stem)
    return int(match.group(1)) if match else 0


def _latest_output_run_dir(inferred_states_dir: Path) -> Path:
    runs = sorted(d for d in inferred_states_dir.iterdir() if d.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run folders found in {inferred_states_dir}")
    return runs[-1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="Session name (must exist under inferred_states/sessions/)")
    parser.add_argument("--run", default=None, help="Output run ID to write into / resume from")
    parser.add_argument("--resume", action="store_true", help="Resume, skipping clips already in the output file")
    args = parser.parse_args()

    session_dir = PIPELINE_DIR / "sessions" / args.session
    if not session_dir.exists():
        available = [d.name for d in (PIPELINE_DIR / "sessions").iterdir() if d.is_dir()]
        sys.exit(f"Error: session '{args.session}' not found under sessions/\nAvailable: {', '.join(sorted(available)) or 'none'}")

    clips_dir = session_dir / "clips"
    inferred_states_dir = session_dir / "output" / "inferred_states"

    if args.run:
        run_id = args.run
    elif args.resume:
        run_id = _latest_output_run_dir(inferred_states_dir).name
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = inferred_states_dir / run_id / "inferred_states.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    clips = sorted(clips_dir.glob("activity_*.mp4"), key=_activity_number)
    if not clips:
        sys.exit(f"No activity_*.mp4 clips found in {clips_dir}")

    timeline = []
    already_processed = set()
    if args.resume and output_file.exists():
        timeline = json.loads(output_file.read_text())
        already_processed = {entry["timestamp"] for entry in timeline}
        logger.info("Resuming: %d clips already done", len(timeline))

    remaining = [c for c in clips if c.stem not in already_processed]
    logger.info("Session: %s | Run: %s | %d clips, %d remaining", args.session, run_id, len(clips), len(remaining))

    for i, clip_path in enumerate(remaining, 1):
        timestamp = clip_path.stem
        logger.info("[%d/%d] %s", i, len(remaining), timestamp)

        local_state = _call_gemini(clip_path)
        timeline.append({
            "timestamp": timestamp,
            "video": clip_path.name,
            "inferred_local_state": local_state,
        })
        timeline.sort(key=lambda e: _activity_number(Path(e["timestamp"])))

        output_file.write_text(json.dumps(timeline, indent=2))

    logger.info("Done. Wrote %d entries to %s", len(timeline), output_file)


if __name__ == "__main__":
    main()

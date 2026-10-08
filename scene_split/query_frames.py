"""
query_frames.py — Send screenshots to Gemini for scene splitting or merge checking.

Subcommands:
  split  — Send 10 consecutive frames and ask Gemini to identify scene boundaries.
  merge  — Send the last frame of one interval + first frames of the next and ask
            whether they belong to the same activity.

Usage:
    python3 scene_split/query_frames.py split --folder session01 --start 1
    python3 scene_split/query_frames.py split --folder session01 --start 50

    python3 scene_split/query_frames.py merge --folder session01 --last 10 --next-start 11
    python3 scene_split/query_frames.py merge --folder session01 --last 10 --next-start 11 --n-next 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Vertex AI, not the Gemini Developer API — same service-account auth vqa_pipeline uses.
REPO_ROOT = Path(__file__).resolve().parent.parent
DOTENV_FILE = REPO_ROOT / ".env"
load_dotenv(DOTENV_FILE)

# GOOGLE_APPLICATION_CREDENTIALS in the .env is a relative path (key.json), which resolves
# against the process's cwd, not the .env file's location — fix it up to an absolute path so
# this works regardless of where we're run from (same fix query_gemini_annotations.py applies).
_creds_path = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
if not _creds_path.is_absolute():
    _creds_path = (DOTENV_FILE.parent / _creds_path).resolve()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_creds_path)
VERTEX_PROJECT = json.loads(_creds_path.read_text())["project_id"]
VERTEX_LOCATION = "global"

from google import genai
from google.genai import types

from prompt import split_prompt, merge_prompt

FRAMES_DIR = Path(__file__).resolve().parent / "frames"
MODEL = "gemini-2.5-flash"
N_FRAMES = 10
N_NEXT_DEFAULT = 1  # frames from the next interval to include in merge check

_client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_images(folder: str) -> list[Path]:
    folder_path = FRAMES_DIR / folder
    if not folder_path.exists():
        sys.exit(f"Error: frames folder not found: {folder_path}")
    images = sorted(
        p for p in folder_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        sys.exit(f"Error: no images found in {folder_path}")
    return images


def load_split_frames(folder: str, start: int) -> list[Path]:
    """Return up to N_FRAMES consecutive frames starting at `start` (1-indexed)."""
    images = _all_images(folder)
    idx = start - 1
    if idx < 0 or idx >= len(images):
        sys.exit(
            f"Error: --start {start} out of range "
            f"(folder has {len(images)} frames, valid range 1–{len(images)})"
        )
    selected = images[idx: idx + N_FRAMES]
    if len(selected) < N_FRAMES:
        print(
            f"Warning: only {len(selected)} frames available starting at {start} "
            f"(requested {N_FRAMES}). Proceeding with {len(selected)} frames."
        )
    return selected


def load_merge_frames(folder: str, last: int, next_start: int, n_next: int) -> tuple[list[Path], list[Path]]:
    """Return (last_frames, next_frames) for a merge check.

    last_frames  — the single frame at index `last` (1-indexed), i.e. the final
                   frame of the previous interval.
    next_frames  — `n_next` frames starting at `next_start` (1-indexed), i.e. the
                   opening frames of the new interval.
    """
    images = _all_images(folder)
    n = len(images)

    def _get(idx_1based: int, label: str) -> Path:
        i = idx_1based - 1
        if i < 0 or i >= n:
            sys.exit(f"Error: --{label} {idx_1based} out of range (folder has {n} frames)")
        return images[i]

    last_frame = _get(last, "last")

    next_idx = next_start - 1
    if next_idx < 0 or next_idx >= n:
        sys.exit(f"Error: --next-start {next_start} out of range (folder has {n} frames)")
    next_frames = images[next_idx: next_idx + n_next]
    if len(next_frames) < n_next:
        print(
            f"Warning: only {len(next_frames)} frames available from next-start {next_start}."
        )
    return [last_frame], next_frames


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------

def load_transcript_slice(folder: str, start_frame: int, end_frame: int) -> str | None:
    """Return transcript text covering the time window for frames [start_frame, end_frame].

    Returns None if no transcript exists for the folder.
    """
    folder_path = FRAMES_DIR / folder
    transcript_path = folder_path / "transcript.json"
    if not transcript_path.exists():
        return None

    metadata_path = folder_path / "metadata.json"
    fps = 1.0
    if metadata_path.exists():
        fps = json.loads(metadata_path.read_text()).get("fps", 1.0)

    t_start = (start_frame - 1) / fps
    t_end = end_frame / fps

    segments = json.loads(transcript_path.read_text())
    words = []
    for seg in segments:
        for word in seg.get("words", []):
            if word["start"] < t_end and word["end"] > t_start:
                words.append(word["word"])

    text = " ".join(words).strip()
    return text if text else None


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

def query_gemini(frames: list[Path], prompt: str, transcript: str | None = None) -> str:
    parts: list = []
    for frame in frames:
        image_bytes = frame.read_bytes()
        suffix = frame.suffix.lower().lstrip(".")
        mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
    if transcript:
        prompt = prompt + f"\n\n## Audio Transcript\n{transcript}"
    parts.append(types.Part.from_text(text=prompt))
    response = _client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(temperature=0.2),
    )
    if response.text is None:
        # Observed intermittently (not reproducible on retry) — an empty candidate/parts list
        # with no text. Raise rather than let callers (e.g. pipeline.py's parse_merge_answer)
        # crash on None with a confusing TypeError; the message is matched by
        # query_gemini_with_retry's is_transient check so this gets retried automatically.
        finish_reason = response.candidates[0].finish_reason if response.candidates else None
        raise RuntimeError(f"Empty response from Gemini (finish_reason={finish_reason})")
    return response.text


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_split(args: argparse.Namespace) -> None:
    frames = load_split_frames(args.folder, args.start)
    print(f"Mode   : split")
    print(f"Folder : {args.folder}")
    print(f"Frames : {[f.name for f in frames]}")
    print(f"Model  : {MODEL}")
    print("-" * 60)
    print(query_gemini(frames, split_prompt))


def cmd_merge(args: argparse.Namespace) -> None:
    last_frames, next_frames = load_merge_frames(
        args.folder, args.last, args.next_start, args.n_next
    )
    # Gemini sees: [last frame of prev interval, first N frames of next interval]
    # The merge_prompt instructs: "if the clip starting from the second frame is
    # a consistent activity with the first frame"
    all_frames = last_frames + next_frames
    print(f"Mode        : merge")
    print(f"Folder      : {args.folder}")
    print(f"Last frame  : {last_frames[0].name}  (end of previous interval)")
    print(f"Next frames : {[f.name for f in next_frames]}  (start of new interval)")
    print(f"Model       : {MODEL}")
    print("-" * 60)
    print(query_gemini(all_frames, merge_prompt))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query Gemini with screenshots for scene splitting or merge checking."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- split subcommand --
    p_split = sub.add_parser("split", help="Identify scene boundaries in 10 consecutive frames.")
    p_split.add_argument("--folder", required=True, help="Subfolder inside scene_split/frames/")
    p_split.add_argument("--start", type=int, default=1, help="1-indexed starting frame (default: 1)")

    # -- merge subcommand --
    p_merge = sub.add_parser(
        "merge",
        help="Check if the last frame of one interval and the start of the next are the same activity.",
    )
    p_merge.add_argument("--folder", required=True, help="Subfolder inside scene_split/frames/")
    p_merge.add_argument(
        "--last", type=int, required=True,
        help="1-indexed frame number of the last frame of the previous interval",
    )
    p_merge.add_argument(
        "--next-start", type=int, required=True, dest="next_start",
        help="1-indexed frame number of the first frame of the next interval",
    )
    p_merge.add_argument(
        "--n-next", type=int, default=N_NEXT_DEFAULT, dest="n_next",
        help=f"How many frames to take from the next interval (default: {N_NEXT_DEFAULT})",
    )

    args = parser.parse_args()
    if args.command == "split":
        cmd_split(args)
    else:
        cmd_merge(args)


if __name__ == "__main__":
    main()

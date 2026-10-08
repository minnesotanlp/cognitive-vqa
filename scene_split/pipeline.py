"""
pipeline.py — Full scene-segmentation pipeline.

Interleaved split + merge:
    1. Split chunk 0
    2. Split chunk 1  →  merge junction(chunk 0 | chunk 1)
    3. Split chunk 2  →  merge junction(chunk 1 | chunk 2)
    ...

Progress is saved after every split and every merge so a crashed run can resume.
A checkpoint file (.checkpoint.json) tracks which chunks/junctions are done.

Usage:
    python3 scene_split/pipeline.py --folder session01
    python3 scene_split/pipeline.py --folder session01 --resume
    python3 scene_split/pipeline.py --folder session01 --verbose
    python3 scene_split/pipeline.py --folder session01 --output results.json
"""

import argparse
import json
import re
import time
import sys
from pathlib import Path

# dotenv/Vertex client setup lives in query_frames.py (imported below) — no separate load here.
from query_frames import _all_images, query_gemini, load_transcript_slice, N_FRAMES, FRAMES_DIR
from prompt import split_prompt, merge_prompt


# ---------------------------------------------------------------------------
# Retry wrapper for 503 / transient errors
# ---------------------------------------------------------------------------

def query_gemini_with_retry(frames, prompt, transcript: str | None = None, retries: int = 8, base_wait: float = 10.0) -> str:
    """Call query_gemini with exponential backoff on 429/503/transient errors."""
    for attempt in range(retries):
        try:
            return query_gemini(frames, prompt, transcript=transcript)
        except Exception as e:
            msg = str(e)
            is_transient = ("503" in msg or "UNAVAILABLE" in msg or "502" in msg or "500" in msg
                             or "429" in msg or "RESOURCE_EXHAUSTED" in msg
                             or "Empty response from Gemini" in msg)
            if is_transient and attempt < retries - 1:
                wait = base_wait * (2 ** attempt)
                print(f"  [retry {attempt + 1}/{retries - 1}] Gemini unavailable, waiting {wait:.0f}s... ({msg[:60]})")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def parse_split_boundaries(response: str) -> list[int]:
    match = re.search(r"\[3\.\s*Frames\]\s*:\s*(\[.*?\])", response, re.IGNORECASE)
    if not match:
        return []
    try:
        raw = json.loads(match.group(1))
        return [int(x) for x in raw]
    except (json.JSONDecodeError, ValueError):
        return []


def parse_merge_answer(response: str) -> bool:
    match = re.search(r"\b(yes|no)\b", response, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "yes"
    return False


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _segments_dir() -> Path:
    d = Path(__file__).parent / "segments"
    d.mkdir(exist_ok=True)
    return d

def _checkpoint_path(folder: str) -> Path:
    return _segments_dir() / f"{folder}.checkpoint.json"

def _segments_path(folder: str) -> Path:
    return _segments_dir() / f"{folder}_segments.json"

def load_checkpoint(folder: str) -> dict:
    p = _checkpoint_path(folder)
    if p.exists():
        return json.loads(p.read_text())
    return {"segment_starts": [1], "split_done_chunks": [], "merge_done_junctions": [], "n": None}

def save_checkpoint(folder: str, checkpoint: dict) -> None:
    _checkpoint_path(folder).write_text(json.dumps(checkpoint, indent=2))

def save_segments_incremental(folder: str, segment_starts: set[int], n: int, out_path: Path) -> None:
    sorted_starts = sorted(segment_starts)
    segments = []
    for i, start in enumerate(sorted_starts):
        end = sorted_starts[i + 1] - 1 if i + 1 < len(sorted_starts) else n
        segments.append((start, end))
    out_path.write_text(
        json.dumps(
            [{"activity": i + 1, "start": s, "end": e} for i, (s, e) in enumerate(segments)],
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(folder: str, verbose: bool = False, resume: bool = False) -> list[tuple[int, int]]:
    """Run the interleaved split → merge pipeline and return activity segments."""
    images = _all_images(folder)
    n = len(images)
    out_path = _segments_path(folder)

    if resume:
        checkpoint = load_checkpoint(folder)
        if checkpoint["n"] is not None and checkpoint["n"] != n:
            print(f"Warning: checkpoint has n={checkpoint['n']} but folder has {n} frames. Ignoring checkpoint.")
            checkpoint = {"segment_starts": [1], "split_done_chunks": [], "merge_done_junctions": [], "n": n}
        else:
            checkpoint["n"] = n
        segment_starts: set[int] = set(checkpoint["segment_starts"])
        split_done: set[int] = set(checkpoint["split_done_chunks"])
        merge_done: set[int] = set(checkpoint["merge_done_junctions"])
        print(f"Resuming: {len(split_done)} chunks split, {len(merge_done)} junctions merged.")
    else:
        checkpoint = {"segment_starts": [1], "split_done_chunks": [], "merge_done_junctions": [], "n": n}
        segment_starts = {1}
        split_done = set()
        merge_done = set()

    has_transcript = (FRAMES_DIR / folder / "transcript.json").exists()
    if has_transcript:
        print("  [transcript found — will include audio context in prompts]")

    n_chunks = (n + N_FRAMES - 1) // N_FRAMES
    print(f"\n{'='*60}")
    print(f"INTERLEAVED SPLIT+MERGE  ({n} frames, {n_chunks} chunks of {N_FRAMES})")
    print(f"{'='*60}")

    for chunk_idx, chunk_start_0 in enumerate(range(0, n, N_FRAMES)):
        chunk_end_0 = min(chunk_start_0 + N_FRAMES, n)
        chunk_label = f"frames {chunk_start_0 + 1}–{chunk_end_0}"

        # ── Split this chunk ────────────────────────────────────────────────
        if chunk_start_0 in split_done:
            print(f"\n[Chunk {chunk_idx + 1}/{n_chunks}] {chunk_label}  (split skipped)")
        else:
            chunk_frames = images[chunk_start_0:chunk_end_0]

            if chunk_start_0 > 0:
                segment_starts.add(chunk_start_0 + 1)

            print(f"\n[Chunk {chunk_idx + 1}/{n_chunks}] {chunk_label}  SPLIT")
            transcript = load_transcript_slice(folder, chunk_start_0 + 1, chunk_end_0) if has_transcript else None
            response = query_gemini_with_retry(chunk_frames, split_prompt, transcript=transcript)

            if verbose:
                print(response)

            local_bounds = [k for k in parse_split_boundaries(response) if k > 1]
            global_bounds = [chunk_start_0 + k for k in local_bounds]
            for gb in global_bounds:
                segment_starts.add(gb)

            if local_bounds:
                print(f"  Local boundaries : {local_bounds}  →  global frames {global_bounds}")
            else:
                print(f"  Single activity, no internal boundaries.")

            split_done.add(chunk_start_0)
            checkpoint["segment_starts"] = sorted(segment_starts)
            checkpoint["split_done_chunks"] = sorted(split_done)
            save_checkpoint(folder, checkpoint)
            save_segments_incremental(folder, segment_starts, n, out_path)
            print(f"  [checkpoint saved]")

        # ── Merge the junction between the previous chunk and this one ───────
        # Junction exists only from chunk 1 onwards (need a previous chunk)
        if chunk_start_0 == 0:
            continue

        junction = chunk_start_0  # 0-indexed start of this chunk = junction point
        last_1 = chunk_start_0       # last frame (1-indexed) of the previous chunk
        next_1 = chunk_start_0 + 1   # first frame (1-indexed) of this chunk

        if junction in merge_done:
            print(f"  [Junction] frame {last_1} → {next_1}  (merge skipped)")
        else:
            print(f"  [Junction] frame {last_1} → {next_1}  MERGE")
            merge_frames = [images[last_1 - 1], images[next_1 - 1]]
            print(f"    Images sent: {[f.name for f in merge_frames]}")

            transcript = load_transcript_slice(folder, last_1, next_1) if has_transcript else None
            response = query_gemini_with_retry(merge_frames, merge_prompt, transcript=transcript)

            if verbose:
                print(response)

            same = parse_merge_answer(response)
            print(f"    Same activity? {response.strip()[:80]!r}  →  {'MERGE' if same else 'KEEP boundary'}")

            if same:
                segment_starts.discard(next_1)

            merge_done.add(junction)
            checkpoint["segment_starts"] = sorted(segment_starts)
            checkpoint["merge_done_junctions"] = sorted(merge_done)
            save_checkpoint(folder, checkpoint)
            save_segments_incremental(folder, segment_starts, n, out_path)
            print(f"  [checkpoint saved]")

    # ── Build final segments ─────────────────────────────────────────────────
    sorted_starts = sorted(segment_starts)
    segments: list[tuple[int, int]] = []
    for i, start in enumerate(sorted_starts):
        end = sorted_starts[i + 1] - 1 if i + 1 < len(sorted_starts) else n
        segments.append((start, end))

    return segments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the interleaved scene-split + merge pipeline on a frames folder."
    )
    parser.add_argument("--folder", required=True, help="Subfolder inside scene_split/frames/")
    parser.add_argument("--verbose", action="store_true", help="Print full Gemini responses")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint, skipping already-processed chunks/junctions")
    parser.add_argument("--output", default=None, help="Optional path to save results as JSON")
    args = parser.parse_args()

    segments = run_pipeline(args.folder, verbose=args.verbose, resume=args.resume)

    print(f"\n{'='*60}")
    print(f"FINAL SEGMENTS  ({len(segments)} activities)")
    print(f"{'='*60}")
    for i, (start, end) in enumerate(segments, 1):
        print(f"  Activity {i:>3}: frames {start}–{end}  ({end - start + 1} frames)")

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = _segments_path(args.folder)
    out_path.write_text(
        json.dumps(
            [{"activity": i + 1, "start": s, "end": e} for i, (s, e) in enumerate(segments)],
            indent=2,
        )
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

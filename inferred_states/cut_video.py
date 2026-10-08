"""
cut_video.py — Split all annotation videos into activity clips using scene segments.

Reads:
  data/raw_videos/ (or --videos-dir)        — source video files
  scene_split/segments/{stem}_segments.json — scene segment boundaries per video
  scene_split/frames/{stem}/metadata.json  — fps the frames were extracted at (extract_frames.py)

segments.json uses 1-indexed frame numbers captured at the frame-extraction fps recorded in
metadata.json (defaults to 1.0, i.e. frame N = second N, if that file is missing — e.g. frames
extracted before extract_frames.py started writing it). For a segment with start=S, end=E: video
time span is (S-1)/fps seconds to E/fps seconds.

Output: inferred_states/sessions/{stem}/clips/activity_N_Xs_Ys.mp4

Usage:
    python3 inferred_states/cut_video.py
    python3 inferred_states/cut_video.py --videos session01.mp4
"""

import argparse
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

PIPELINE_DIR     = Path(__file__).resolve().parent
REPO_ROOT        = PIPELINE_DIR.parent
VIDEOS_DIR       = REPO_ROOT / "data" / "raw_videos"
SEGMENTS_DIR     = REPO_ROOT / "scene_split" / "segments"
FRAMES_DIR       = REPO_ROOT / "scene_split" / "frames"
VIDEO_EXTS       = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def load_extraction_fps(stem: str) -> float:
    """fps the frames for this video were extracted at, per scene_split/frames/{stem}/metadata.json
    (written by extract_frames.py). segments.json's frame numbers are 1-indexed at THIS rate, not
    necessarily 1fps. Defaults to 1.0 (prior behavior) if no metadata.json is found."""
    meta_path = FRAMES_DIR / stem / "metadata.json"
    if not meta_path.exists():
        return 1.0
    return json.loads(meta_path.read_text()).get("fps", 1.0)


def load_segments(segments_path: Path) -> list[dict]:
    if not segments_path.exists():
        logger.error("Segments file not found: %s", segments_path)
        sys.exit(1)
    with open(segments_path) as f:
        return json.load(f)


def get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip())


def _cut_clip(video_path: Path, output_path: Path, activity: int, start_sec: float, end_sec: float, overwrite: bool) -> str:
    if output_path.exists() and not overwrite:
        logger.info("  activity %d: already exists, skipping.", activity)
        return str(output_path)

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-c:a", "aac",
         str(output_path)],
        capture_output=True, check=True
    )
    logger.info("  activity %d: %.1fs–%.1fs → %s", activity, start_sec, end_sec, output_path)
    return str(output_path)


def split_video(video_path: Path, output_dir: Path, segments: list[dict], fps: float = 1.0,
                 overwrite: bool = False) -> list[str]:
    """Cut video into clips in parallel using scene segment boundaries.

    Each segment {activity, start, end} uses 1-indexed frame numbers at `fps` (the rate frames
    were extracted at — see load_extraction_fps):
      clip start = (start - 1) / fps seconds
      clip end   = end / fps seconds
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = get_video_duration(video_path)

    jobs = []
    for seg in segments:
        start_sec = (seg["start"] - 1) / fps
        end_sec   = seg["end"] / fps
        if start_sec >= duration:
            logger.warning("  activity %d: start %.1fs is past video end (%.1fs) — skipping", seg["activity"], start_sec, duration)
            continue
        if end_sec > duration:
            logger.warning("  activity %d: end %.1fs exceeds video duration (%.1fs) — clamping", seg["activity"], end_sec, duration)
            end_sec = duration
        jobs.append((seg["activity"], start_sec, end_sec,
                     output_dir / f"activity_{seg['activity']}_{start_sec:g}s_{end_sec:g}s.mp4"))

    clips = [None] * len(jobs)
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_cut_clip, video_path, output_path, activity, start_sec, end_sec, overwrite): i
            for i, (activity, start_sec, end_sec, output_path) in enumerate(jobs)
        }
        for future in as_completed(futures):
            clips[futures[future]] = future.result()

    return clips


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", default=None, metavar="VIDEO",
                        help="One or more video filenames to process (e.g. --videos session01.mp4). Default: all.")
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR,
                        help="Directory of source videos (default: data/raw_videos).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-cut and overwrite clips that already exist.")
    parser.add_argument("--workers", type=int, default=1, metavar="N",
                        help="Number of videos to process in parallel (default: 1).")
    args = parser.parse_args()

    downloaded   = args.videos_dir
    all_videos   = [{"name": p.name} for p in sorted(downloaded.iterdir()) if p.suffix.lower() in VIDEO_EXTS]

    if args.videos:
        names = set(args.videos)
        all_videos = [v for v in all_videos if v["name"] in names]
        missing = names - {v["name"] for v in all_videos}
        if missing:
            logger.error("Not found in %s: %s", downloaded, ", ".join(missing))
            sys.exit(1)

    def process_video(entry):
        name       = entry["name"]
        stem       = Path(name).stem
        video_path = downloaded / name
        seg_path   = SEGMENTS_DIR / f"{stem}_segments.json"
        output_dir = PIPELINE_DIR / "sessions" / stem / "clips"

        if not video_path.exists():
            logger.warning("Skipping '%s' — video not found: %s", name, video_path)
            return
        if not seg_path.exists():
            logger.warning("Skipping '%s' — segments not found: %s", name, seg_path)
            return

        segments = load_segments(seg_path)
        fps = load_extraction_fps(stem)
        logger.info("Processing: %s  |  %d segments @ %.2f fps → %s", name, len(segments), fps, output_dir)
        clips = split_video(video_path, output_dir, segments, fps=fps, overwrite=args.overwrite)
        logger.info("Done. %d clips created.\n", len(clips))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_video, entry) for entry in all_videos]
        for future in as_completed(futures):
            future.result()

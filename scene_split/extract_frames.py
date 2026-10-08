"""
extract_frames.py — Extract frames at 1 fps from every video in
data/raw_videos/ (or --videos-dir) and save them to scene_split/frames/<stem>/.

Frame files are named frame_0001.jpg, frame_0002.jpg, ... (4-digit, 1-indexed)
as expected by pipeline.py and query_frames.py.

Usage:
    python3 scene_split/extract_frames.py
    python3 scene_split/extract_frames.py --fps 2
    python3 scene_split/extract_frames.py --overwrite
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
VIDEOS_DIR  = REPO_ROOT / "data" / "raw_videos"
FRAMES_DIR  = Path(__file__).resolve().parent / "frames"
VIDEO_EXTS  = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def extract(video_path: Path, out_dir: Path, fps: float, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("frame_*.jpg"))
    if existing and not overwrite:
        print(f"  Skipping — {len(existing)} frames already exist (use --overwrite to redo).")
    else:
        pattern = str(out_dir / "frame_%04d.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",          # JPEG quality (2 = near-lossless, 31 = worst)
            pattern,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR:\n{result.stderr[-800:]}")
            sys.exit(1)
        count = len(sorted(out_dir.glob("frame_*.jpg")))
        print(f"  Extracted {count} frames → {out_dir}")

    # Save fps metadata so downstream tools can map frame numbers to timestamps
    meta_path = out_dir / "metadata.json"
    if not meta_path.exists() or overwrite:
        meta_path.write_text(json.dumps({"fps": fps}, indent=2))

    # Extract audio (skip if already present)
    audio_path = out_dir / "audio.mp3"
    if audio_path.exists() and not overwrite:
        print(f"  Audio already exists, skipping (use --overwrite to redo).")
    else:
        audio_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",              # no video
            "-q:a", "2",        # VBR quality
            str(audio_path),
        ]
        audio_result = subprocess.run(audio_cmd, capture_output=True, text=True)
        if audio_result.returncode != 0:
            print(f"  Audio extraction ERROR:\n{audio_result.stderr[-400:]}")
        else:
            print(f"  Extracted audio → {audio_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from downloaded videos.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract (default: 1)")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if frames already exist")
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR,
                        help=f"Directory of source videos (default: {VIDEOS_DIR.relative_to(REPO_ROOT)})")
    args = parser.parse_args()
    videos_dir = args.videos_dir

    videos = sorted(p for p in videos_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"No videos found in {videos_dir}")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) in {videos_dir}\n")
    for video in videos:
        out_dir = FRAMES_DIR / video.stem
        print(f"[{video.name}]  →  frames/{video.stem}/")
        extract(video, out_dir, args.fps, args.overwrite)

    print("\nDone.")


if __name__ == "__main__":
    main()

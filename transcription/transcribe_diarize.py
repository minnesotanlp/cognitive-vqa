"""
transcribe_diarize.py — Transcribe + diarize a video's audio with WhisperX.

Extracts audio from a video, transcribes it (faster-whisper), aligns
word-level timestamps, and assigns speaker labels via pyannote diarization.
Writes a segment-level transcript JSON matching the schema used elsewhere in
this repo — a list of {id, start, end, text, speaker, speaker_label} — i.e.
data/<session>/transcripts/<video-stem>_transcript.json, which the label
annotations and vqa_pipeline read.

Requires: pip install whisperx
Diarization needs a HuggingFace token that has accepted the terms for
pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0 on
huggingface.co. Either run `huggingface-cli login` once (token is then
picked up automatically) or pass --hf-token.

Usage:
    python3 transcription/transcribe_diarize.py \
        --video /path/to/recording.mp4 \
        --out-dir data/session04

    python3 transcription/transcribe_diarize.py --video video.mp4 --out-dir DATASET \
        --model medium --min-speakers 2 --max-speakers 4
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def extract_audio(video_path: Path, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path),
        ],
        check=True, capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe + diarize a video's audio with WhisperX.")
    parser.add_argument("--video", required=True, type=Path, help="Path to the source video")
    parser.add_argument("--out-dir", required=True, type=Path,
                         help="Dataset root to write into, e.g. data/session04 "
                              "(video is copied to <out-dir>/videos/, transcript to <out-dir>/transcripts/)")
    parser.add_argument("--model", default="medium", help="Whisper model size (default: medium)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    parser.add_argument("--compute-type", default="int8", help="Faster-whisper compute type (default: int8)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--language", default=None, help="Force a language code (default: auto-detect)")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--hf-token", default=None,
                         help="HuggingFace token for pyannote diarization models "
                              "(default: cached `huggingface-cli login` token)")
    parser.add_argument("--no-copy-video", action="store_true",
                         help="Don't copy the video into <out-dir>/videos/ (transcript only)")
    parser.add_argument("--no-diarize", action="store_true",
                         help="Skip speaker diarization — transcribe + align only, with "
                              "speaker/speaker_label left null (matches the *_no_speaker.json convention)")
    parser.add_argument("--keep-audio", action="store_true", help="Keep the extracted 16kHz wav instead of deleting it")
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    out_dir = args.out_dir.expanduser().resolve()
    transcripts_dir = out_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_copy_video:
        videos_dir = out_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        dest_video = videos_dir / video_path.name
        if dest_video.exists():
            print(f"Video already present at {dest_video}, skipping copy.")
        else:
            print(f"Copying video to {dest_video} ...")
            shutil.copy2(video_path, dest_video)

    try:
        import whisperx
        from whisperx.asr import load_model
        from whisperx.alignment import load_align_model, align
        if not args.no_diarize:
            from whisperx.diarize import DiarizationPipeline, assign_word_speakers
    except ImportError:
        sys.exit("whisperx is not installed. Run: pip install whisperx")

    audio_path = out_dir / f".{video_path.stem}_16k.wav"
    print("Extracting audio...")
    extract_audio(video_path, audio_path)

    try:
        print(f"Loading Whisper model '{args.model}' on {args.device} ({args.compute_type})...")
        model = load_model(args.model, args.device, compute_type=args.compute_type, language=args.language)

        audio = whisperx.load_audio(str(audio_path))

        print("Transcribing...")
        result = model.transcribe(audio, batch_size=args.batch_size)
        print(f"  Detected language: {result['language']}")

        print("Aligning word timestamps...")
        align_model, metadata = load_align_model(language_code=result["language"], device=args.device)
        result = align(result["segments"], align_model, metadata, audio, args.device, return_char_alignments=False)

        if not args.no_diarize:
            print("Diarizing speakers...")
            diarize_model = DiarizationPipeline(token=args.hf_token, device=args.device)
            diarize_segments = diarize_model(
                audio, min_speakers=args.min_speakers, max_speakers=args.max_speakers
            )
            result = assign_word_speakers(diarize_segments, result)
    finally:
        if not args.keep_audio:
            audio_path.unlink(missing_ok=True)

    segments = []
    for i, seg in enumerate(result["segments"]):
        speaker = seg.get("speaker", None if args.no_diarize else "UNKNOWN")
        segments.append({
            "id": i,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "speaker": speaker,
            "speaker_label": speaker,
        })

    out_path = transcripts_dir / f"{video_path.stem}_transcript.json"
    out_path.write_text(json.dumps(segments, indent=2))
    print(f"\nSaved {len(segments)} segments -> {out_path}")


if __name__ == "__main__":
    main()

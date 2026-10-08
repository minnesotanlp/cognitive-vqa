# cognitive-vqa

Code and questions for a video multiple-choice QA benchmark that tests whether a model can follow
the *reasoning* of an experimenter at work — why they do what they do, what they are preparing for,
and when they draw an assessment — from video of real laboratory sessions.

This repository contains:

| Step | Directory | What it produces |
|---|---|---|
| 1. Transcription | `transcription/` | speaker-diarized transcript per session |
| 2. Scene splitting | `scene_split/` | activity segments (frame ranges) per video |
| 3. Per-clip inferred states | `inferred_states/` | `inferred_states.json` (clip summary + current action per segment) |
| 4. Ground-truth timeline | `ground_truth/` | `ground_truth_actions.json` (methodology, prompt, validator) |
| 5. VQA generation | `vqa_pipeline/` | `vqa.json` (questions + distractors) |
| 6. VQA evaluation & scoring | `vqa_pipeline/` | `eval_results*.json`, per-stage accuracy |
| Released questions | `data/session0{1..5}/` | blind-filtered `vqa.json` + `item_metadata.json` |

**Not yet released:** the session videos, transcripts, human span annotations
(`label_annotations.json`, `interview_annotations.json`), `inferred_states.json` and
`ground_truth_actions.json`. These contain identifiable participants and will be released upon
publication. Until then, the code is runnable end-to-end on your own recordings, and the released
questions can be evaluated in the **blind** setting (question + choices only); see
[Evaluating the released questions](#evaluating-the-released-questions).

The released `vqa.json` files contain only the **blind-filtered (hard) subset** — 574 of the 1,251
generated items; see [`data/README.md`](data/README.md).

## Setup

```
pip install -r requirements.txt
brew install ffmpeg            # or your platform's package; ffmpeg + ffprobe must be on PATH
cp .env.example .env           # then point GOOGLE_APPLICATION_CREDENTIALS at a Vertex AI service-account key
```

All model calls go through **Vertex AI** (Gemini) using a service-account key; the GCP project is
read from the key's `project_id`. A relative key path in `.env` is resolved against the repo root.
Run every command from the repo root.

Session data lives in `data/<session>/` by default; set `COGVQA_DATA_DIR` to use another location.

## Data layout

```
data/<session>/
  vqa.json                       # released: the questions
  item_metadata.json             # released: per-item timing + video duration (see data/README.md)
  videos/<session>.mp4           # on publication
  transcripts/<name>_transcript.json
  goal/                          # optional reference material describing the experiment's goal
  annotations/
    label_annotations.json       # human span labels over the transcript (GOAL/PLAN/ACTION/...)
    interview_annotations.json   # highlighted spans of a post-hoc interview with the experimenter
    inferred_states.json         # step 3 output
    ground_truth_actions.json    # step 4 output
  gemini_queries.json            # step 5 intermediates
  interview_queries.json
  adversarial_items.json
  adversarial_items_grounded.json
  eval_results*.json             # step 6 output
```

## Pipeline

### 1. Transcription

```
python3 transcription/transcribe_diarize.py --video /path/to/recording.mp4 --out-dir data/session01
```

WhisperX (faster-whisper + alignment + pyannote diarization). Diarization needs a HuggingFace token
that has accepted the pyannote model terms (`huggingface-cli login`, or `--hf-token`).
`pip install whisperx` separately — it is not in `requirements.txt`.

The human span annotations (`label_annotations.json`, `interview_annotations.json`) were then made
over these transcripts by annotators; the annotation tool is not part of this release.

### 2. Scene splitting

```
python3 scene_split/extract_frames.py --videos-dir data/raw_videos   # 1 fps frames -> scene_split/frames/<stem>/
python3 scene_split/pipeline.py --folder <stem>                       # -> scene_split/segments/<stem>_segments.json
python3 scene_split/pipeline.py --folder <stem> --resume              # resume after a crash
```

Chunks of 10 frames are sent to `gemini-2.5-flash` with a *split* prompt (find activity boundaries
inside the chunk); each junction between chunks is then checked with a *merge* prompt (same
activity across the junction?). Prompts are in `scene_split/prompt.py`. `scene_split/query_frames.py`
runs a single split/merge call for debugging. Details: [`ground_truth/README.md`](ground_truth/README.md#phase-1--how-inferred_statesjson-is-made).

### 3. Per-clip inferred states

```
python3 inferred_states/cut_video.py --videos-dir data/raw_videos     # -> inferred_states/sessions/<stem>/clips/
python3 inferred_states/clip_summaries.py --session <stem>            # -> inferred_states/sessions/<stem>/output/inferred_states/<run>/inferred_states.json
cp inferred_states/sessions/<stem>/output/inferred_states/<run>/inferred_states.json data/<session>/annotations/
```

### 4. Ground-truth action timeline

An LLM reconciliation of `inferred_states.json` (vision, MLLM-guessed) against the transcript
(speech, which often narrates ahead of the action). See
[`ground_truth/README.md`](ground_truth/README.md) for the methodology and
[`ground_truth/reconciliation_prompt.md`](ground_truth/reconciliation_prompt.md) for the exact
prompt; check the result with `python3 ground_truth/validate_ground_truth.py <file>`.

### 5. VQA generation

Run in this order, all with the same `--dataset`:

```
python3 vqa_pipeline/query_gemini_annotations.py --dataset session01
python3 vqa_pipeline/query_gemini_annotations.py --dataset session01 --interview
python3 vqa_pipeline/build_vqa.py --dataset session01
python3 vqa_pipeline/generate_adversarial.py --dataset session01
python3 vqa_pipeline/generate_adversarial_grounded.py --dataset session01
python3 vqa_pipeline/evaluate_vqa.py --dataset session01 --only-video   # see note below
python3 vqa_pipeline/build_vqa.py --dataset session01
```

1. **`query_gemini_annotations.py`** — for each relevant span in `label_annotations.json`, asks
   Gemini (`gemini-3.1-pro-preview`) to write a question + correct answer:
   - **stage1** — "why" questions about the experimenter's intent (PLAN / ASSESSMENT /
     INTENT_REASONING / RISK_MANAGEMENT spans), with a ~30 s clip around the span;
   - **stage2** — planning-dependency questions (PLAN spans): *when* did the experimenter prepare for
     a later event; answers are timestamps;
   - **stage3** — assessment-localization questions (ASSESSMENT spans): *when* did the experimenter
     draw a given inference; answers are timestamps; the model may decline unsuitable moments.

   Writes `gemini_queries.json`.
2. **`--interview`** — one "why" question per highlighted span of the post-hoc interview
   (`interview_annotations.json`), grounded in the whole experiment video. Writes
   `interview_queries.json`. Skip if the session has no interview.
3. **`build_vqa.py`** (1st pass) — assembles `vqa.json` with correct answers only.
4. **`generate_adversarial.py`** — two *blind* (no video) incorrect answers per question →
   `adversarial_items.json`.
5. **`generate_adversarial_grounded.py`** — one *video-grounded* incorrect answer per question →
   `adversarial_items_grounded.json`.
6. **`build_vqa.py`** (2nd pass) — merges one blind + one grounded distractor into each item. Of the
   two blind candidates it keeps the one the judge picked over the correct answer in
   `eval_results_only_video.json`, if that file exists; otherwise the first. Exactly reproducing the
   released `vqa.json` therefore also requires that intermediate only-video run.

All generation scripts are incremental/resumable and support `--limit N` and `--stage` for small
test runs. Whole-video contexts are cached in Vertex `CachedContent` (`vqa_pipeline/vertex_cache.py`)
and reused across runs; the registry is `data/<session>/.vertex_cache_registry.json` (6 h TTL).

### 6. Evaluation and scoring

```
python3 vqa_pipeline/evaluate_vqa.py --dataset session01 --no-video --no-text                         # blind
python3 vqa_pipeline/evaluate_vqa.py --dataset session01 --only-video                                 # video only
python3 vqa_pipeline/evaluate_vqa.py --dataset session01 --raw-transcript --no-ground-truth --no-timestamps   # full context
python3 vqa_pipeline/evaluate_vqa.py --dataset session01 --model <vertex-model-id> ...                 # other judge models
```

Answer choices are shuffled with a per-item deterministic seed; results are written incrementally to
`data/<session>/eval_results<suffix>.json` (the suffix encodes the setting, model and temperature).
The three main settings are **blind**, **video-only** and **full context** (video + complete raw
transcript, no timestamps). The docstring at the top of `evaluate_vqa.py` documents the additional
ablation flags.

```
python3 vqa_pipeline/score_vqa.py data/session0*/eval_results_only_video.json              # per-stage accuracy
python3 vqa_pipeline/score_vqa.py data/session0*/eval_results_only_video.json --hard-only  # hard subset
python3 vqa_pipeline/hard_items.py                                                         # hard-subset sizes
```

**Hard subset.** An item is *hard* if at least one blind run (`--no-video --no-text`; the temp-0
run plus any `--temperature` repeats, pooled) of the **same** judge model answers it wrongly — i.e.
it can't be reliably guessed from the question and choices alone. Run the blind setting first (e.g.
also `--temperature 1.0` twice; filenames get a `_temp1_0` suffix — rename to
`_temp1.0_run1`/`_run2` so they're pooled). `evaluate_vqa.py --hard-only` evaluates only those
items. The stage3 hard set is not a reliable difficulty control: blind stage3 correctness appears
driven by answer-position bias.

## Evaluating the released questions

The released `vqa.json` files are already the hard subset, so `--hard-only` is unnecessary (and
needs blind runs you'd have to produce first). Note that a blind score on this set is low by
construction for `gemini-3.1-pro-preview` — every item was missed by at least one of its blind
runs — so it isn't a guessability baseline for that model; for other models it still is.

Until the videos and transcripts are released, only the blind setting can be run:

```
python3 vqa_pipeline/evaluate_vqa.py --dataset session01 --no-video --no-text
python3 vqa_pipeline/score_vqa.py data/session01/eval_results_no_video_no_text.json
```

`evaluate_vqa.py` falls back to `item_metadata.json` when `gemini_queries.json`,
`interview_queries.json`, `label_annotations.json` or the video are absent, for settings that don't
need them. Prompts are byte-identical to those of the paper's blind runs. Once the videos are
released, placing them at `data/<session>/videos/<session>.mp4` also enables `--only-video`.

To evaluate a non-Gemini model, reproduce `build_choices()` and the prompt templates in
`evaluate_vqa.py`, and write results in the same `{"results": [{"key", "stage", "choices",
"correct_letter", "response", ...}]}` shape so `score_vqa.py` can score them.

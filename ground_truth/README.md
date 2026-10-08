# How `ground_truth_actions.json` is built

`data/<session>/annotations/ground_truth_actions.json` is the human-trusted action/state timeline
for a session's experiment video. Everything in the VQA pipeline that needs to know "what was
actually happening at time *t*" reads this file, not `inferred_states.json`.

It is produced in two phases:

- **an automated phase** (frames → scene segments → clips → `inferred_states.json`), which is
  MLLM-guessed and per-clip;
- **a reconciliation phase**, in which an LLM session (Claude Opus) merges `inferred_states.json`
  against the diarized transcript into a single corrected timeline, followed by a separate QC pass.

Only the second phase produces `ground_truth_actions.json`. The first phase produces one of its two
inputs.

---

## Schema

A flat JSON list, sorted by `start`, of two entry types:

```json
{"type": "scientist_action", "start": 140.0, "end": 212.0,
 "action": "Scientist sets up an oil bath on a hot plate and adjusts the temperature controls."}

{"type": "state", "start": 608.0, "end": 621.0,
 "subject": "oil bath", "state": "covers most of the flask"}
```

`scientist_action` entries are meant to tile the video with no gaps and no overlaps (or with gaps
only where nothing distinctly loggable happens — camera pans, filler, a 24-hour reaction skip).
`state` entries are observations *about* the scene rather than acts, and are allowed to share a span
with an action entry.

| session | entries | actions / states | covered range | `inferred_states` entries | `inferred_local_state` fields |
|---|---|---|---|---|---|
| session01 | 62 | 41 / 21 | 140–2593 s | 103 | 9 |
| session02 | 87 | 77 / 10 | 93–4305 s | 232 | 9 |
| session03 | 91 | 78 / 13 | 118–2322 s | 104 | 9 |
| session04 | 92 | 87 / 5 | 0–5309 s | 230 | 9 |
| session05 | 147 | 131 / 16 | 0–4361 s | 162 | 2 |

session04's file is **not sorted by `start`** (two inversions, indices 22 and 72). It is released
as used in the paper; `validate_ground_truth.py` reports it.

---

# Phase 1 — how `inferred_states.json` is made

Four steps, all automated, all Gemini on Vertex AI.

## 1. Frame extraction — `scene_split/extract_frames.py`

Reads every video in `data/raw_videos/` and writes JPEGs to
`scene_split/frames/<video-stem>/frame_NNNN.jpg` at 1 fps by default, plus a `metadata.json`
recording the extraction fps. Frame numbers are **1-indexed at that fps**, which is the coordinate
system every later step converts out of.

## 2. Scene segmentation — `scene_split/pipeline.py`

An interleaved split-then-merge sweep over the frames, 10 frames (`N_FRAMES`) per Gemini call,
`gemini-2.5-flash`:

- **Split.** Each chunk of 10 frames goes to Gemini with `split_prompt` (`scene_split/prompt.py`),
  which asks for a description, then whether the chunk is one activity or several, then the
  1-indexed boundary frames. Only `[3. Frames]: [...]` is parsed out — the description is thrown
  away. A chunk start is always provisionally a boundary.
- **Merge.** For each junction between adjacent chunks, the last frame of the previous chunk and the
  first frame of this one go to Gemini with `merge_prompt`, which asks a yes/no question: is the clip
  starting at the second frame the same activity as the first? "Yes" deletes the junction boundary.
  This is what lets an activity run longer than one 10-frame chunk.
- Both prompts share one definition of an activity: *a segment in which the presenter is working
  toward a single, nameable unit of progress that a viewer would naturally describe with one verb
  phrase*, ending when that unit completes or is explicitly abandoned. Off-task micro-actions
  (taking a sip of a drink) are explicitly ignorable. Intellectual activity counts.
- If a `transcript.json` exists in the frames folder, the matching transcript slice is passed
  alongside the frames for both calls.
- A checkpoint is written after **every** split and **every** merge, so a crashed run resumes with
  `--resume`. Transient errors (429/500/502/503/504, `UNAVAILABLE`, `RESOURCE_EXHAUSTED`, empty
  responses) are retried with exponential backoff (8 attempts).

Output: `scene_split/segments/<stem>_segments.json` — `[{"activity": 1, "start": 1, "end": 47}, ...]`,
in 1-indexed frame numbers.

## 3. Clip cutting — `inferred_states/cut_video.py`

Converts frame numbers to video time using the recorded extraction fps — a segment `(S, E)` becomes
`(S-1)/fps` to `E/fps` seconds — and ffmpeg-cuts the source video into
`inferred_states/sessions/<stem>/clips/activity_N_Xs_Ys.mp4`. Existing clips are skipped unless
`--overwrite`.

## 4. Per-clip inference → `inferred_states.json` — `inferred_states/clip_summaries.py`

One Gemini call per clip (`gemini-2.5-flash-lite`, temperature 0.2, clip bytes inlined), video
only, no memory threaded between clips, asking for `clip_summary` + `current_action`. These are the
only two fields anything downstream reads (`query_gemini_annotations.py`'s `load_segment_context()`
and the reconciliation below). Output is written incrementally and is resumable.

Sessions 01–04 were processed with an earlier, heavier per-clip engine (three calls per clip plus a
memory threaded across the session, producing 9 fields: the two above plus intent/cognitive-state
fields). Nothing downstream reads the extra seven, so only the lean engine (used for session05) is
released.

The finished file is copied to `data/<session>/annotations/inferred_states.json`.

### What this means for the merge step

`inferred_states.json` is **per-clip, vision-only, and unverified across clips**. Its
characteristic failure modes, all observed in practice:

- **Misread displays and labels** — a balance showing 3.61 g read as 0.61 g (session02), a hot
  plate's safety limit read as its setpoint (session03), a pump set to "80 **milliliters** per
  minute" when the transcript says 80 µL/min (session05).
- **Wrong object names** — "water bath" for an oil bath, "graduated cylinder" for a 20 mL tube,
  a vial label read as "SVS" when it is SDBS.
- **Near-duplicate runs** — 15 consecutive clips all re-describing "picking up a Pasteur pipette"
  over 980–1238 s in session01.
- **Content-free stretches** — from ~2200 s onward in session05 most entries are "progress bar is
  filling", "camera pans across a laboratory", "viewing a file explorer". The lean engine has no
  memory to notice this; the whole back-half structure had to come from the transcript.
- **No diagnosis** — it reports "adjusting equipment" where the transcript says liquid is
  back-flowing out of the air inlet.

## The transcript, the other input

`data/<session>/transcripts/<name>_transcript.json` — WhisperX ASR + pyannote diarization
(`transcription/transcribe_diarize.py`), speaker-labelled and timestamped.
`annotations/label_annotations.json` holds the human span labels over that transcript
(GOAL / PLAN / ACTION / INTENT_REASONING / ASSESSMENT / RISK_MANAGEMENT / KNOWLEDGE). Its failure
modes are complementary to `inferred_states`': ASR mangles chemical names and numbers
("STBS / SBTS / SDDS / HTBS" for SDBS; "coalescence" as "colors / corals / Colossus /
fluorescence"; "181.2 micromolar" for 181.2 µL), diarization splits one speaker's self-talk across
two labels, and speakers **narrate ahead of acting**.

---

# Phase 2 — the reconciliation that produces `ground_truth_actions.json`

This step is not scripted. It is a single Claude Opus agentic session (with file access to the
session's directory) handed both sources and asked to reason out the true timeline. The exact prompt
is in [`reconciliation_prompt.md`](reconciliation_prompt.md); it points the model at an
already-finished session's file as a format example.

## The reconciliation rules that emerged

The prompt deliberately does not specify a method — it names the core tension (MLLM-guessed vision
vs. a transcript where people may narrate ahead of the action) and lets the model reason it out. The
same rules came out each run, and they are the actual methodology of this step:

1. **Explain-before-do: the transcript says *what*, `inferred_states` says *when*.** Where the
   transcript uses future phrasing, anchor the span to the clip where the act is visibly happening.
   Transcript says "I'm going to add grease" at 537 s; `activity_13` shows greasing through 554 s;
   the entry is 537–559.
2. **On identity, quantities and units, the transcript wins.** Oil bath over "water bath",
   150 °C over the 250 °C safety limit, 3.61 g over a misread 0.61 g, 80 µL/min over
   "80 mL/min". Corroborate with arithmetic where possible — SDBS over SDS because MW 348.5 makes
   the dilution land on exactly the 0.0026 mM the speaker later quotes, and SDS (MW 288.4) does not.
3. **On what is visible but unnarrated, `inferred_states` wins** — and this is where most `state`
   entries come from: the green suspension after the acetic acid goes in, the yellow mixture with
   dark granular solid after the overnight reflux, the solid failing to pack down, the
   static-clinging powder.
4. **Collapse near-duplicate clip runs** into one entry; **split coarse clips** using the
   transcript's spoken steps (session01's 166 s "preparing reaction vessel" clip became five
   entries: measure solvent → pour into flask → add stir bar → place in oil bath → set stir rate).
5. **Normalize ASR garble only where context makes the reading unambiguous**, and say that you did.
6. **Decline to commit when both sources conflict and neither is checkable.** session05's capture
   resolution and frame rate were written generically ("sets the recording resolution and frame
   rate") rather than picking a number from two disagreeing sources.
7. **Distinguish stated intentions from performed acts.** session05's teardown at 4331 s is
   recorded as an intention, because the video ends 30 s later and only the pumps being switched off
   is visible.
8. **Flag, don't smooth.** The report-back is part of the deliverable: contradictions, soft numbers
   with no source in either input, recording dropouts, discontinuities, and judgment calls get named
   explicitly.
9. **Run a separate QC pass.** Re-checking the finished file against both sources found real defects
   that the build pass missed — a spurious pH state that was a probe transient rather than a
   reading, an entry asserting a calibration was finished that the video later contradicts, and the
   unsorted entries in session04.

## Self-checks — `validate_ground_truth.py`

Each run verified: schema key-set as specified above, entries sorted by `start`, action entries
contiguous with no gaps or overlaps across the covered range, and every `state` entry sharing a span
with an action entry. These structural checks are scripted:

```
python3 ground_truth/validate_ground_truth.py data/session01/annotations/ground_truth_actions.json
```

## Where the timeline starts

Coverage does not always start at 0. session01/session02 start at ~93–140 s and session03 at
118 s, excluding the opening to-camera narration; session04 and session05 start at 0 because their
real work begins immediately. `vqa_pipeline/dataset_config.py`'s `VIDEO_FLOOR_SECONDS_BY_DATASET`
floors clip sampling and eval windows at a comparable point (note: session05 uses the 98 s default
floor in the released results).

---

## What reads the file downstream

- **Stage 1 and stage 2 generation** get the whole file, descriptions included, as experiment context.
- **Stage 3 generation and all stage2/3 evaluation** get `ground_truth_timestamps_only()` — bare
  spans with descriptions stripped, so a timestamp question can't be answered by reading the answer
  off the sheet, and so generation is checked against the same withheld view the test-taker gets.

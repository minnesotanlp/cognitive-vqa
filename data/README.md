# Released data

One folder per recorded session (`session01` … `session05`), each a single laboratory experiment
recorded by its experimenter while narrating.

**The released questions are the blind-filtered ("hard") subset.** The full generated set
(1,251 items) was first evaluated *blind* — question and answer choices only, no video and no text
context — with the default judge (`gemini-3.1-pro-preview`) three times: once at temperature 0 and
twice at temperature 1.0. Only items answered **incorrectly in at least one** of those three runs
are released; items every blind run got right are guessable from the question's phrasing alone and
were dropped. (Every item parsed to a letter in every run, so none were excluded as ambiguous.)
Caveat: for stage3, blind correctness appears driven by answer-position bias more than by
guessability, so the stage3 filter is a weaker difficulty control than for the other stages.

| session | released items | stage1 | stage2 | stage3 | interview | of generated | video length |
|---|---|---|---|---|---|---|---|
| session01 | 105 | 44 | 41 | 13 | 7 | 210 | 43 min |
| session02 | 180 | 84 | 57 | 29 | 10 | 375 | 72 min |
| session03 | 101 | 36 | 41 | 12 | 12 | 262 | 39 min |
| session04 | 122 | 58 | 47 | 5 | 12 | 256 | 88 min |
| session05 | 66 | 26 | 29 | 11 | — | 148 | 73 min |
| **total** | **574** | 248 | 215 | 70 | 41 | 1,251 | |

## `vqa.json`

A list of items:

| field | |
|---|---|
| `question` | the question text |
| `correct_answer` | the correct option (prose for stage1/interview; a `"<start>s-<end>s"` video window for stage2/3) |
| `incorrect_answer1`, `incorrect_answer2` | distractors (one blind-generated, one video-grounded) |
| `rationale` | the generator's note on what the question tests (stage1/3/interview) |
| `current_point` | stage2 only: the video window the question's "now" refers to; given to the judge as text |
| `stage` | `stage1` (intent "why"), `stage2` (planning dependency, when), `stage3` (assessment, when), `interview` (intent "why", from a post-hoc interview) |
| `category` | the human span label(s) the item was generated from |
| `annotation_id` | stable item id; `annotation_id:stage` is unique |

Do not show `rationale` to a model under evaluation.

## `item_metadata.json`

Timing only, so `evaluate_vqa.py` can build its prompts without the unreleased files:
`video` (expected filename under `videos/`), `video_duration_seconds`, and, keyed by
`annotation_id`, each item's source span (`reference_start`, `reference_end`) and, for the stage1–3
items, the ~30 s `video_window` used for stage1 clips.

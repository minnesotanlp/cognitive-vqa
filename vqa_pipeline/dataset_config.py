"""
Per-dataset overrides for pipeline constants tuned against session01/session02 that don't
automatically generalize to a new dataset.
"""

# Seconds of intro to exclude from the start of a dataset's video (via VideoMetadata
# start_offset) before stage2/3's shared whole-video clip and stage1's per-item eval window
# floor. Default (98.0) matches session01/session02, which both open with pure to-camera
# narration before any hands-on lab work starts. Override per dataset when that assumption
# doesn't hold — e.g. session04's real action starts at 0s (see
# annotations/ground_truth_actions.json), so flooring at 98s would cut into genuine early
# footage instead of skipping filler.
DEFAULT_VIDEO_FLOOR_SECONDS = 98.0
VIDEO_FLOOR_SECONDS_BY_DATASET = {
    "session04": 0.0,
    # session03: floored at the end of its single GOAL-tagged transcript line (1.6-117.42s,
    # annotations/label_annotations.json) — skip the to-camera goal narration, keep everything
    # from the first hands-on lab work onward. (session01/session02's own GOAL lines end around
    # 94.2s, close to but not exactly the 98.0 default — that default wasn't derived this
    # precisely, just roughly eyeballed at the same "narration done" point.)
    "session03": 117.42,
}


def video_floor_seconds(dataset: str) -> float:
    return VIDEO_FLOOR_SECONDS_BY_DATASET.get(dataset, DEFAULT_VIDEO_FLOOR_SECONDS)

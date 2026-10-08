# Reconciliation prompt

Given verbatim (dataset names anonymized) to a Claude Opus agentic session with read/write access to
the session directory. `<session>` is the session being built; `<reference_session>` is one whose
`ground_truth_actions.json` was already finished, used only as a format example.

---

> for <session>, you need to create ground_truth_actions -- see <reference_session> for an example. The idea is
> that we know some things about the actions from the transcript, and some things from
> inferred_states. But inferred_states is guessed by an MLLM, and in the transcript the person might
> explain an action before they actually do it. So using this information, do your best to create a
> timeline in ground_truth_actions.
>
> ground_truth_actions.json is a flat list of entries, each either
> `{"type": "scientist_action", "start": <float seconds>, "end": <float seconds>, "action": "<description>"}`
> or `{"type": "state", "start": <float seconds>, "end": <float seconds>, "subject": "<what the state describes>", "state": "<the state itself>"}`).
>
> Everything for the dataset lives under `data/<session>/`: the MLLM's guesses in
> `annotations/inferred_states.json`, the diarized transcript in
> `transcripts/<name>_transcript.json`, and the session's goal in `goal/abstract.txt`. Write your
> result to `annotations/ground_truth_actions.json` in that same directory — <reference_session>'s finished
> file sits at that same relative path if you want to check the format against a real one.
>
> Read the goal first; it's usually the quickest way to make sense of what you're looking at.
> inferred_states.json is your other input: useful visual evidence, but MLLM-guessed and worth
> cross-checking against the transcript, correcting where you have good reason to (however dense or
> sparse it happens to be here).
>
> The exact second of a timestamp isn't very important, but it should generally be reasonable.
>
> When done, report back: how many entries you wrote, what time range they cover, and flag any real defects, contradictions between the two sources, or judgment calls you
> weren't fully sure about, rather than silently smoothing them over.

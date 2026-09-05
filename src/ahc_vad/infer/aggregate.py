"""Stage 3 - turn per-window verdicts into an event list.

The VLM never predicts timestamps (docs/architecture.md Stage 3): training data carries
almost no sub-clip localization signal, so asking it to would be asking it to guess. Instead
intervals are derived from WHERE windows fired.

Mechanism per class:
  - hysteresis: open an event at score >= theta_high, sustain while score >= theta_low.
    Stops a single event fragmenting into many when scores wobble around one threshold.
  - merge_gap: same-class events separated by less than this are joined.
  - min_duration: events shorter than this are dropped as noise.

IMPORTANT (docs/architecture.md 1.2): these parameters must be fit on long-form data - the
synthesized dev set - and NOT derived from train clip durations. Train `loitering` is
always ~30s because of how clips were CUT, while test loitering runs 2.6-37.6s; a dwell rule
copied from train statistics would miss most of the real events. The defaults below are
deliberately permissive placeholders until that fitting is done.
"""

from __future__ import annotations

from dataclasses import dataclass

from ahc_vad.eval.matching import Event


@dataclass(frozen=True)
class WindowVerdict:
    """One scored window from Stage 2."""

    start_time_sec: float
    end_time_sec: float
    class_name: str
    score: float


@dataclass(frozen=True)
class AggregationPolicy:
    theta_high: float = 0.5
    theta_low: float = 0.35
    merge_gap_sec: float = 5.0
    min_duration_sec: float = 0.0
    # If consecutive windows are further apart than this, treat it as a break in coverage
    # and close any open span. Without this a sparse verdict stream (one that omits normal
    # windows instead of scoring them 0) keeps a span open across the entire video and
    # collapses distinct events into one - which silently destroys Level 2/3 recall.
    max_window_gap_sec: float = 8.0


DEFAULT_POLICY = AggregationPolicy()

# Per-class overrides. Left mostly empty on purpose - see the module docstring. Populate
# only from measurements on long-form validation data.
CLASS_POLICIES: dict[str, AggregationPolicy] = {}


def policy_for(class_name: str) -> AggregationPolicy:
    return CLASS_POLICIES.get(class_name, DEFAULT_POLICY)


def aggregate_class(
    video_id: str,
    class_name: str,
    verdicts: list[WindowVerdict],
    policy: AggregationPolicy | None = None,
) -> list[Event]:
    """Hysteresis scan over the window stream, scoring one class.

    `verdicts` should be the DENSE window stream for the video - every window, including
    ones the model called normal. A window scores `w.score` when it predicted this class
    and 0.0 otherwise, so non-firing windows actively close spans. Passing only the firing
    windows still works via `max_window_gap_sec`, but dense input is strongly preferred.
    """
    policy = policy or policy_for(class_name)
    windows = sorted(verdicts, key=lambda w: w.start_time_sec)

    spans: list[tuple[float, float, list[float]]] = []  # (start, end, scores)
    open_span: list | None = None
    prev_end: float | None = None

    for w in windows:
        score = w.score if w.class_name == class_name else 0.0

        # A hole in window coverage closes any open span - see max_window_gap_sec.
        if (
            open_span is not None
            and prev_end is not None
            and w.start_time_sec - prev_end > policy.max_window_gap_sec
        ):
            spans.append(tuple(open_span))
            open_span = None

        if open_span is None:
            if score >= policy.theta_high:
                open_span = [w.start_time_sec, w.end_time_sec, [score]]
        else:
            if score >= policy.theta_low:
                open_span[1] = max(open_span[1], w.end_time_sec)
                open_span[2].append(score)
            else:
                spans.append(tuple(open_span))
                open_span = None

        prev_end = w.end_time_sec if prev_end is None else max(prev_end, w.end_time_sec)

    if open_span is not None:
        spans.append(tuple(open_span))

    merged: list[list] = []
    for start, end, scores in spans:
        if merged and start - merged[-1][1] <= policy.merge_gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].extend(scores)
        else:
            merged.append([start, end, list(scores)])

    return [
        Event(
            video_id=video_id,
            class_name=class_name,
            start_time_sec=start,
            end_time_sec=end,
            score=max(scores),
        )
        for start, end, scores in merged
        if (end - start) >= policy.min_duration_sec
    ]


def aggregate_video(video_id: str, verdicts: list[WindowVerdict]) -> list[Event]:
    """Aggregate every class independently, then return all events sorted by time.

    Classes are independent so one video can legitimately emit several overlapping events of
    different classes - which the data requires (T026 holds four different classes).
    Returns an empty list when nothing fires, the correct output for videos like T029/T030.
    """
    candidate_classes = {v.class_name for v in verdicts} - {"normal"}

    events: list[Event] = []
    for class_name in sorted(candidate_classes):
        # Pass the FULL window stream, not just this class's windows: windows predicting
        # other classes (or normal) score 0 here and correctly terminate spans.
        events.extend(aggregate_class(video_id, class_name, verdicts))

    return sorted(events, key=lambda e: (e.start_time_sec, e.class_name))

"""Sub-grid boundary refinement for aggregated events.

Why this exists: the aggregator takes an event's start from the FIRST firing window's start
and its end from the LAST firing window's end, so a span is quantised to the window grid and
systematically over-covers the true event by up to one window at each edge. The official
metric gates matches at temporal IoU >= 0.5, and measurement on the public test set shows
that quantisation alone caps what is achievable:

    2.6s event  -> best possible IoU 0.32   (cannot match at ANY threshold >= 0.5)
    5.0s events -> best possible IoU 0.62-0.80
    >=20s events-> 0.94-1.00

So timing, not classification, is the binding constraint at Levels 2/3 - the oracle scores
only 0.78 on L2 timing with a *perfect* window-level model.

Two corrections, both cheap and neither fitted to the test set:

1. `shrink_to_midpoint` - a window fires because the event is somewhere inside it, and in
   expectation that is its middle. Pulling each outer edge inward by half a window converts
   "the event is somewhere in these windows" into "the event spans these window centres",
   which is the unbiased estimate.

2. `interpolate_crossing` - when per-window scores are continuous (they are: P(anomaly) from
   token logprobs), the threshold crossing between the last quiet window and the first firing
   one can be located BETWEEN window centres by linear interpolation, giving genuine
   sub-window resolution rather than a grid snap.
"""

from __future__ import annotations

from ahc_vad.eval.matching import Event
from ahc_vad.infer.aggregate import WindowVerdict


def shrink_to_midpoint(event: Event, window_sec: float) -> Event:
    """Pull both edges inward by half a window, the unbiased estimate of the true boundary.

    Never collapses the interval: if the span is too short to shrink, it is centred and
    given a minimal positive duration so `end > start` still holds (a submission rule).
    """
    start, end = event.start_time_sec, event.end_time_sec
    if start is None or end is None:
        return event

    half = window_sec / 2.0
    new_start, new_end = start + half, end - half

    if new_end <= new_start:
        mid = (start + end) / 2.0
        new_start, new_end = mid - 0.25, mid + 0.25

    return Event(event.video_id, event.class_name, round(new_start, 3), round(new_end, 3),
                 event.score)


def interpolate_crossing(
    event: Event,
    verdicts: list[WindowVerdict],
    threshold: float,
) -> Event:
    """Place each edge at the score threshold crossing, interpolated between window centres.

    Uses the dense window stream for the event's class, so it needs the same verdicts the
    aggregator saw. Falls back to the input edge whenever there is no neighbouring window to
    interpolate against (e.g. an event touching the start or end of the video).
    """
    start, end = event.start_time_sec, event.end_time_sec
    if start is None or end is None:
        return event

    scored = sorted(
        ((w.start_time_sec + w.end_time_sec) / 2.0,
         w.score if w.class_name == event.class_name else 0.0)
        for w in verdicts
    )
    if len(scored) < 2:
        return event

    def cross(before, after):
        """Time where the score line between two windows crosses `threshold`."""
        (t0, s0), (t1, s1) = before, after
        if s1 == s0:
            return None
        frac = (threshold - s0) / (s1 - s0)
        if not 0.0 <= frac <= 1.0:
            return None
        return t0 + frac * (t1 - t0)

    new_start, new_end = start, end

    # Leading edge: last centre below threshold -> first centre at/above it.
    for i in range(len(scored) - 1):
        t_lo, s_lo = scored[i]
        t_hi, s_hi = scored[i + 1]
        if s_lo < threshold <= s_hi and t_hi >= start:
            c = cross((t_lo, s_lo), (t_hi, s_hi))
            if c is not None:
                new_start = c
            break

    # Trailing edge: last centre at/above threshold -> first centre below it.
    for i in range(len(scored) - 1, 0, -1):
        t_hi, s_hi = scored[i]
        t_lo, s_lo = scored[i - 1]
        if s_hi < threshold <= s_lo and t_lo <= end:
            c = cross((t_hi, s_hi), (t_lo, s_lo))
            if c is not None:
                new_end = c
            break

    if new_end <= new_start:
        return event
    return Event(event.video_id, event.class_name, round(new_start, 3), round(new_end, 3),
                 event.score)


def refine_events(
    events: list[Event],
    verdicts: list[WindowVerdict],
    window_sec: float = 4.0,
    mode: str = "midpoint",
    threshold: float = 0.5,
) -> list[Event]:
    """Apply a refinement mode to every event. `mode='none'` is the unrefined baseline."""
    if mode == "none":
        return events
    if mode == "midpoint":
        return [shrink_to_midpoint(e, window_sec) for e in events]
    if mode == "interpolate":
        return [interpolate_crossing(e, verdicts, threshold) for e in events]
    if mode == "both":
        return [
            shrink_to_midpoint(interpolate_crossing(e, verdicts, threshold), 0.0)
            for e in events
        ]
    raise ValueError(f"unknown refinement mode: {mode!r}")

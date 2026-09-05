"""Sub-window generation with overlap-based labeling (docs/architecture.md 3.1).

Runtime sees 4s/32s windows, so training should too. But a sub-window's label must come
from its TEMPORAL OVERLAP with the annotated interval, never from which class folder the
clip lives in.

Why this matters: `road_spill_or_debris` events cover only 38% of their clip on average
(`wrong_way_driving` 45%, `vehicle_blocking_traffic` 50%). Labeling every sub-window of
those clips by folder name would train the model to fire on ordinary traffic - manufacturing
exactly the false alarms the whole design exists to prevent.

Two overlap ratios are needed, not one:

    window_coverage = overlap / window_duration   -> how much of the WINDOW is event
    event_coverage  = overlap / event_duration    -> how much of the EVENT is in the window

A single ratio gets one case wrong. A 1s accident inside a 4s window has
window_coverage=0.25 (looks like background by that measure) but event_coverage=1.0 - the
whole event is right there and the window is genuinely positive. Conversely a 4s window
sitting inside a 60s congestion event has event_coverage=0.07 but window_coverage=1.0. A
window is positive when EITHER ratio clears its threshold.

Windows that overlap an event but clear neither threshold are AMBIGUOUS and are dropped
rather than guessed at - mislabeling them is worse than losing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ahc_vad.data.dataset import Event

# Short scale serves event-like classes; long scale serves classes defined by persistence
# (docs/architecture.md Stage 0) - loitering cannot be judged from 4 seconds.
SHORT_SCALE = (4.0, 2.0)  # (window_sec, stride_sec)
LONG_SCALE = (32.0, 8.0)

DWELL_CLASSES = frozenset(
    {
        "loitering_or_suspicious_presence",
        "stalled_or_broken_down_vehicle",
        "fighting_or_violence",
        "wrong_way_driving",
    }
)


@dataclass(frozen=True)
class LabeledWindow:
    video_path: Path
    video_id: str
    start_time_sec: float
    end_time_sec: float
    class_name: str  # "normal" or one of the anomaly classes
    scale: str  # "short" | "long"

    @property
    def duration_sec(self) -> float:
        return self.end_time_sec - self.start_time_sec


def scale_for(class_name: str) -> tuple[float, float]:
    return LONG_SCALE if class_name in DWELL_CLASSES else SHORT_SCALE


def label_window(
    win_start: float,
    win_end: float,
    ev_start: float | None,
    ev_end: float | None,
    class_name: str,
    window_coverage_threshold: float = 0.5,
    event_coverage_threshold: float = 0.5,
    min_overlap_sec: float = 1.0,
) -> str | None:
    """Return the window's label, or None when it is ambiguous and should be dropped.

    `min_overlap_sec` guards against labeling a window for an event too brief to survive
    frame sampling: at 8 frames over 4s the sampler steps every 0.5s, so a 0.2s event can
    fall entirely between sampled frames. The label would then describe content the model
    never sees. Such windows are dropped, not called normal - the event is real, we just
    cannot show it.
    """
    if ev_start is None or ev_end is None:
        return "normal"

    overlap = min(win_end, ev_end) - max(win_start, ev_start)
    if overlap <= 0:
        return "normal"
    if overlap < min_overlap_sec:
        return None

    win_dur = win_end - win_start
    ev_dur = ev_end - ev_start
    window_coverage = overlap / win_dur if win_dur > 0 else 0.0
    event_coverage = overlap / ev_dur if ev_dur > 0 else 1.0

    if window_coverage >= window_coverage_threshold or event_coverage >= event_coverage_threshold:
        return class_name
    return None  # touches the event but not convincingly - drop rather than guess


def windows_for_event(
    event: Event,
    duration_sec: float,
    scale: str = "short",
    **thresholds,
) -> list[LabeledWindow]:
    """Generate labeled sub-windows across one training clip."""
    window_sec, stride_sec = SHORT_SCALE if scale == "short" else LONG_SCALE

    ev_start, ev_end = event.start_time_sec, event.end_time_sec

    # 49 train events (2.2%) are annotated with start == end - a labeling artifact, not a
    # genuinely instantaneous event, and they include 11 accidents and 16 stalled vehicles.
    # Taken literally the overlap is zero and every window of those clips would be labeled
    # normal, teaching the model that footage containing a stalled vehicle is unremarkable.
    # Train clips are pre-trimmed (median event coverage 0.999, 70% start at t=0), so
    # "the event spans the clip" is the right prior for a degenerate annotation.
    if ev_start is not None and ev_end is not None and ev_end - ev_start <= 0:
        ev_start, ev_end = 0.0, duration_sec

    # Clips are often shorter than the long window; use the whole clip rather than skipping.
    if duration_sec <= window_sec:
        spans = [(0.0, duration_sec)]
    else:
        spans = []
        t = 0.0
        while t < duration_sec:
            end = min(t + window_sec, duration_sec)
            spans.append((t, end))
            if end >= duration_sec:
                break
            t += stride_sec

    out = []
    for start, end in spans:
        label = label_window(start, end, ev_start, ev_end, event.class_name, **thresholds)
        if label is None:
            continue
        out.append(
            LabeledWindow(
                video_path=event.video_path,
                video_id=event.video_id,
                start_time_sec=start,
                end_time_sec=end,
                class_name=label,
                scale=scale,
            )
        )
    return out

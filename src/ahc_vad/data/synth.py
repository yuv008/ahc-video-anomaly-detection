"""Synthesize Level-2-style long videos (docs/architecture.md 3.2).

Level 2 test videos are constructed: T025's events sit at exactly 20-40, 60-80, 100-120,
140-160, 180-200, 220-240s, and T028's at 30-35, 90-95, 150-155, 210-215. Round numbers and
regular spacing mean short clips were concatenated into normal background. We reproduce that
construction to get training and validation data with the right shape.

Why this is the highest-leverage item: training clips are pre-trimmed, so an anomaly fills
~100% of every clip the model ever sees. Long videos are ~8% anomaly. Without long-form
data the model never sees sustained background in context, and it has no way to learn not to
fire on it. These synthesized videos supply exactly that, plus an honest dev set for fitting
gate and aggregator thresholds - which must NOT be fit on the 34-video public test set.

NO VIDEO IS RENDERED. A synthetic video is a manifest: an ordered list of segments naming a
source clip and a time offset. `sample_frames_synthetic` resolves a timeline position to the
underlying source file at read time. This keeps the dataset free (no extra GB on disk, no
ffmpeg dependency) and makes regenerating with different parameters instant.

LIMITATION (docs/architecture.md 8.1): this reproduces Level 2 only. Level 3 is natural
footage whose events run up to 125s - longer than any training clip (max 30s) - so no amount
of stitching represents it. Synthesized videos inherit the <=30s event ceiling.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from ahc_vad.data.dataset import Event

# Matched to the real Level 2 videos: exactly 240s, 1-6 events, ~8% anomaly coverage.
TARGET_DURATION_SEC = 240.0
EVENTS_PER_VIDEO = (1, 6)
TARGET_COVERAGE = 0.08


@dataclass(frozen=True)
class Segment:
    """One source clip placed on the synthetic timeline."""

    start_sec: float  # position in the synthetic timeline
    end_sec: float
    source_path: str
    source_start_sec: float  # where playback begins inside the source clip
    class_name: str | None = None  # None => normal filler


@dataclass
class SyntheticVideo:
    video_id: str
    duration_sec: float
    segments: list[Segment]
    events: list[dict]  # {class_name, start_time_sec, end_time_sec}

    def to_json(self) -> str:
        return json.dumps(
            {
                "video_id": self.video_id,
                "duration_sec": self.duration_sec,
                "segments": [asdict(s) for s in self.segments],
                "events": self.events,
            }
        )

    @staticmethod
    def from_json(line: str) -> SyntheticVideo:
        d = json.loads(line)
        return SyntheticVideo(
            video_id=d["video_id"],
            duration_sec=d["duration_sec"],
            segments=[Segment(**s) for s in d["segments"]],
            events=d["events"],
        )

    def resolve(self, t: float) -> tuple[str, float] | None:
        """Map a timeline position to (source_path, time within that source clip)."""
        for seg in self.segments:
            if seg.start_sec <= t < seg.end_sec:
                return seg.source_path, seg.source_start_sec + (t - seg.start_sec)
        return None


def probe_duration(path: Path, cache: dict[str, float] | None = None) -> float:
    key = str(path)
    if cache is not None and key in cache:
        return cache[key]
    cap = cv2.VideoCapture(key)
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    dur = n / fps if fps else 0.0
    if cache is not None:
        cache[key] = dur
    return dur


def build_synthetic_video(
    video_id: str,
    normal_events: list[Event],
    anomaly_events: list[Event],
    rng: random.Random,
    duration_cache: dict[str, float],
    target_duration: float = TARGET_DURATION_SEC,
    target_coverage: float = TARGET_COVERAGE,
) -> SyntheticVideo | None:
    """Lay anomaly clips into normal background, separated by filler."""
    n_events = rng.randint(*EVENTS_PER_VIDEO)
    budget = target_duration * target_coverage

    chosen: list[tuple[Event, float]] = []
    used = 0.0
    # An empty anomaly pool is a deliberate request for pure background - the dev set needs
    # event-free videos to measure a false alarm rate at all (mirrors real T029/T030).
    for ev in rng.sample(anomaly_events, min(len(anomaly_events), n_events * 4)):
        dur = probe_duration(ev.video_path, duration_cache)
        if dur <= 0:
            continue
        if used + dur > budget * 1.5 and chosen:
            continue
        chosen.append((ev, dur))
        used += dur
        if len(chosen) >= n_events:
            break

    if not chosen and anomaly_events:
        return None  # pool was non-empty but nothing usable was found

    filler_total = max(0.0, target_duration - used)
    n_gaps = max(len(chosen) + 1, 1)
    # Random split of the filler across gaps, so events do not land on a fixed grid.
    cuts = sorted(rng.uniform(0, filler_total) for _ in range(n_gaps - 1))
    gaps = [b - a for a, b in zip([0.0, *cuts], [*cuts, filler_total], strict=True)]

    segments: list[Segment] = []
    events: list[dict] = []
    cursor = 0.0

    def add_filler(amount: float) -> None:
        nonlocal cursor
        remaining = amount
        while remaining > 0.5:
            src = rng.choice(normal_events)
            src_dur = probe_duration(src.video_path, duration_cache)
            if src_dur <= 0.5:
                continue
            take = min(src_dur, remaining)
            segments.append(
                Segment(cursor, cursor + take, str(src.video_path), 0.0, None)
            )
            cursor += take
            remaining -= take

    for i, (ev, dur) in enumerate(chosen):
        add_filler(gaps[i])

        clip_start = cursor
        segments.append(Segment(cursor, cursor + dur, str(ev.video_path), 0.0, ev.class_name))
        cursor += dur

        # Ground truth is the event's interval INSIDE the clip, shifted onto the timeline -
        # not the whole clip. For low-coverage classes those differ substantially.
        ev_start = ev.start_time_sec if ev.start_time_sec is not None else 0.0
        ev_end = ev.end_time_sec if ev.end_time_sec is not None else dur
        if ev_end - ev_start <= 0:  # degenerate annotation: treat clip as the event
            ev_start, ev_end = 0.0, dur
        events.append(
            {
                "class_name": ev.class_name,
                "start_time_sec": round(clip_start + ev_start, 3),
                "end_time_sec": round(clip_start + min(ev_end, dur), 3),
            }
        )

    add_filler(gaps[-1])

    return SyntheticVideo(video_id, round(cursor, 3), segments, events)


def sample_frames_synthetic(video: SyntheticVideo, start: float, end: float, num_frames: int):
    """Sample frames across a synthetic timeline window, reading from source clips."""
    from PIL import Image

    timestamps = [
        start + (end - start) * i / max(num_frames - 1, 1) for i in range(num_frames)
    ]

    # Group by source file so each clip is opened once, not once per frame.
    wanted: dict[str, list[tuple[int, float]]] = {}
    for idx, t in enumerate(timestamps):
        resolved = video.resolve(min(t, video.duration_sec - 1e-3))
        if resolved is None:
            continue
        path, src_t = resolved
        wanted.setdefault(path, []).append((idx, src_t))

    frames: dict[int, Image.Image] = {}
    for path, items in wanted.items():
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        for idx, src_t in items:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(src_t * fps)))
            ok, frame = cap.read()
            if ok:
                frames[idx] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    return [frames[i] for i in sorted(frames)]

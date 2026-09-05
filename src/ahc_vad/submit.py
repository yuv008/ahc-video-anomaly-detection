"""Build the arena submission JSON from aggregated events.

Encodes the format doc's rules, including the ones listed under "Things that catch people
out" - each of these is a hard rejection or a silent zero, not a style preference:

  1. `"class_name": "normal"` is REJECTED. A normal video must be `"events": []`.
  2. Timestamps on Level-1 events are REJECTED. They must be null.
  3. Level 1 wants ONE label for the clip; repeating a class earns nothing, so dedupe.
  4. Many fragments for one event: only the best-overlapping can match, the rest count
     against you. So fragments are merged/pruned before writing.
  5. Predicting anything on a normal Level-2/3 video scores that video ZERO. Emitting
     nothing when unsure is worth more than a speculative event.
  6. `runtime_metadata` is REQUIRED on every video and is where the latency bonus comes
     from - a missing one invalidates the file.
  7. Omitted videos are NOT cleared; they keep the previous answer, and a video never
     answered is scored as normal.

Also: `explanation` is optional, bonus-only, and 20-500 characters. It never costs anything
to omit, so it is only written when a description is actually available and in range.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ahc_vad.schema import ANOMALY_CLASSES

VALID_CLASSES = set(ANOMALY_CLASSES)  # 11 anomaly classes; "normal" is NOT emittable


@dataclass
class RuntimeMetadata:
    """Per-video timing. Required on every video.

    `end_to_end_internal_time_ms` must cover decoding, preprocessing, inference and
    postprocessing for that video, and must EXCLUDE model loading and downloads.
    """

    frames_processed: int
    chunks_processed: int
    end_to_end_internal_time_ms: float
    model_runtimes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frames_processed": int(self.frames_processed),
            "chunks_processed": int(self.chunks_processed),
            "end_to_end_internal_time_ms": round(float(self.end_to_end_internal_time_ms), 1),
            "model_runtimes": self.model_runtimes,
        }


def make_model_runtime(model_name: str, call_times_ms: list[float]) -> dict:
    """Build a model_runtimes entry.

    The validator checks average == total / calls within 2%, and that any call_times_ms has
    exactly call_count entries - so all of it is derived from one list rather than passed in
    separately, which is where inconsistencies creep in.
    """
    n = len(call_times_ms)
    total = float(sum(call_times_ms))
    ordered = sorted(call_times_ms)

    def pct(p: float) -> float:
        if not ordered:
            return 0.0
        idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
        return round(ordered[idx], 1)

    return {
        "model_name": model_name,
        "call_count": n,
        "total_time_ms": round(total, 1),
        "average_time_ms": round(total / n, 1) if n else 0.0,
        "p50_time_ms": pct(0.50),
        "p95_time_ms": pct(0.95),
        "max_time_ms": round(max(call_times_ms), 1) if call_times_ms else 0.0,
    }


def _dedupe_level1(events: list[dict]) -> list[dict]:
    """Level 1 wants one label for the clip; repeats earn nothing. Keep the best-scoring."""
    if not events:
        return []
    best = max(events, key=lambda e: e.get("_score", 0.0))
    return [{"class_name": best["class_name"], "start_time_sec": None, "end_time_sec": None}]


def _clean_l23(events: list[dict]) -> list[dict]:
    """Level 2/3 events: require a valid interval with end > start."""
    out = []
    for e in events:
        s, t = e.get("start_time_sec"), e.get("end_time_sec")
        if s is None or t is None or t <= s or s < 0:
            continue
        out.append({
            "class_name": e["class_name"],
            "start_time_sec": round(float(s), 3),
            "end_time_sec": round(float(t), 3),
        })
    return out


def build_video_entry(
    video_id: str,
    level: int,
    events: list[dict],
    runtime: RuntimeMetadata,
    explanations: dict[str, str] | None = None,
) -> dict:
    """One `predictions[]` entry, with the level's rules applied."""
    usable = [e for e in events if e.get("class_name") in VALID_CLASSES]

    if level == 1:
        cleaned = _dedupe_level1(usable)
    else:
        cleaned = _clean_l23(usable)

    if explanations:
        for e in cleaned:
            text = explanations.get(e["class_name"])
            if text and 20 <= len(text) <= 500:  # outside that range it is rejected
                e["explanation"] = text

    return {
        "video_id": video_id,
        "events": cleaned,          # empty list == "normal"; never class_name "normal"
        "runtime_metadata": runtime.to_dict(),
    }


def build_submission(
    entries: list[dict],
    submission_id: str,
    model_name: str,
    total_wall_time_ms: float,
    hardware: str,
) -> dict:
    return {
        "schema_version": "1.0",
        "submission_id": submission_id,
        "model_name": model_name,
        "run_metadata": {
            "total_wall_time_ms": round(float(total_wall_time_ms), 1),
            "hardware": hardware,
        },
        "predictions": entries,
    }


def validate(submission: dict, manifest_levels: dict[str, int] | None = None) -> list[str]:
    """Re-check the rules locally. A rejected file does not cost a run, but a wasted
    upload does cost time, and some mistakes (false alarms on normal videos) are silent."""
    problems: list[str] = []
    preds = submission.get("predictions")
    if not isinstance(preds, list) or not preds:
        return ["predictions missing or empty"]

    seen: set[str] = set()
    for entry in preds:
        vid = entry.get("video_id")
        if not vid:
            problems.append("entry with no video_id")
            continue
        if vid in seen:
            problems.append(f"{vid}: appears more than once")
        seen.add(vid)

        if "runtime_metadata" not in entry:
            problems.append(f"{vid}: runtime_metadata missing (required on every video)")
        else:
            rm = entry["runtime_metadata"]
            for mr in rm.get("model_runtimes", []):
                n, total, avg = mr.get("call_count"), mr.get("total_time_ms"), mr.get("average_time_ms")
                if n and total is not None and avg is not None:
                    expected = total / n
                    if expected and abs(avg - expected) / expected > 0.02:
                        problems.append(
                            f"{vid}: model_runtimes average {avg} != total/calls {expected:.1f} (>2%)"
                        )
                times = mr.get("call_times_ms")
                if times is not None and n is not None and len(times) != n:
                    problems.append(f"{vid}: call_times_ms has {len(times)} entries, call_count={n}")

        level = (manifest_levels or {}).get(vid)
        for e in entry.get("events", []):
            cls = e.get("class_name")
            if cls == "normal":
                problems.append(f"{vid}: class_name 'normal' is rejected - use events: []")
            elif cls not in VALID_CLASSES:
                problems.append(f"{vid}: unknown class_name {cls!r}")

            s, t = e.get("start_time_sec"), e.get("end_time_sec")
            if level == 1:
                if s is not None or t is not None:
                    problems.append(f"{vid}: Level-1 events must have null timestamps")
            elif level in (2, 3):
                if s is None or t is None:
                    problems.append(f"{vid}: Level-{level} events need timestamps")
                elif s < 0 or t <= s:
                    problems.append(f"{vid}: bad interval [{s}, {t}] (need end > start >= 0)")

            expl = e.get("explanation")
            if expl is not None and not (20 <= len(expl) <= 500):
                problems.append(f"{vid}: explanation must be 20-500 chars (got {len(expl)})")

        if level == 1 and len(entry.get("events", [])) > 1:
            problems.append(f"{vid}: Level 1 wants one label; {len(entry['events'])} given")

    return problems


def write_submission(submission: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(submission, indent=1), encoding="utf-8")
    mb = path.stat().st_size / 1e6
    print(f"wrote {path}  ({mb:.2f} MB)")
    if mb > 5:  # hard limit in the doc
        print(f"  WARNING: exceeds the 5 MB limit - drop `explanation` fields or reduce indent")

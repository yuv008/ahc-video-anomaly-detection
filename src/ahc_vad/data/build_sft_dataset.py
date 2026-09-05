"""Build window-level supervised fine-tuning data for the Stage-2 VLM.

Examples are WINDOWS, not whole clips (docs/architecture.md 3.1). Runtime sees 4s/32s
windows, so training must too, and each window's label comes from its temporal overlap with
the annotated interval rather than from the class folder it came from. That distinction
recovers a large amount of correctly-labeled background from inside anomaly clips - measured
at 67% of `road_spill_or_debris` windows, 61% of `vehicle_blocking_traffic` - which would
otherwise be mislabeled as anomalous and train the model to raise false alarms.

The target is a minimal JSON verdict with NO description field. `description_summary` in the
dataset is templated boilerplate (300 loitering clips share one string, docs/architecture.md
1.3), so there is nothing to learn from generating it, and every extra decoded token costs
latency at runtime across many feeds.

Two outputs:
  - JSONL for ms-swift (`swift sft --dataset ...`)
  - a list of dicts in Unsloth conversation format, built with a list comprehension because
    dataset.map() breaks on multi-image samples (per the hackathon primer)
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import cv2

from ahc_vad.data.dataset import load_train_events
from ahc_vad.data.sampling import sample_frames
from ahc_vad.data.windows import LabeledWindow, scale_for, windows_for_event
from ahc_vad.schema import ANOMALY_CLASSES

SYSTEM_PROMPT = (
    "You are a real-time visual anomaly detector for city drone, CCTV and dashcam footage. "
    "Given a short sequence of frames from one time window, decide whether they show one of "
    "these anomalies: " + ", ".join(ANOMALY_CLASSES) + ", or normal if nothing of concern is "
    "happening. Most footage is ordinary and should be called normal. "
    'Reply with a single JSON object: {"is_anomaly": true|false, "class_name": "<label>"}.'
)

USER_PROMPT = "What is happening in this window?"


def target_json(class_name: str) -> str:
    return json.dumps(
        {"is_anomaly": class_name != "normal", "class_name": class_name},
        separators=(",", ":"),
    )


def _probe_duration(path: Path, cache: dict[str, float]) -> float:
    key = str(path)
    if key not in cache:
        cap = cv2.VideoCapture(key)
        fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        cache[key] = n / fps if fps else 0.0
    return cache[key]


def build_windows(
    dataset_root: Path,
    include_long_scale: bool = True,
) -> list[LabeledWindow]:
    """Expand every training clip into labeled windows at the appropriate scale(s)."""
    events = load_train_events(dataset_root)
    cache: dict[str, float] = {}
    windows: list[LabeledWindow] = []

    for event in events:
        duration = _probe_duration(event.video_path, cache)
        if duration <= 0:
            continue

        scales = ["short"]
        if include_long_scale and scale_for(event.class_name)[0] != 4.0:
            scales = ["short", "long"]

        for scale in scales:
            windows.extend(windows_for_event(event, duration, scale=scale))

    return windows


def balance_windows(
    windows: list[LabeledWindow],
    rng: random.Random,
    normal_ratio: float = 1.5,
    max_per_class: int | None = None,
) -> list[LabeledWindow]:
    """Correct the 7.3:1 class imbalance and set the normal:anomaly ratio deliberately.

    At runtime background windows vastly outnumber event windows; the raw training set does
    not reflect that (docs/architecture.md 3.3). `normal_ratio` is normals per anomaly
    window - above 1.0 biases toward precision, which is what the brief asks for since
    "false alarms matter as much as missed detections".
    """
    by_class: dict[str, list[LabeledWindow]] = {}
    for w in windows:
        by_class.setdefault(w.class_name, []).append(w)

    normals = by_class.pop("normal", [])

    if max_per_class is None:
        counts = [len(v) for v in by_class.values()]
        max_per_class = int(sorted(counts)[len(counts) // 2]) if counts else 0

    kept: list[LabeledWindow] = []
    for cls, items in by_class.items():
        if len(items) > max_per_class:
            kept.extend(rng.sample(items, max_per_class))
        else:
            kept.extend(items)  # under-represented classes keep everything

    n_normal = int(len(kept) * normal_ratio)
    kept.extend(rng.sample(normals, min(n_normal, len(normals))))

    rng.shuffle(kept)
    return kept


def build_swift_jsonl(windows: list[LabeledWindow], out_path: Path, num_frames: int = 8) -> None:
    """Write ms-swift compatible JSONL. Frames are resolved at train time, not here."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for w in windows:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": "<image>" * num_frames + f"\n{USER_PROMPT}"},
                            {"role": "assistant", "content": target_json(w.class_name)},
                        ],
                        "video": str(w.video_path),
                        "start_time_sec": w.start_time_sec,
                        "end_time_sec": w.end_time_sec,
                    }
                )
                + "\n"
            )


def build_unsloth_examples(windows: list[LabeledWindow], num_frames: int = 8) -> list[dict]:
    """Materialize frames and return Unsloth conversation format.

    Built with a list comprehension - dataset.map() breaks on multi-image samples.
    """
    return [
        {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img}
                        for img in sample_frames(
                            w.video_path, num_frames, w.start_time_sec, w.end_time_sec
                        )
                    ]
                    + [{"type": "text", "text": USER_PROMPT}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target_json(w.class_name)}],
                },
            ]
        }
        for w in windows
    ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--normal-ratio", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-long-scale", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    windows = build_windows(args.dataset_root, include_long_scale=not args.no_long_scale)
    print(f"raw windows: {len(windows)}")
    print("  " + str(Counter(w.class_name for w in windows).most_common()))

    balanced = balance_windows(windows, rng, normal_ratio=args.normal_ratio)
    print(f"\nbalanced windows: {len(balanced)}")
    for cls, n in Counter(w.class_name for w in balanced).most_common():
        print(f"  {cls:35s} {n:5d}")

    build_swift_jsonl(balanced, args.out, args.num_frames)
    print(f"\nwrote {len(balanced)} examples -> {args.out}")


if __name__ == "__main__":
    main()

"""Measure the pipeline's ORACLE CEILING - the best score achievable with a perfect model.

Simulates a Stage-2 VLM that is always right at the window level, then runs the real
aggregator and evaluator. Whatever this reports is the upper bound imposed by the
window/aggregation configuration ALONE. Model quality can never exceed it.

Use it to tell two very different failures apart:
  - ceiling is high but real score is low  -> the model is the problem
  - ceiling itself is low                  -> window size, stride, or aggregation params
                                              are throwing away events before the model
                                              gets a chance

Run it again after changing --window-sec/--stride-sec/merge_gap to see what the config
change actually bought.

Usage:
    python scripts/oracle_ceiling.py --dataset-root data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from ahc_vad.data.sampling import sliding_windows
from ahc_vad.eval.evaluate import evaluate, load_predictions, print_report
from ahc_vad.infer.aggregate import WindowVerdict, aggregate_video


def video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps if fps else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--stride-sec", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=Path("data/processed/pred_oracle.csv"))
    args = parser.parse_args()

    test_root = args.dataset_root / "test"
    gt = pd.read_csv(test_root / "ground_truth.csv")
    videos = pd.read_csv(test_root / "videos.csv")

    rows = []
    for row in videos.itertuples(index=False):
        duration = video_duration(test_root / row.filename)
        events = gt[
            (gt["video_id"] == row.video_id)
            & (gt["is_anomaly"])
            & (gt["class_name"] != "normal")
        ]

        verdicts = []
        for start, end in sliding_windows(duration, args.window_sec, args.stride_sec):
            hit = None
            for e in events.itertuples(index=False):
                ev_start, ev_end = e.start_time_sec, e.end_time_sec
                if pd.isna(ev_start):  # Level 1: untimed, event spans the clip
                    ev_start, ev_end = 0.0, duration
                if min(end, ev_end) - max(start, ev_start) > 0:
                    hit = e.class_name
                    break
            # Dense stream: normal windows are emitted too, and are what close spans.
            verdicts.append(WindowVerdict(start, end, hit or "normal", 0.9 if hit else 0.0))

        for ev in aggregate_video(row.video_id, verdicts):
            rows.append(
                {
                    "video_id": ev.video_id,
                    "class_name": ev.class_name,
                    "start_time_sec": ev.start_time_sec,
                    "end_time_sec": ev.end_time_sec,
                    "score": ev.score,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)

    print(
        f"ORACLE CEILING  (window={args.window_sec}s stride={args.stride_sec}s)\n"
        f"Perfect window-level model + real aggregator + real evaluator.\n"
    )
    print_report(evaluate(args.dataset_root, load_predictions(args.out)))


if __name__ == "__main__":
    main()

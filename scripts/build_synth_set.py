"""Generate the synthesized long-form dev set (docs/architecture.md 3.2 / 3.4).

Output is laid out so the existing evaluator works on it unchanged:

    <out>/test/ground_truth.csv   same schema as the real test set (level=2)
    <out>/test/videos.csv         video_id -> manifest reference
    <out>/manifest.jsonl          one SyntheticVideo per line (segments + events)

so you can score against it with:

    python -m ahc_vad.eval.evaluate --predictions preds.csv --dataset-root <out>

This is the PRIMARY dev set for fitting gate and aggregator thresholds. The 34-video public
test set is too small to tune on (one video is ~3% of the score) and is not the private set.

Usage:
    python scripts/build_synth_set.py --n-videos 40 --out data/processed/synth
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ahc_vad.data.dataset import load_train_events
from ahc_vad.data.synth import build_synthetic_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/synth"))
    parser.add_argument("--n-videos", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--normal-fraction",
        type=float,
        default=0.2,
        help="Fraction of synthetic videos containing NO events. Without these the dev set "
        "cannot measure a false alarm rate - which is its main purpose. The real test set "
        "is 6/34 normal (~18%%).",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.3,
        help="Fraction of source clips reserved for synthesis, so synthetic dev videos are "
        "built from clips the model did not train on (docs/architecture.md 3.4).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    events = load_train_events(args.dataset_root)

    normal = [e for e in events if e.class_name == "normal"]
    anomaly = [e for e in events if e.class_name != "normal"]

    # Split BY SOURCE VIDEO so a clip cannot appear in both training and the dev set.
    rng.shuffle(normal)
    rng.shuffle(anomaly)
    n_hold = int(len(normal) * args.holdout_fraction)
    a_hold = int(len(anomaly) * args.holdout_fraction)
    normal_pool, anomaly_pool = normal[:n_hold], anomaly[:a_hold]

    print(
        f"synthesis pool: {len(normal_pool)} normal / {len(anomaly_pool)} anomaly clips "
        f"({args.holdout_fraction:.0%} holdout)"
    )

    duration_cache: dict[str, float] = {}
    videos, gt_rows, vid_rows = [], [], []

    n_normal_videos = int(args.n_videos * args.normal_fraction)

    for i in tqdm(range(args.n_videos), desc="synthesizing"):
        is_normal_video = i < n_normal_videos
        sv = build_synthetic_video(
            video_id=f"SYN{i:04d}",
            normal_events=normal_pool,
            # An empty anomaly pool yields pure-background footage, mirroring T029/T030.
            anomaly_events=[] if is_normal_video else anomaly_pool,
            rng=rng,
            duration_cache=duration_cache,
        )
        if sv is None:
            continue
        videos.append(sv)
        vid_rows.append({"video_id": sv.video_id, "filename": f"manifest:{sv.video_id}"})

        if not sv.events:
            # Matches the real schema: one row, is_anomaly=False, blank timestamps.
            gt_rows.append(
                {
                    "video_id": sv.video_id,
                    "level": 2,
                    "is_anomaly": False,
                    "class_name": "normal",
                    "start_time_sec": "",
                    "end_time_sec": "",
                    "description_summary": "",
                }
            )
        for ev in sv.events:
            gt_rows.append(
                {
                    "video_id": sv.video_id,
                    "level": 2,
                    "is_anomaly": True,
                    "class_name": ev["class_name"],
                    "start_time_sec": ev["start_time_sec"],
                    "end_time_sec": ev["end_time_sec"],
                    "description_summary": "",
                }
            )

    test_dir = args.out / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gt_rows).to_csv(test_dir / "ground_truth.csv", index=False)
    pd.DataFrame(vid_rows).to_csv(test_dir / "videos.csv", index=False)
    with (args.out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for sv in videos:
            f.write(sv.to_json() + "\n")

    total = sum(v.duration_sec for v in videos)
    ev_total = sum(e["end_time_sec"] - e["start_time_sec"] for v in videos for e in v.events)
    print(f"\nwrote {len(videos)} videos -> {args.out}")
    print(f"  total duration : {total/60:.1f} min")
    print(f"  events         : {len(gt_rows)}  ({len(gt_rows)/max(len(videos),1):.1f} per video)")
    print(f"  anomaly coverage: {ev_total/total:.1%}   (real Level 2/3 median is 7.7%)")


if __name__ == "__main__":
    main()

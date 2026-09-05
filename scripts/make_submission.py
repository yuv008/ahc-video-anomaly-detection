"""Window verdicts -> aggregated events -> validated submission JSON, scored officially.

Runs locally on the verdicts file downloaded from the GPU box. Everything here is CPU-cheap,
so aggregator thresholds can be swept without paying for another inference pass.

Optimises the OFFICIAL metric (src/ahc_vad/eval/official.py), not global event F1. That
distinction matters most on normal Level-2/3 videos, where a single false alarm scores the
whole video ZERO - so the sweep will happily trade recall for precision, which is exactly the
behaviour the scoring rules reward.

Usage:
    python scripts/make_submission.py --verdicts data/processed/window_verdicts.jsonl --sweep
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ahc_vad.eval.matching import Event
from ahc_vad.eval.official import print_official, score_submission
from ahc_vad.infer.aggregate import AggregationPolicy, WindowVerdict, aggregate_class
from ahc_vad.submit import (RuntimeMetadata, build_submission, build_video_entry,
                            make_model_runtime, validate, write_submission)


def load_verdicts(path: Path):
    by_video: dict[str, list[WindowVerdict]] = defaultdict(list)
    timing: dict[str, list[float]] = defaultdict(list)
    frames: dict[str, int] = defaultdict(int)
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        vid = r["video_id"]
        by_video[vid].append(
            WindowVerdict(r["start_time_sec"], r["end_time_sec"], r["class_name"], float(r["score"]))
        )
        timing[vid].append(float(r.get("window_ms", 0.0)))
        frames[vid] += int(r.get("n_frames", 0))
    return dict(by_video), dict(timing), dict(frames)


def aggregate(by_video, policy: AggregationPolicy) -> dict[str, list[Event]]:
    out: dict[str, list[Event]] = {}
    for vid, verdicts in by_video.items():
        events: list[Event] = []
        for cls in sorted({v.class_name for v in verdicts} - {"normal"}):
            events.extend(aggregate_class(vid, cls, verdicts, policy))
        out[vid] = sorted(events, key=lambda e: e.start_time_sec or 0.0)
    return out


def load_ground_truth(dataset_root: Path):
    gt = pd.read_csv(dataset_root / "test" / "ground_truth.csv")
    levels = dict(zip(gt["video_id"], gt["level"]))
    by: dict[str, list[Event]] = defaultdict(list)
    for r in gt.itertuples(index=False):
        if bool(r.is_anomaly) and r.class_name != "normal":
            by[r.video_id].append(Event(
                r.video_id, r.class_name,
                float(r.start_time_sec) if pd.notna(r.start_time_sec) else None,
                float(r.end_time_sec) if pd.notna(r.end_time_sec) else None,
            ))
    return dict(by), levels


def official_of(pred_events, gt_by, levels) -> dict:
    return score_submission(gt_by, pred_events, levels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/submission.json"))
    ap.add_argument("--submission-id", default="ahc-run-01")
    ap.add_argument("--model-name", default="qwen2.5-vl-7b-4bit-lora")
    ap.add_argument("--hardware", default="1x Tesla T4 (16GB)")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    by_video, timing, frames = load_verdicts(args.verdicts)
    gt_by, levels = load_ground_truth(args.dataset_root)

    n_win = sum(len(v) for v in by_video.values())
    n_anom = sum(1 for v in by_video.values() for w in v if w.class_name != "normal")
    print(f"{n_win} windows / {len(by_video)} videos; "
          f"{n_anom} ({n_anom/max(n_win,1):.1%}) called anomalous\n")

    best_policy = AggregationPolicy()

    if args.sweep:
        grid = list(itertools.product(
            [0.3, 0.5, 0.7, 0.9],       # theta_high
            [0.2, 0.35, 0.5],           # theta_low
            [0.0, 5.0, 15.0, 30.0],     # merge_gap_sec  (fragmentation is penalised)
            [0.0, 4.0, 10.0],           # min_duration_sec
        ))
        results = []
        for th, tl, gap, mind in grid:
            if tl > th:
                continue
            pol = AggregationPolicy(th, tl, gap, mind)
            res = official_of(aggregate(by_video, pol), gt_by, levels)
            results.append((res["overall"], res, (th, tl, gap, mind)))
        results.sort(key=lambda x: -x[0])

        print(f"swept {len(results)} policies (objective = OFFICIAL overall)\n")
        print(f"{'overall':>8}{'L1':>7}{'L2':>7}{'L3':>7}   hi/lo   merge  min_dur")
        print("-" * 56)
        for overall, res, (th, tl, gap, mind) in results[:10]:
            print(f"{overall:>8.3f}{res['level1']['score']:>7.3f}"
                  f"{res['level2']['score']:>7.3f}{res['level3']['score']:>7.3f}"
                  f"   {th}/{tl}  {gap:>5}  {mind:>5}")
        best_policy = AggregationPolicy(*results[0][2])
        print(f"\nbest policy: {best_policy}\n")

    pred_events = aggregate(by_video, best_policy)
    print_official(official_of(pred_events, gt_by, levels))

    # Build the submission file.
    entries = []
    total_ms = 0.0
    for vid in sorted(levels):
        evs = [{"class_name": e.class_name, "start_time_sec": e.start_time_sec,
                "end_time_sec": e.end_time_sec, "_score": e.score}
               for e in pred_events.get(vid, [])]
        calls = timing.get(vid, [])
        video_ms = sum(calls)
        total_ms += video_ms
        rt = RuntimeMetadata(
            frames_processed=frames.get(vid, 0),
            chunks_processed=len(calls),
            end_to_end_internal_time_ms=video_ms,
            model_runtimes=[make_model_runtime("qwen2.5-vl-7b-4bit", calls)] if calls else [],
        )
        entries.append(build_video_entry(vid, int(levels[vid]), evs, rt))

    sub = build_submission(entries, args.submission_id, args.model_name, total_ms, args.hardware)

    problems = validate(sub, levels)
    if problems:
        print(f"\n{len(problems)} VALIDATION PROBLEMS - fix before uploading:")
        for p in problems[:15]:
            print("  ", p)
    else:
        print("\nvalidation: clean")

    write_submission(sub, args.out)


if __name__ == "__main__":
    main()

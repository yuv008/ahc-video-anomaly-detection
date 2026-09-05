"""Aggregate per-window verdicts into events and score them - locally, no GPU.

The GPU pass (scripts/colab_infer.py) emits raw per-window verdicts. Everything after that
is cheap CPU work, so aggregator thresholds can be swept here for free instead of paying for
another inference run per candidate setting. That is the whole reason the two stages are
split.

Also supports --sweep, which grids over hysteresis and merge parameters and reports the best
by F1. Fit these on the SYNTHESIZED dev set (data/processed/synth), not on the 34-video
public test set - one public-test video is ~3% of the score and it is not the private set
(docs/architecture.md 3.4, 8.4).

Usage:
    python scripts/score_verdicts.py --verdicts window_verdicts.jsonl
    python scripts/score_verdicts.py --verdicts window_verdicts.jsonl --sweep
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from ahc_vad.eval.evaluate import evaluate, load_predictions, print_report
from ahc_vad.infer.aggregate import AggregationPolicy, WindowVerdict, aggregate_class


def load_verdicts(path: Path) -> dict[str, list[WindowVerdict]]:
    by_video: dict[str, list[WindowVerdict]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_video.setdefault(r["video_id"], []).append(
                WindowVerdict(r["start_time_sec"], r["end_time_sec"],
                              r["class_name"], float(r["score"]))
            )
    return by_video


def aggregate_all(by_video, policy: AggregationPolicy) -> list[dict]:
    rows = []
    for video_id, verdicts in by_video.items():
        classes = {v.class_name for v in verdicts} - {"normal"}
        for cls in sorted(classes):
            for ev in aggregate_class(video_id, cls, verdicts, policy):
                rows.append({
                    "video_id": ev.video_id, "class_name": ev.class_name,
                    "start_time_sec": ev.start_time_sec, "end_time_sec": ev.end_time_sec,
                    "score": ev.score,
                })
    return rows


def score(rows, dataset_root: Path, tmp: Path, iou=(0.1, 0.5)):
    pd.DataFrame(rows, columns=["video_id", "class_name", "start_time_sec",
                                "end_time_sec", "score"]).to_csv(tmp, index=False)
    return evaluate(dataset_root, load_predictions(tmp), tuple(iou))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/preds.csv"))
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    by_video = load_verdicts(args.verdicts)
    n_win = sum(len(v) for v in by_video.values())
    n_anom = sum(1 for v in by_video.values() for w in v if w.class_name != "normal")
    print(f"{n_win} windows across {len(by_video)} videos; "
          f"{n_anom} ({n_anom/max(n_win,1):.1%}) called anomalous\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if not args.sweep:
        rows = aggregate_all(by_video, AggregationPolicy())
        results = score(rows, args.dataset_root, args.out)
        print_report(results)
        print(f"\nwrote {args.out}")
        return

    grid = list(itertools.product(
        [0.3, 0.5, 0.7, 0.9],      # theta_high
        [0.2, 0.35, 0.5],          # theta_low
        [0.0, 5.0, 15.0, 30.0],    # merge_gap_sec
        [0.0, 4.0, 10.0],          # min_duration_sec
    ))
    print(f"sweeping {len(grid)} policies ...\n")

    best = []
    for th, tl, gap, mind in grid:
        if tl > th:
            continue
        pol = AggregationPolicy(theta_high=th, theta_low=tl,
                                merge_gap_sec=gap, min_duration_sec=mind)
        rows = aggregate_all(by_video, pol)
        r = score(rows, args.dataset_root, Path(str(args.out) + ".tmp"))
        m = r["by_iou"][0.1]["overall"]
        best.append((m.f1, m.precision, m.recall, len(rows), (th, tl, gap, mind)))

    best.sort(reverse=True)
    print(f"{'F1':>7}{'P':>8}{'R':>8}{'events':>8}   theta_hi/lo  merge  min_dur")
    print("-" * 62)
    for f1, p, rc, n, (th, tl, gap, mind) in best[:12]:
        print(f"{f1:>7.3f}{p:>8.3f}{rc:>8.3f}{n:>8}   {th}/{tl}   {gap:>5}  {mind:>5}")

    f1, p, rc, n, (th, tl, gap, mind) = best[0]
    print(f"\nbest: theta_high={th} theta_low={tl} merge_gap={gap} min_duration={mind}")
    print(f"  -> F1={f1:.3f} P={p:.3f} R={rc:.3f} over {n} events")
    rows = aggregate_all(by_video, AggregationPolicy(th, tl, gap, mind))
    print_report(score(rows, args.dataset_root, args.out))
    Path(str(args.out) + ".tmp").unlink(missing_ok=True)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

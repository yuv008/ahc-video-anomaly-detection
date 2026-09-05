"""Score predictions against test/ground_truth.csv.

PREDICTION FORMAT - one row per DETECTED EVENT (not per video):

    video_id,class_name,start_time_sec,end_time_sec,score
    T026,traffic_accident,12.0,38.5,0.91
    T026,fighting_or_violence,148.0,205.0,0.77

  - A video with no detected events contributes NO rows. That is the correct output for
    T029/T030 (240s videos containing nothing), see docs/architecture.md 1.7.
  - `score` is optional (defaults to 1.0) but improves matching order and lets you sweep
    an operating point.
  - `start_time_sec`/`end_time_sec` may be blank for Level 1 videos, where ground truth is
    itself untimed and matching falls back to class equality.
  - Rows with class_name=normal are ignored - "normal" is the absence of an event.

Reported metrics:
  - Event-level precision/recall/F1 at one or more temporal IoU thresholds, overall and
    broken down by task level (1/2/3), since those are very different problems.
  - Video-level anomaly detection: did we correctly decide whether each video contains any
    anomaly at all. False alarm rate here is measured against the 6 truly-normal test videos.
  - Per-class event recall, to expose which classes are being missed.

The private leaderboard formula is not published (docs/architecture.md 8.3) - these are
reasoned interpretations, and the IoU threshold is a CLI flag so the rule can move quickly.

Usage:
    python -m ahc_vad.eval.evaluate --predictions preds.csv --dataset-root data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ahc_vad.eval.matching import Event, match_events, temporal_iou
from ahc_vad.schema import CLASS_NAMES

DEFAULT_IOU_THRESHOLDS = (0.1, 0.3, 0.5)


def _to_float(value) -> float | None:
    return float(value) if pd.notna(value) else None


def load_ground_truth(dataset_root: Path) -> tuple[list[Event], pd.DataFrame]:
    """Return (anomalous GT events, per-video metadata with level + has_anomaly)."""
    gt = pd.read_csv(Path(dataset_root) / "test" / "ground_truth.csv")

    events = [
        Event(
            video_id=r.video_id,
            class_name=r.class_name,
            start_time_sec=_to_float(r.start_time_sec),
            end_time_sec=_to_float(r.end_time_sec),
        )
        for r in gt.itertuples(index=False)
        if bool(r.is_anomaly) and r.class_name != "normal"
    ]

    video_meta = (
        gt.groupby("video_id")
        .agg(level=("level", "max"), has_anomaly=("is_anomaly", "any"))
        .reset_index()
    )
    return events, video_meta


def load_predictions(path: Path) -> list[Event]:
    df = pd.read_json(path, lines=True) if path.suffix == ".jsonl" else pd.read_csv(path)
    if df.empty:
        return []

    for col in ("start_time_sec", "end_time_sec"):
        if col not in df.columns:
            df[col] = pd.NA
    if "score" not in df.columns:
        df["score"] = 1.0

    # Tolerate the legacy one-row-per-video format that carried an is_anomaly flag.
    if "is_anomaly" in df.columns:
        df = df[df["is_anomaly"].astype(bool)]

    df = df[df["class_name"] != "normal"]

    return [
        Event(
            video_id=r.video_id,
            class_name=r.class_name,
            start_time_sec=_to_float(r.start_time_sec),
            end_time_sec=_to_float(r.end_time_sec),
            score=float(r.score) if pd.notna(r.score) else 1.0,
        )
        for r in df.itertuples(index=False)
    ]


def video_level_metrics(
    gt_meta: pd.DataFrame, pred_events: list[Event]
) -> dict:
    """Did we correctly decide whether each video contains ANY anomaly?"""
    predicted_anomalous = {e.video_id for e in pred_events}

    tp = fp = fn = tn = 0
    for r in gt_meta.itertuples(index=False):
        predicted = r.video_id in predicted_anomalous
        if r.has_anomaly and predicted:
            tp += 1
        elif r.has_anomaly and not predicted:
            fn += 1
        elif not r.has_anomaly and predicted:
            fp += 1
        else:
            tn += 1

    n_normal = fp + tn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        # Fraction of genuinely-normal videos we wrongly alarmed on. Unlike 1-precision this
        # stays meaningful when nothing was predicted at all.
        "false_alarm_rate": fp / n_normal if n_normal else 0.0,
        "n_normal_videos": n_normal,
    }


def evaluate(
    dataset_root: Path,
    pred_events: list[Event],
    iou_thresholds=DEFAULT_IOU_THRESHOLDS,
) -> dict:
    gt_events, gt_meta = load_ground_truth(dataset_root)
    level_of = dict(zip(gt_meta["video_id"], gt_meta["level"], strict=True))

    results: dict = {
        "n_gt_events": len(gt_events),
        "n_pred_events": len(pred_events),
        "video_level": video_level_metrics(gt_meta, pred_events),
        "by_iou": {},
    }

    for tau in iou_thresholds:
        overall = match_events(gt_events, pred_events, iou_threshold=tau)

        by_level = {}
        for level in sorted(gt_meta["level"].unique()):
            lvl_gt = [e for e in gt_events if level_of.get(e.video_id) == level]
            lvl_pred = [e for e in pred_events if level_of.get(e.video_id) == level]
            by_level[int(level)] = match_events(lvl_gt, lvl_pred, iou_threshold=tau)

        matched_ious = [
            temporal_iou(p, g) for p, g in overall.true_positives if p.is_timed and g.is_timed
        ]

        results["by_iou"][tau] = {
            "overall": overall,
            "by_level": by_level,
            "mean_matched_iou": sum(matched_ious) / len(matched_ious) if matched_ious else None,
        }

    primary = results["by_iou"][iou_thresholds[0]]["overall"]
    results["per_class_recall"] = {}
    for cls in CLASS_NAMES:
        if cls == "normal":
            continue
        n_gt = sum(1 for e in gt_events if e.class_name == cls)
        n_hit = sum(1 for _, g in primary.true_positives if g.class_name == cls)
        if n_gt:
            results["per_class_recall"][cls] = (n_hit, n_gt)

    return results


def print_report(results: dict) -> None:
    v = results["video_level"]
    print(f"Ground-truth events: {results['n_gt_events']}   Predicted events: {results['n_pred_events']}")
    print()
    print("VIDEO-LEVEL (does the video contain any anomaly?)")
    print(f"  precision={v['precision']:.3f}  recall={v['recall']:.3f}")
    print(
        f"  false alarms: {v['fp']}/{v['n_normal_videos']} normal videos "
        f"(rate={v['false_alarm_rate']:.3f})   missed: {v['fn']}"
    )
    print()

    print("EVENT-LEVEL")
    for tau, block in results["by_iou"].items():
        o = block["overall"]
        iou_txt = (
            f"  mean matched IoU={block['mean_matched_iou']:.3f}"
            if block["mean_matched_iou"] is not None
            else ""
        )
        print(
            f"  IoU>={tau}:  P={o.precision:.3f} R={o.recall:.3f} F1={o.f1:.3f}  "
            f"(TP={len(o.true_positives)} FP={len(o.false_positives)} FN={len(o.false_negatives)}){iou_txt}"
        )
        for level, m in block["by_level"].items():
            print(
                f"      level {level}:  P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}  "
                f"(TP={len(m.true_positives)} FP={len(m.false_positives)} FN={len(m.false_negatives)})"
            )
    print()

    print("PER-CLASS EVENT RECALL (at lowest IoU threshold)")
    for cls, (hit, total) in results["per_class_recall"].items():
        bar = "#" * int(20 * hit / total) if total else ""
        print(f"  {cls:35s} {hit:3d}/{total:<3d} {bar}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_IOU_THRESHOLDS),
        help="Temporal IoU thresholds for event matching (Levels 2-3).",
    )
    args = parser.parse_args()

    pred_events = load_predictions(args.predictions)
    results = evaluate(args.dataset_root, pred_events, tuple(args.iou_thresholds))
    print_report(results)


if __name__ == "__main__":
    main()

"""The organizers' scoring rules, as published in the Submission Format doc.

Implemented separately from eval/evaluate.py because the official metric is NOT global
event-level P/R/F1 - it is per-video, averaged, with different rules per level. Tuning
against the wrong objective is worse than not tuning, so this is what thresholds should be
fitted on.

Rules taken directly from the doc:

  Level 1  - pooled across all Level-1 videos:
             half anomaly-versus-normal accuracy, half class accuracy.
             Repeating a class on one video earns nothing extra (so dedupe).
  Level 2/3 - scored per video, then averaged:
             * ground truth normal  -> predict nothing = 1, predict anything = 0
             * ground truth has events -> weighted mix of (did you alert),
               (matched events), (how well timings line up). Timing weighs more at L3.
  Matching  - an event counts only when the class is right AND temporal IoU >= 0.5.
             Several fragments for one real event do not help: at most one matches and
             the rest count against you.

UNKNOWN, and therefore explicit here: the doc does not publish the exact weights of the
three components for Levels 2/3. `Weights` below is a reasoned default with timing raised
at Level 3, exposed so it can be corrected the moment the real numbers are known. Treat
absolute scores as indicative and rankings between configs as the reliable signal.
"""

from __future__ import annotations

from dataclasses import dataclass

from ahc_vad.eval.matching import Event, temporal_iou

OFFICIAL_IOU = 0.5  # stated in the doc, not a tunable


@dataclass(frozen=True)
class Weights:
    """Component weights for a Level-2/3 video that genuinely contains events."""

    alert: float = 0.2      # did you raise anything at all
    match: float = 0.5      # F1 over correctly matched events
    timing: float = 0.3     # how well matched intervals line up

    @staticmethod
    def for_level(level: int) -> Weights:
        # "Timing weighs more at Level 3."
        return Weights(0.2, 0.4, 0.4) if level >= 3 else Weights(0.2, 0.5, 0.3)


def match_one_video(gt: list[Event], pred: list[Event], iou_threshold: float = OFFICIAL_IOU):
    """Greedy score-ordered 1-to-1 matching within a single video.

    Extra fragments overlapping an already-matched event become false positives, which is
    exactly the doc's "at most one can match, and the rest count against you".
    """
    matched_gt: set[int] = set()
    pairs: list[tuple[Event, Event, float]] = []
    false_positives: list[Event] = []

    for p in sorted(pred, key=lambda e: -e.score):
        best, best_iou = None, 0.0
        for g in gt:
            if id(g) in matched_gt or g.class_name != p.class_name:
                continue
            iou = temporal_iou(p, g)
            if iou >= iou_threshold and iou > best_iou:
                best, best_iou = g, iou
        if best is not None:
            matched_gt.add(id(best))
            pairs.append((p, best, best_iou))
        else:
            false_positives.append(p)

    missed = [g for g in gt if id(g) not in matched_gt]
    return pairs, false_positives, missed


def score_level1(gt_by_video: dict, pred_by_video: dict, video_ids: list[str]) -> dict:
    """Pooled: half anomaly-vs-normal accuracy, half class accuracy."""
    n_binary_correct = 0
    n_class_correct = 0
    n_class_total = 0

    for vid in video_ids:
        gt_events = gt_by_video.get(vid, [])
        pred_events = pred_by_video.get(vid, [])
        gt_is_anom = len(gt_events) > 0
        pred_is_anom = len(pred_events) > 0

        if gt_is_anom == pred_is_anom:
            n_binary_correct += 1

        # Class accuracy is only meaningful where the truth is an anomaly. Repeating a
        # class earns nothing extra, so compare against the SET of predicted classes.
        if gt_is_anom:
            n_class_total += 1
            gt_classes = {e.class_name for e in gt_events}
            pred_classes = {e.class_name for e in pred_events}
            if gt_classes & pred_classes:
                n_class_correct += 1

    n = len(video_ids)
    acc_binary = n_binary_correct / n if n else 0.0
    acc_class = n_class_correct / n_class_total if n_class_total else 0.0
    return {
        "n_videos": n,
        "acc_binary": acc_binary,
        "acc_class": acc_class,
        "score": 0.5 * acc_binary + 0.5 * acc_class,
    }


def score_video_l23(gt: list[Event], pred: list[Event], level: int) -> dict:
    """One Level-2/3 video."""
    # "Ground truth is normal -> you predict nothing = 1, you predict anything = 0."
    # This is why false alarms are expensive: a single spurious event zeroes the video.
    if not gt:
        return {
            "score": 1.0 if not pred else 0.0,
            "is_normal": True,
            "alert": float(bool(pred)),
            "match_f1": 0.0,
            "timing": 0.0,
        }

    pairs, fps, missed = match_one_video(gt, pred)

    alert = 1.0 if pred else 0.0
    tp = len(pairs)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    match_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    timing = sum(iou for _, _, iou in pairs) / tp if tp else 0.0

    w = Weights.for_level(level)
    return {
        "score": w.alert * alert + w.match * match_f1 + w.timing * timing,
        "is_normal": False,
        "alert": alert,
        "match_f1": match_f1,
        "timing": timing,
        "tp": tp,
        "fp": len(fps),
        "fn": len(missed),
    }


def score_submission(gt_by_video: dict, pred_by_video: dict, level_by_video: dict) -> dict:
    """Full official score: Level 1 pooled, Levels 2 and 3 averaged per video."""
    l1_ids = [v for v, lvl in level_by_video.items() if lvl == 1]
    out: dict = {"level1": score_level1(gt_by_video, pred_by_video, l1_ids)}

    for level in (2, 3):
        ids = [v for v, lvl in level_by_video.items() if lvl == level]
        per_video = [
            score_video_l23(gt_by_video.get(v, []), pred_by_video.get(v, []), level)
            for v in ids
        ]
        scores = [p["score"] for p in per_video]
        out[f"level{level}"] = {
            "n_videos": len(ids),
            "score": sum(scores) / len(scores) if scores else 0.0,
            "per_video": dict(zip(ids, per_video, strict=True)),
            "n_normal_zeroed": sum(
                1 for p in per_video if p["is_normal"] and p["score"] == 0.0
            ),
        }

    present = [out[k]["score"] for k in ("level1", "level2", "level3") if out[k]["n_videos"]]
    out["overall"] = sum(present) / len(present) if present else 0.0
    return out


def print_official(res: dict) -> None:
    l1 = res["level1"]
    print(f"LEVEL 1  ({l1['n_videos']} videos)   score={l1['score']:.3f}")
    print(f"  anomaly-vs-normal accuracy {l1['acc_binary']:.3f}   class accuracy {l1['acc_class']:.3f}")
    for level in (2, 3):
        b = res[f"level{level}"]
        if not b["n_videos"]:
            continue
        real = [p for p in b["per_video"].values() if not p["is_normal"]]
        norm = [p for p in b["per_video"].values() if p["is_normal"]]
        print(f"\nLEVEL {level}  ({b['n_videos']} videos)   score={b['score']:.3f}")
        if real:
            print(f"  event videos ({len(real)}): "
                  f"alert={sum(p['alert'] for p in real)/len(real):.2f} "
                  f"match_f1={sum(p['match_f1'] for p in real)/len(real):.2f} "
                  f"timing={sum(p['timing'] for p in real)/len(real):.2f}")
        if norm:
            print(f"  normal videos ({len(norm)}): {b['n_normal_zeroed']} zeroed by false alarms")
    print(f"\nOVERALL {res['overall']:.3f}   (mean of level scores)")

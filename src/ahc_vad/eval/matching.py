"""Event-level matching between predicted and ground-truth events.

The task is multi-event and multi-class per video (docs/architecture.md 1.7): a single test
video can contain several events of different classes, and some videos contain none. So
scoring is temporal-action-detection style - match predicted events to ground-truth events,
then count TP/FP/FN - not a per-video classification.

Two matching regimes, because ground truth is not uniform:

  - TIMED ground truth (test Levels 2-3): a prediction matches a GT event when the class
    is equal AND temporal IoU >= threshold.
  - UNTIMED ground truth (test Level 1, where start/end are blank): temporal IoU is
    undefined, so a prediction matches on class equality alone, scoped to that video.

Matching is greedy by descending prediction score, and each GT event can be matched at most
once, so duplicate predictions on the same event count as false positives.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    video_id: str
    class_name: str
    start_time_sec: float | None = None
    end_time_sec: float | None = None
    score: float = 1.0

    @property
    def is_timed(self) -> bool:
        return self.start_time_sec is not None and self.end_time_sec is not None


@dataclass
class MatchResult:
    true_positives: list[tuple[Event, Event]] = field(default_factory=list)  # (pred, gt)
    false_positives: list[Event] = field(default_factory=list)
    false_negatives: list[Event] = field(default_factory=list)

    @property
    def precision(self) -> float:
        n_pred = len(self.true_positives) + len(self.false_positives)
        return len(self.true_positives) / n_pred if n_pred else 0.0

    @property
    def recall(self) -> float:
        n_gt = len(self.true_positives) + len(self.false_negatives)
        return len(self.true_positives) / n_gt if n_gt else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def temporal_iou(a: Event, b: Event) -> float:
    """IoU of two time intervals. Returns 0.0 if either is untimed."""
    if not (a.is_timed and b.is_timed):
        return 0.0
    inter = max(0.0, min(a.end_time_sec, b.end_time_sec) - max(a.start_time_sec, b.start_time_sec))
    union = max(a.end_time_sec, b.end_time_sec) - min(a.start_time_sec, b.start_time_sec)
    return inter / union if union > 0 else 0.0


def match_events(
    gt_events: list[Event],
    pred_events: list[Event],
    iou_threshold: float = 0.1,
) -> MatchResult:
    """Greedy score-ordered matching, scoped per (video_id, class_name)."""
    result = MatchResult()

    gt_by_key: dict[tuple[str, str], list[Event]] = {}
    for e in gt_events:
        gt_by_key.setdefault((e.video_id, e.class_name), []).append(e)

    matched: set[int] = set()  # ids of GT events already consumed

    for pred in sorted(pred_events, key=lambda e: -e.score):
        candidates = gt_by_key.get((pred.video_id, pred.class_name), [])
        best_gt, best_iou = None, -1.0

        for gt in candidates:
            if id(gt) in matched:
                continue
            if gt.is_timed and pred.is_timed:
                iou = temporal_iou(pred, gt)
                if iou >= iou_threshold and iou > best_iou:
                    best_gt, best_iou = gt, iou
            else:
                # Untimed GT (Level 1): class equality within the video is the whole test.
                # Prefer an exact-but-unscored match over nothing; IoU stays undefined.
                if best_gt is None:
                    best_gt, best_iou = gt, 0.0

        if best_gt is not None:
            matched.add(id(best_gt))
            result.true_positives.append((pred, best_gt))
        else:
            result.false_positives.append(pred)

    for e in gt_events:
        if id(e) not in matched:
            result.false_negatives.append(e)

    return result

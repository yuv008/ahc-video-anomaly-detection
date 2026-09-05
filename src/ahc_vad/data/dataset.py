"""Load the AHC train/test dataset layout described in docs/dataset.md.

train/
  <class_name>/
    videos/*.mp4
    videos.csv
    ground_truth.csv
test/
  videos/*.mp4
  videos.csv
  ground_truth.csv

Schema confirmed against the real download (2026-09-04):
  - videos.csv columns: video_id, filename - `filename` is already relative to the split
    dir, e.g. "videos/T001.mp4" (includes the videos/ prefix, don't add it again).
  - ground_truth.csv has `level` in test/ only - it's absent from every train/<class>/
    ground_truth.csv, so `level` is optional and None for train events.
  - is_anomaly is lowercase true/false; pandas parses it to bool natively.
  - video_id repeats in test/ground_truth.csv for multi-event videos (18 of 34 test videos
    have >1 row); train ground_truth.csv is one row per video (no repeats observed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ahc_vad.schema import CLASS_NAMES, GROUND_TRUTH_COLUMNS


@dataclass(frozen=True)
class Event:
    video_id: str
    video_path: Path
    is_anomaly: bool
    class_name: str
    start_time_sec: float | None
    end_time_sec: float | None
    description_summary: str | None
    level: int | None = None  # only populated for test/ events


def _load_split_dir(split_dir: Path) -> pd.DataFrame:
    """Load one directory that directly contains videos.csv + ground_truth.csv + videos/."""
    videos = pd.read_csv(split_dir / "videos.csv")
    gt = pd.read_csv(split_dir / "ground_truth.csv")

    missing = set(GROUND_TRUTH_COLUMNS) - set(gt.columns)
    if missing:
        raise ValueError(f"{split_dir}/ground_truth.csv missing columns: {missing}")
    if "level" not in gt.columns:
        gt = gt.copy()
        gt["level"] = pd.NA

    merged = gt.merge(videos, on="video_id", how="left", validate="many_to_one")
    unresolved = merged.loc[merged["filename"].isna(), "video_id"].unique()
    if len(unresolved):
        raise ValueError(f"video_id(s) in ground_truth.csv missing from videos.csv: {unresolved}")

    merged["video_path"] = merged["filename"].apply(lambda f: str(split_dir / f))
    return merged


def load_train_events(dataset_root: Path) -> list[Event]:
    """Walk train/<class_name>/{videos.csv,ground_truth.csv,videos/} and flatten to events."""
    train_root = Path(dataset_root) / "train"
    if not train_root.exists():
        raise FileNotFoundError(
            f"{train_root} not found. Run scripts/download_dataset.py first."
        )

    events: list[Event] = []
    for class_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        if class_dir.name not in CLASS_NAMES:
            continue  # ignore stray folders
        df = _load_split_dir(class_dir)
        events.extend(_rows_to_events(df))
    return events


def load_test_events(dataset_root: Path) -> list[Event]:
    test_root = Path(dataset_root) / "test"
    if not test_root.exists():
        raise FileNotFoundError(
            f"{test_root} not found. Run scripts/download_dataset.py first."
        )
    df = _load_split_dir(test_root)
    return _rows_to_events(df)


def _rows_to_events(df: pd.DataFrame) -> list[Event]:
    events = []
    for row in df.itertuples(index=False):
        events.append(
            Event(
                video_id=row.video_id,
                video_path=Path(row.video_path),
                level=(int(row.level) if pd.notna(row.level) else None),
                is_anomaly=bool(row.is_anomaly),
                class_name=row.class_name,
                start_time_sec=(
                    float(row.start_time_sec) if pd.notna(row.start_time_sec) else None
                ),
                end_time_sec=(
                    float(row.end_time_sec) if pd.notna(row.end_time_sec) else None
                ),
                description_summary=(
                    row.description_summary if pd.notna(row.description_summary) else None
                ),
            )
        )
    return events


def events_to_dataframe(events: list[Event]) -> pd.DataFrame:
    return pd.DataFrame([e.__dict__ for e in events])

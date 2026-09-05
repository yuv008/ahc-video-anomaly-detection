"""Pre-extract window frames locally as JPEGs, for shipping to a remote GPU.

Why not just upload the videos: measured upload to the Colab session is ~1.1 MB/s, so the
15GB dataset is out of the question. Frames are far smaller - the whole public test set is
~270MB as 8-frame windows versus 1.5GB of video.

It is also better engineering than decoding on the GPU box:
  - resolution is fixed HERE, at the same cap used at inference, so there is no train/serve
    skew (docs/architecture.md 5.5)
  - video decode is CPU-bound; doing it once locally stops the GPU waiting on it every epoch
  - windows are already labelled by temporal overlap (data/windows.py), so the remote side
    needs no dataset logic at all

Output layout:
    <out>/frames/<window_id>/f0.jpg .. fN.jpg
    <out>/index.jsonl        one row per window: id, class_name, video_id, start, end

Usage:
    python scripts/export_frames.py train --limit 3000 --out data/processed/export_train
    python scripts/export_frames.py test --out data/processed/export_test
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from ahc_vad.data.build_sft_dataset import balance_windows, build_windows
from ahc_vad.data.sampling import DEFAULT_MAX_TOKENS_PER_FRAME, sample_frames, sliding_windows


def _export_one(task):
    """Extract and write one window. Runs in a worker process."""
    wid, rec, frames_dir, num_frames, max_tokens, quality = task
    try:
        frames = sample_frames(
            rec["video_path"], num_frames, rec["start"], rec["end"],
            max_tokens_per_frame=max_tokens,
        )
    except Exception as e:
        return None, f"{wid}: {type(e).__name__}: {str(e)[:60]}"
    if not frames:
        return None, f"{wid}: no frames"

    wdir = Path(frames_dir) / wid
    wdir.mkdir(parents=True, exist_ok=True)
    for j, img in enumerate(frames):
        # optimize=True runs a second Huffman pass - measurably slow at this volume and
        # worth only a few percent of size. Not worth it for tens of thousands of frames.
        img.convert("RGB").save(wdir / f"f{j}.jpg", "JPEG", quality=quality)

    return {
        "id": wid, "class_name": rec.get("class_name"), "video_id": rec["video_id"],
        "start": rec["start"], "end": rec["end"], "n_frames": len(frames),
    }, None


def export(records, out_dir: Path, num_frames: int, max_tokens: int, quality: int,
           workers: int) -> None:
    """Parallel export. Video decode is CPU-bound and each window is independent, so this
    scales nearly linearly with cores - the serial version was the bottleneck, not disk."""
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (f"w{i:06d}", {**r, "video_path": str(r["video_path"])},
         str(frames_dir), num_frames, max_tokens, quality)
        for i, r in enumerate(records)
    ]

    written, errors = [], []
    if workers <= 1:
        for t in tqdm(tasks, desc=f"export {out_dir.name}"):
            rec, err = _export_one(t)
            (written if rec else errors).append(rec or err)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for rec, err in tqdm(pool.map(_export_one, tasks, chunksize=8),
                                 total=len(tasks), desc=f"export {out_dir.name}"):
                if rec:
                    written.append(rec)
                else:
                    errors.append(err)

    with (out_dir / "index.jsonl").open("w", encoding="utf-8") as idx:
        for rec in written:
            idx.write(json.dumps(rec) + "\n")

    n_bytes = sum(p.stat().st_size for p in frames_dir.rglob("*.jpg"))
    print(f"\nwrote {len(written)} windows to {out_dir}  ({n_bytes/1e6:.0f} MB)")
    if errors:
        print(f"  {len(errors)} skipped, first few: {errors[:3]}")


def collect_train(dataset_root: Path, limit: int, seed: int, normal_ratio: float):
    windows = build_windows(dataset_root)
    print(f"raw windows: {len(windows)}")
    rng = random.Random(seed)
    windows = balance_windows(windows, rng, normal_ratio=normal_ratio)
    if limit and len(windows) > limit:
        # Stratified trim: keep class proportions rather than truncating arbitrarily.
        by_cls: dict[str, list] = {}
        for w in windows:
            by_cls.setdefault(w.class_name, []).append(w)
        keep, total = [], len(windows)
        for cls, items in by_cls.items():
            n = max(1, round(limit * len(items) / total))
            keep.extend(rng.sample(items, min(n, len(items))))
        rng.shuffle(keep)
        windows = keep[:limit]

    print(f"exporting {len(windows)} windows")
    for cls, n in Counter(w.class_name for w in windows).most_common():
        print(f"  {cls:35s} {n:5d}")

    return [{
        "video_path": w.video_path, "video_id": w.video_id,
        "start": w.start_time_sec, "end": w.end_time_sec, "class_name": w.class_name,
    } for w in windows]


def collect_test(dataset_root: Path, window_sec: float, stride_sec: float):
    test_root = dataset_root / "test"
    videos = pd.read_csv(test_root / "videos.csv")
    records = []
    for row in videos.itertuples(index=False):
        path = test_root / row.filename
        cap = cv2.VideoCapture(str(path))
        fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration = n / fps if fps else 0.0
        for start, end in sliding_windows(duration, window_sec, stride_sec):
            records.append({"video_path": path, "video_id": row.video_id,
                            "start": start, "end": end, "class_name": None})
    print(f"test: {len(videos)} videos -> {len(records)} windows")
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "test"])
    ap.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS_PER_FRAME)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--limit", type=int, default=3000, help="train only")
    ap.add_argument("--normal-ratio", type=float, default=1.5, help="train only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="Parallel decode workers. Video decode is CPU-bound.")
    ap.add_argument("--window-sec", type=float, default=4.0, help="test only")
    ap.add_argument("--stride-sec", type=float, default=4.0, help="test only")
    args = ap.parse_args()

    if args.split == "train":
        records = collect_train(args.dataset_root, args.limit, args.seed, args.normal_ratio)
    else:
        records = collect_test(args.dataset_root, args.window_sec, args.stride_sec)

    export(records, args.out, args.num_frames, args.max_tokens, args.quality,
           args.workers)


if __name__ == "__main__":
    main()

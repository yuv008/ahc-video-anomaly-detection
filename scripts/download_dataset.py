"""Download the AHC hackathon train/test dataset from a Google Drive mirror.

Usage:
    python scripts/download_dataset.py [--mirror 1-5] [--dest data/raw]

Mirrors are identical copies of the same train/test pack (~15-17 GB). Pick
whichever is fastest/least rate-limited for you.
"""

import argparse
import sys
from pathlib import Path

MIRRORS = {
    1: "https://drive.google.com/drive/folders/1sEFKR7ctd5GfFw-nMlYd_MnTw1VVYz9K",
    2: "https://drive.google.com/drive/folders/13E_CePn14lcbwMA_yZEiHpAVx6i09UIG",
    3: "https://drive.google.com/drive/folders/13V8JqgZRMzn2TCF0HTsCqVgUH0UOMmpb",
    4: "https://drive.google.com/drive/folders/1fS_i7QKXRDI6mnaI6UWqYzKSOYWG8rFv",
    5: "https://drive.google.com/drive/folders/1efhUZhB6Kyvpw3RulZJSwd0brb8KhuZf",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", type=int, default=1, choices=sorted(MIRRORS))
    parser.add_argument("--dest", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required: pip install gdown")

    args.dest.mkdir(parents=True, exist_ok=True)
    url = MIRRORS[args.mirror]
    print(f"Downloading mirror {args.mirror} -> {args.dest}")
    # use_cookies=True (gdown default) reads ~/.cache/gdown/cookies.txt if present, so
    # downloads count against your own Google account's quota instead of the shared
    # anonymous one. See docs/dataset.md for how to export that cookie file.
    gdown.download_folder(url=url, output=str(args.dest), quiet=False, use_cookies=True)

    expected = [args.dest / "train", args.dest / "test"]
    missing = [p for p in expected if not p.exists()]
    if missing:
        print(
            "Warning: expected folders not found after download: "
            f"{[str(p) for p in missing]}. Google Drive folder downloads can nest an extra "
            "directory level - check `data/raw/` and adjust `--dest` or move files if needed."
        )
    else:
        print("Done. Layout looks correct: data/raw/{train,test}/")


if __name__ == "__main__":
    main()

"""Pull the dataset onto the Colab VM and build window-level training data there.

Runs remotely via `colab exec`. Downloading from the Drive mirror inside Google's network
is far faster than re-uploading 15GB from a laptop, and it sidesteps the anonymous-download
quota that blocks gdown locally (docs/dataset.md).

Idempotent: skips the download if data/raw/{train,test} already exist, so it is safe to
re-run after a session restart.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/content/ahc")
RAW = ROOT / "data" / "raw"
MIRROR = "https://drive.google.com/drive/folders/1sEFKR7ctd5GfFw-nMlYd_MnTw1VVYz9K"


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
    if r.stdout:
        print(r.stdout[-3000:], flush=True)
    if r.returncode != 0:
        print(f"  stderr: {r.stderr[-2000:]}", flush=True)
    return r.returncode


def main():
    RAW.mkdir(parents=True, exist_ok=True)

    if (RAW / "train").exists() and (RAW / "test").exists():
        n_train = len(list((RAW / "train").glob("*/videos/*.mp4")))
        n_test = len(list((RAW / "test" / "videos").glob("*.mp4")))
        print(f"dataset already present: {n_train} train / {n_test} test videos", flush=True)
    else:
        sh("pip install -q gdown")
        import gdown

        print("downloading dataset from Drive mirror ...", flush=True)
        gdown.download_folder(url=MIRROR, output=str(RAW), quiet=False, use_cookies=False)

        # Drive folder downloads usually nest an extra directory level.
        for cand in RAW.iterdir():
            if cand.is_dir() and (cand / "train").exists():
                print(f"flattening {cand.name}/ ...", flush=True)
                for sub in ("train", "test"):
                    src, dst = cand / sub, RAW / sub
                    if src.exists() and not dst.exists():
                        src.rename(dst)
                break

    train_dir, test_dir = RAW / "train", RAW / "test"
    if not (train_dir.exists() and test_dir.exists()):
        print(f"FAILED: expected {train_dir} and {test_dir}", flush=True)
        print("contents:", [p.name for p in RAW.iterdir()][:20], flush=True)
        sys.exit(1)

    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    n_train = sum(len(list((train_dir / c / "videos").glob("*.mp4"))) for c in classes)
    n_test = len(list((test_dir / "videos").glob("*.mp4")))
    print(f"\nOK: {len(classes)} classes, {n_train} train videos, {n_test} test videos", flush=True)
    print("classes:", classes, flush=True)


main()

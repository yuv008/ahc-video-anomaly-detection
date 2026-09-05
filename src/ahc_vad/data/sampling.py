"""Frame sampling for VLM input.

Small VLMs consume a handful of frames per call, not the full clip. Real-time inference
means we sample a rolling window and slide it forward, rather than processing whole videos.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image


# Qwen2.5-VL vision geometry: patch 14, merge 2 -> each token covers 28x28 px, and image
# dims are rounded to multiples of 28. Capping tokens/frame is the single biggest latency
# lever measured on a T4: 8 frames at the uncapped default is 9,742 tokens / 17.8s per
# window, versus 2,190 tokens / 4.65s at 256 tok/frame (docs/architecture.md 5.1).
PATCH, MERGE = 14, 2
FACTOR = PATCH * MERGE
PX_PER_TOKEN = PATCH * PATCH * MERGE * MERGE

# Must match inference. Training at a different resolution than you deploy at is a silent
# train/serve skew - the model would learn from detail it never sees at runtime.
DEFAULT_MAX_TOKENS_PER_FRAME = 256


def resize_for_token_budget(img: Image.Image, max_tokens: int | None) -> Image.Image:
    """Downscale so the frame costs at most `max_tokens` vision tokens, keeping aspect."""
    if max_tokens is None:
        return img
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    beta = math.sqrt((w * h) / (max_tokens * PX_PER_TOKEN))
    if beta <= 1.0:
        return img  # already within budget
    new_w = max(FACTOR, math.floor(w / beta / FACTOR) * FACTOR)
    new_h = max(FACTOR, math.floor(h / beta / FACTOR) * FACTOR)
    return img.resize((new_w, new_h), Image.BILINEAR)


def sample_frames(
    video_path: str | Path,
    num_frames: int = 8,
    start_sec: float | None = None,
    end_sec: float | None = None,
    max_tokens_per_frame: int | None = DEFAULT_MAX_TOKENS_PER_FRAME,
) -> list[Image.Image]:
    """Evenly sample `num_frames` PIL frames from [start_sec, end_sec] (or the whole video).

    Uses decord when available (faster batch reads) and falls back to OpenCV otherwise.
    decord has no Windows wheel, so the fallback is what actually runs on this machine.
    """
    try:
        return _sample_decord(video_path, num_frames, start_sec, end_sec, max_tokens_per_frame)
    except ImportError:
        return _sample_cv2(video_path, num_frames, start_sec, end_sec, max_tokens_per_frame)


def _frame_indices(n_total: int, fps: float, num_frames: int,
                   start_sec: float | None, end_sec: float | None):
    start_idx = int((start_sec or 0.0) * fps)
    end_idx = int(end_sec * fps) if end_sec is not None else n_total - 1
    start_idx = max(0, min(start_idx, n_total - 1))
    end_idx = max(start_idx, min(end_idx, n_total - 1))
    return np.linspace(start_idx, end_idx, num=num_frames, dtype=int)


def _sample_decord(video_path, num_frames, start_sec, end_sec, max_tokens):
    import decord

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(video_path))
    idxs = _frame_indices(len(vr), vr.get_avg_fps(), num_frames, start_sec, end_sec)
    frames = vr.get_batch(idxs).asnumpy()
    return [resize_for_token_budget(Image.fromarray(f), max_tokens) for f in frames]


def _sample_cv2(video_path, num_frames, start_sec, end_sec, max_tokens):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        out = []
        for idx in _frame_indices(n_total, fps, num_frames, start_sec, end_sec):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            out.append(resize_for_token_budget(img, max_tokens))
        return out
    finally:
        cap.release()


def sliding_windows(duration_sec: float, window_sec: float = 4.0, stride_sec: float = 4.0):
    """Yield (start, end) windows covering a video for real-time-style rolling inference.

    Default stride EQUALS the window: no overlap. Measured on the public test set via
    scripts/oracle_ceiling.py, 50%% overlap is strictly dominated - identical F1 at every
    IoU threshold, WORSE boundary IoU (0.747 vs 0.840 @0.5), and twice the compute.
    Overlapping windows widen the aggregated span past the true event.
    """
    t = 0.0
    while t < duration_sec:
        end = min(t + window_sec, duration_sec)
        yield t, end
        if end >= duration_sec:
            break
        t += stride_sec

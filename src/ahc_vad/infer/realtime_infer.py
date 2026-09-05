"""Runtime detector: sliding-window inference producing an EVENT LIST per video.

Per the hackathon constraint, no hosted/large model calls happen here - this is the
inference path that has to stay cheap enough to run across many drone feeds. Hosted APIs
(NVIDIA NIM, Gemini - see docs/setup_guide.md) are for dev-time comparison and training-data
generation only.

Pipeline (docs/architecture.md 2):
    window sampler -> [gate] -> VLM -> temporal aggregator -> events

Stage 1 (gate) is NOT built, and measurement showed it is not needed for the single-feed
real-time claim: with stride = window = 4s, several configurations clear the 4s/window budget
on a T4 outright (docs/architecture.md 5.3). Its remaining value is multiplying feeds per GPU,
which is a throughput argument rather than a correctness one.

Model: Qwen3-VL-8B 4-bit LoRA (docs/architecture.md 0). At 8 frames it measures 4.64 s/window
= 116% of the real-time budget, so scored inference subsamples to 4 frames via --frames.

Usage:
    python -m ahc_vad.infer.realtime_infer \
        --model outputs/qwen3-vl-8b-lora --dataset-root data/raw --out preds.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from ahc_vad.data.build_sft_dataset import SYSTEM_PROMPT, USER_PROMPT
from ahc_vad.data.sampling import sample_frames, sliding_windows
from ahc_vad.data.windows import DWELL_CLASSES, LONG_SCALE
from ahc_vad.infer.aggregate import WindowVerdict, aggregate_video
from ahc_vad.infer.scoring import verdict_from_generation


def video_duration_sec(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n_frames / fps if fps else 0.0


def generate(model, processor, frames) -> tuple[str, float]:
    """One VLM call -> (class_name, score).

    Scores are read from the logits at the is_anomaly boolean position rather than invented,
    so the aggregator's hysteresis thresholds act on a real quantity (see infer/scoring.py).
    """
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [{"type": "image", "image": f} for f in frames]
            + [{"type": "text", "text": USER_PROMPT}],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated = out.sequences[0][inputs["input_ids"].shape[1] :]
    decoded = processor.decode(generated, skip_special_tokens=True)
    return verdict_from_generation(processor, generated.tolist(), out.scores, decoded)


def scan_scale(
    model,
    processor,
    video_path: Path,
    duration: float,
    num_frames: int,
    window_sec: float,
    stride_sec: float,
) -> list[WindowVerdict]:
    """Run the VLM across one temporal scale, returning the DENSE window stream.

    Every window is kept, including ones judged normal: normal windows score 0 for every
    class and are what close an open span in the aggregator. Dropping them merges distinct
    events into one and guts Level 2/3 recall.
    """
    verdicts: list[WindowVerdict] = []
    for start, end in sliding_windows(duration, window_sec=window_sec, stride_sec=stride_sec):
        frames = sample_frames(video_path, num_frames, start, end)
        class_name, score = generate(model, processor, frames)
        verdicts.append(WindowVerdict(start, end, class_name, score))
    return verdicts


def run_video(
    model,
    processor,
    video_path: Path,
    video_id: str,
    num_frames: int,
    window_sec: float,
    stride_sec: float,
    multi_scale: bool = True,
) -> list:
    """Scan the WHOLE video and return every event found.

    Deliberately does not stop at the first anomaly: one video can hold several events of
    different classes (docs/architecture.md 1.7 - T026 holds four).

    Two scales when `multi_scale` (docs/architecture.md Stage 0): dwell classes such as
    `loitering` are DEFINED by persistence and carry no evidence inside a 4s window, so a
    long scale is run alongside the short one. Verdicts from the long pass are kept only
    for dwell classes; verdicts from the short pass only for event classes. Without that
    filtering the two scales would contradict each other on the same timeline.
    """
    duration = video_duration_sec(video_path)
    if duration <= 0:
        return []

    verdicts = scan_scale(
        model, processor, video_path, duration, num_frames, window_sec, stride_sec
    )
    if not multi_scale:
        return aggregate_video(video_id, verdicts)

    verdicts = [v for v in verdicts if v.class_name not in DWELL_CLASSES]

    long_window, long_stride = LONG_SCALE
    if duration >= long_window / 2:  # skip clips too short for the long scale to mean anything
        long_verdicts = scan_scale(
            model, processor, video_path, duration, num_frames, long_window, long_stride
        )
        verdicts.extend(v for v in long_verdicts if v.class_name in DWELL_CLASSES)
        # Long-scale normals must survive too, or dwell classes lose the zero-scored windows
        # that terminate their spans.
        verdicts.extend(v for v in long_verdicts if v.class_name == "normal")

    return aggregate_video(video_id, verdicts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Fine-tuned model/adapter dir")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("preds.csv"))
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--stride-sec", type=float, default=4.0)  # =window: no overlap
    parser.add_argument("--gate", choices=["none"], default="none", help="Stage 1 gate (TODO)")
    parser.add_argument("--no-multi-scale", action="store_true",
                        help="Short scale only. Cheaper, but dwell classes (loitering, "
                             "stalled vehicle) become undecidable - see architecture 7.3.")
    parser.add_argument("--limit", type=int, default=None, help="Only process N videos (debug)")
    args = parser.parse_args()

    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(str(args.model), load_in_4bit=True)
    FastVisionModel.for_inference(model)

    test_root = args.dataset_root / "test"
    videos = pd.read_csv(test_root / "videos.csv")
    if args.limit:
        videos = videos.head(args.limit)

    all_events = []
    for row in tqdm(list(videos.itertuples(index=False)), desc="videos"):
        all_events.extend(
            run_video(
                model,
                processor,
                test_root / row.filename,
                row.video_id,
                args.num_frames,
                args.window_sec,
                args.stride_sec,
                multi_scale=not args.no_multi_scale,
            )
        )

    df = pd.DataFrame(
        [
            {
                "video_id": e.video_id,
                "class_name": e.class_name,
                "start_time_sec": e.start_time_sec,
                "end_time_sec": e.end_time_sec,
                "score": e.score,
            }
            for e in all_events
        ],
        columns=["video_id", "class_name", "start_time_sec", "end_time_sec", "score"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} events across {df['video_id'].nunique()} videos to {args.out}")


if __name__ == "__main__":
    main()

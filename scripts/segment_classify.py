"""Segment-level anomaly classification: one VLM call per DETECTED SEGMENT.

Third and correct granularity, after two measured failures (docs/architecture.md 11.9):

  window-level (4s)  -> 0/79 accident windows named correctly. A 4s view of accident
                        AFTERMATH is genuinely indistinguishable from a stalled vehicle;
                        the collision is 1-2 windows out of twenty, so the vote loses.
  video-level        -> 4/8 videos correct, but fails on exactly T025/T026/T028, the
                        "stitched" Level-2 videos (architecture 1.4) that are concatenations
                        of unrelated clips. "The dominant class of this video" is ill-posed
                        when the video contains several different scenes, and those three
                        hold 14 of the 26 Level-2/3 events.
  segment-level      -> this file. Long enough that collision and aftermath are both in
                        context, short enough to cover ONE scene.

Segments come from unsupervised SigLIP novelty localisation, which is already measured as
sufficient: with an oracle class it reaches 8 matched events / 0.669 overall against the
shipped pipeline's 1 match / 0.400. Localisation is not the gap; naming is.

Reads segment spans from a JSON produced locally, so the expensive novelty computation and
all threshold choices stay on the CPU side and only the VLM calls run here.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

ANOMALY_CLASSES = [
    "traffic_accident", "traffic_congestion", "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic", "fire", "smoke", "waterlogging_or_flood",
    "wrong_way_driving", "road_spill_or_debris", "fighting_or_violence",
    "loitering_or_suspicious_presence",
]

SYSTEM = "You are a video surveillance analyst. Answer with JSON only."

# "normal" IS offered here, unlike the video-level pass. That pass forced a choice among 11
# anomaly classes and so labelled the two ground-truth-normal videos (T029/T030) as anomalous
# - which under the official metric zeroes those videos outright (architecture 9.2).
USER = (
    "These frames are sampled from ONE continuous segment of surveillance video.\n"
    "Say what anomaly, if any, is happening in this segment.\n\n"
    "Classes: normal, " + ", ".join(ANOMALY_CLASSES) + "\n\n"
    "Guidance:\n"
    "- If vehicles have COLLIDED, or you see crash damage or debris from an impact, that is "
    "traffic_accident - even if most frames show only the stopped vehicles afterwards.\n"
    "- Use stalled_or_broken_down_vehicle only when a vehicle stopped on its own with no "
    "sign of a collision.\n"
    "- Use traffic_congestion only for dense slow-moving queues with no crash.\n"
    "- loitering_or_suspicious_presence means a person lingers in the area with no clear purpose.\n"
    "- If nothing unusual is happening, answer normal.\n\n"
    'Reply exactly: {"class_name": "<one class>"}'
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--export-dir", type=Path, default=Path("./export_test"))
    ap.add_argument("--segments", type=Path, required=True,
                    help='JSON: {"video_id": [[start,end], ...], ...}')
    ap.add_argument("--out", type=Path, default=Path("./segment_classes.json"))
    ap.add_argument("--n-frames", type=int, default=8)
    args = ap.parse_args()

    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from unsloth import FastVisionModel
    import torch

    rows = [json.loads(l) for l in (args.export_dir / "index.jsonl").open(encoding="utf-8")]
    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r)
    for v in by_video:
        by_video[v].sort(key=lambda r: r["start"])

    segments = json.loads(args.segments.read_text(encoding="utf-8"))
    n_seg = sum(len(v) for v in segments.values())
    print(f"{n_seg} segments across {len(segments)} videos", flush=True)

    model, processor = FastVisionModel.from_pretrained(str(args.model), load_in_4bit=True)
    FastVisionModel.for_inference(model)

    frames_root = args.export_dir / "frames"
    out = defaultdict(list)
    t0 = time.perf_counter()
    done = 0

    for vid, spans in segments.items():
        wins = by_video.get(vid, [])
        for (a, b) in spans:
            inside = [w for w in wins if w["end"] > a and w["start"] < b]
            if not inside:
                out[vid].append({"start": a, "end": b, "class_name": None, "raw": "(no frames)"})
                continue
            step = max(1, len(inside) // args.n_frames)
            picks = []
            for w in inside[::step][: args.n_frames]:
                fs = sorted((frames_root / w["id"]).glob("f*.jpg"),
                            key=lambda p: int(p.stem[1:]))
                if fs:
                    picks.append(fs[len(fs) // 2])
            if not picks:
                out[vid].append({"start": a, "end": b, "class_name": None, "raw": "(no frames)"})
                continue

            imgs = [Image.open(p).convert("RGB") for p in picks]
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": [{"type": "image", "image": im} for im in imgs]
                            + [{"type": "text", "text": USER}]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
            img_in, vid_in = process_vision_info(messages)
            inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True,
                               return_tensors="pt").to(model.device)
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            decoded = processor.decode(gen[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)

            cls = "normal" if '"normal"' in decoded else None
            if cls is None:
                for c in ANOMALY_CLASSES:
                    if c in decoded:
                        cls = c
                        break
            out[vid].append({"start": a, "end": b, "class_name": cls, "raw": decoded.strip()})
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{n_seg}  {(time.perf_counter()-t0):.0f}s", flush=True)

    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\ndone in {(time.perf_counter()-t0)/60:.1f} min -> {args.out}", flush=True)


main()

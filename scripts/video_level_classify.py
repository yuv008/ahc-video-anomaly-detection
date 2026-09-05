"""Video-level anomaly classification: ONE VLM call per video, frames spanning the whole clip.

Why this exists (docs/architecture.md 11.9): every class decision so far was a majority vote
over independent 4s windows, and that is measurably the binding constraint at Levels 2/3.
Measured: unsupervised SigLIP novelty localisation with an ORACLE video-level class reaches
8 matched events / 0.669 overall, against 1 match / 0.400 for the shipped pipeline. So
localisation is solved and *naming the class* is the entire remaining gap.

Window voting fails on `traffic_accident` specifically - 0/79 accident windows are labelled
correctly - because a 4s window of accident AFTERMATH (stopped vehicles at odd angles) is
genuinely indistinguishable from `stalled_or_broken_down_vehicle`. The collision itself
occupies one or two windows out of twenty, so the vote is dominated by aftermath frames.

Sampling frames across the WHOLE video puts the collision and the aftermath in the same
context window, which is the only way the distinction is decidable at all.

Cost: one call per video (10 for the L2/L3 set), reusing frames already exported for the
window pack - no new extraction, no training.
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

# Asks for the DOMINANT class over the whole video, and explicitly names the confusion we
# measured, because that confusion is the reason this pass exists at all.
USER = (
    "These frames are sampled evenly across one continuous surveillance video.\n"
    "Identify the single dominant anomaly type present somewhere in this video.\n\n"
    "Classes: " + ", ".join(ANOMALY_CLASSES) + "\n\n"
    "Guidance:\n"
    "- If vehicles have COLLIDED or you see crash damage/debris from an impact, that is "
    "traffic_accident - even if most frames only show the stopped vehicles afterwards.\n"
    "- Use stalled_or_broken_down_vehicle only when a vehicle stopped on its own with no "
    "sign of a collision.\n"
    "- Use traffic_congestion only for dense slow-moving queues with no crash.\n"
    "- loitering_or_suspicious_presence means a person remains in the area over a long period.\n\n"
    'Reply exactly: {"class_name": "<one class>"}'
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--export-dir", type=Path, default=Path("./export_test"))
    ap.add_argument("--out", type=Path, default=Path("./video_level_classes.json"))
    ap.add_argument("--n-frames", type=int, default=16,
                    help="frames spread over the WHOLE video, not one window")
    ap.add_argument("--levels", default="2,3", help="only classify these levels")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="csv with video_id,level; if absent, every video is processed")
    args = ap.parse_args()

    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from unsloth import FastVisionModel

    rows = [json.loads(l) for l in (args.export_dir / "index.jsonl").open(encoding="utf-8")]
    by_video: dict[str, list] = defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r)
    for v in by_video:
        by_video[v].sort(key=lambda r: r["start"])

    wanted = set(by_video)
    if args.manifest and args.manifest.exists():
        import csv
        keep = {int(x) for x in args.levels.split(",")}
        with args.manifest.open(encoding="utf-8") as f:
            wanted = {r["video_id"] for r in csv.DictReader(f)
                      if r.get("level") and int(r["level"]) in keep}

    targets = [v for v in sorted(by_video) if v in wanted]
    print(f"{len(targets)} videos to classify at video level", flush=True)

    model, processor = FastVisionModel.from_pretrained(str(args.model), load_in_4bit=True)
    FastVisionModel.for_inference(model)

    frames_root = args.export_dir / "frames"
    out: dict[str, dict] = {}
    t0 = time.perf_counter()

    for vi, vid in enumerate(targets):
        wins = by_video[vid]
        # One frame per window, spread evenly across the video's full duration.
        picks = []
        step = max(1, len(wins) // args.n_frames)
        for w in wins[::step][: args.n_frames]:
            fs = sorted((frames_root / w["id"]).glob("f*.jpg"),
                        key=lambda p: int(p.stem[1:]))
            if fs:
                picks.append(fs[len(fs) // 2])  # middle frame is most representative
        if not picks:
            continue
        imgs = [Image.open(p).convert("RGB") for p in picks]

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": [{"type": "image", "image": im} for im in imgs]
                        + [{"type": "text", "text": USER}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # process_vision_info applies the SAME smart-resize the chat template assumed when it
        # expanded the image placeholders. Passing raw PIL images instead makes the processor
        # resize them independently, and the image-token counts then disagree
        # ("Mismatch in `image` token count between text and `input_ids`").
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True,
                           return_tensors="pt").to(model.device)
        import torch
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        decoded = processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        cls = None
        for c in ANOMALY_CLASSES:               # substring match survives sloppy JSON
            if c in decoded:
                cls = c
                break
        out[vid] = {"class_name": cls, "raw": decoded.strip(), "n_frames": len(picks)}
        print(f"  [{vi+1}/{len(targets)}] {vid}: {cls}   ({decoded.strip()[:60]})", flush=True)

    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\ndone in {(time.perf_counter()-t0)/60:.1f} min -> {args.out}", flush=True)


main()

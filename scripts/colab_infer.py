"""Run the fine-tuned model over pre-extracted test window frames -> per-window verdicts.

Emits RAW per-window verdicts (video_id, start, end, class_name, score) rather than events.
Aggregation into events happens locally with src/ahc_vad/infer/aggregate.py, so hysteresis
and merge thresholds can be re-tuned without re-running the GPU pass. That separation is
what makes threshold fitting cheap.

Scores come from the logits at the is_anomaly boolean position, not from the decoded string
(see src/ahc_vad/infer/scoring.py) - otherwise every window scores 1.0 and the aggregator's
thresholds are inert.
"""

import argparse
import json
from pathlib import Path


def build_scoring(processor):
    """Local copy of the true/false logit reader, so the VM needs no repo install."""
    import torch

    def token_ids(strings):
        tok = getattr(processor, "tokenizer", processor)
        ids = set()
        for s in strings:
            for variant in (s, f" {s}"):
                try:
                    enc = tok.encode(variant, add_special_tokens=False)
                except TypeError:
                    enc = tok.encode(variant)
                if enc:
                    ids.add(enc[0])
        return ids

    TRUE_IDS, FALSE_IDS = token_ids(("true", "True")), token_ids(("false", "False"))
    overlap = TRUE_IDS & FALSE_IDS
    true_ids, false_ids = TRUE_IDS - overlap, FALSE_IDS - overlap

    def anomaly_prob(generated_ids, scores):
        if not scores or not true_ids or not false_ids:
            return None
        for step, tid in enumerate(generated_ids):
            if step >= len(scores):
                break
            piece = processor.decode([tid], skip_special_tokens=True).strip()
            if not piece:
                continue
            is_true = piece.startswith(("true", "True"))
            is_false = piece.startswith(("false", "False"))
            if not (is_true or is_false):
                continue
            probs = torch.softmax(scores[step][0].float(), dim=-1)
            v = probs.shape[0]
            pt = float(sum(probs[i].item() for i in true_ids if i < v))
            pf = float(sum(probs[i].item() for i in false_ids if i < v))
            return pt / (pt + pf) if (pt + pf) > 0 else (1.0 if is_true else 0.0)
        return None

    return anomaly_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", type=Path, default=Path("/content/export_test"))
    ap.add_argument("--model", type=Path, default=Path("/content/qwen7b-lora"))
    ap.add_argument("--out", type=Path, default=Path("/content/window_verdicts.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    # Subsample frames from the SAME packs rather than re-exporting. 8f/256tok measures
    # 4.64 s/window on a T4 = 116% of the 4s real-time budget, so it does not qualify as
    # real-time on limited GPU. Taking every other frame halves the vision tokens
    # (768 vs 1536 under Qwen3-VL) and should clear the budget, while keeping resolution
    # identical to training - a frame-count mismatch is far milder than a resolution one.
    ap.add_argument("--frames", type=int, default=None,
                    help="Use N evenly-spaced frames of the 8 available (default: all)")
    args = ap.parse_args()

    import time

    import torch
    from PIL import Image
    from unsloth import FastVisionModel
    from qwen_vl_utils import process_vision_info

    ANOMALY_CLASSES = [
        "traffic_accident", "traffic_congestion", "stalled_or_broken_down_vehicle",
        "vehicle_blocking_traffic", "wrong_way_driving", "road_spill_or_debris",
        "waterlogging_or_flood", "fire", "smoke", "fighting_or_violence",
        "loitering_or_suspicious_presence",
    ]
    VALID = set(ANOMALY_CLASSES) | {"normal"}
    SYSTEM_PROMPT = (
        "You are a real-time visual anomaly detector for city drone, CCTV and dashcam footage. "
        "Given a short sequence of frames from one time window, decide whether they show one of "
        "these anomalies: " + ", ".join(ANOMALY_CLASSES) + ", or normal if nothing of concern is "
        "happening. Most footage is ordinary and should be called normal. "
        'Reply with a single JSON object: {"is_anomaly": true|false, "class_name": "<label>"}.'
    )

    model, processor = FastVisionModel.from_pretrained(str(args.model), load_in_4bit=True)
    FastVisionModel.for_inference(model)
    anomaly_prob = build_scoring(processor)

    rows = [json.loads(l) for l in (args.export_dir / "index.jsonl").open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    frames_root = args.export_dir / "frames"
    print(f"{len(rows)} windows to score", flush=True)

    out_f = args.out.open("w", encoding="utf-8")
    t0 = time.perf_counter()
    n_anom = 0

    for i, r in enumerate(rows):
        t_win = time.perf_counter()
        paths = sorted((frames_root / r["id"]).glob("f*.jpg"), key=lambda p: int(p.stem[1:]))
        if not paths:
            continue
        if args.frames and args.frames < len(paths):
            idx = [round(i * (len(paths) - 1) / (args.frames - 1)) for i in range(args.frames)]
            paths = [paths[i] for i in dict.fromkeys(idx)]
        imgs = [Image.open(p).convert("RGB") for p in paths]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image", "image": im} for im in imgs]
                                        + [{"type": "text", "text": "What is happening in this window?"}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True,
                           return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)
        gen = out.sequences[0][inputs["input_ids"].shape[1]:]
        decoded = processor.decode(gen, skip_special_tokens=True)

        cls, is_anom = "normal", False
        try:
            obj = json.loads(decoded[decoded.index("{"): decoded.rindex("}") + 1])
            c = obj.get("class_name")
            if c in VALID and bool(obj.get("is_anomaly", False)) and c != "normal":
                cls, is_anom = c, True
        except (ValueError, json.JSONDecodeError):
            pass

        p = anomaly_prob(gen.tolist(), out.scores)
        score = (p if p is not None else 1.0) if is_anom else 0.0
        if is_anom:
            n_anom += 1

        # Per-window wall time. The submission needs per-VIDEO runtime_metadata (required on
        # every video, and the source of the latency bonus), so timing is recorded here and
        # summed per video downstream rather than reconstructed from a global average.
        out_f.write(json.dumps({
            "video_id": r["video_id"], "start_time_sec": r["start"], "end_time_sec": r["end"],
            "class_name": cls if is_anom else "normal", "score": score,
            "window_ms": round((time.perf_counter() - t_win) * 1000, 1),
            "n_frames": len(imgs),
            "raw": decoded[:120],
        }) + "\n")

        if (i + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1}/{len(rows)}  {el/(i+1):.2f}s/win  anomalies={n_anom}", flush=True)

    out_f.close()
    el = time.perf_counter() - t0
    print(f"\ndone: {len(rows)} windows in {el/60:.1f} min ({el/max(len(rows),1):.2f}s/window)",
          flush=True)
    print(f"anomaly windows: {n_anom} / {len(rows)} ({n_anom/max(len(rows),1):.1%})", flush=True)
    print(f"wrote {args.out}", flush=True)


main()

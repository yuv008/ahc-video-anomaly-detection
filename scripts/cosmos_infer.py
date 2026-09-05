"""Score windows with NVIDIA Cosmos-Embed1-448p-anomaly-detection (retrieval, not generation).

Why this is worth trying as Stage 2 rather than just a gate:

  - It is PURPOSE-BUILT for this task: LoRA-tuned on Vad-Reasoning (1,755 videos, 24 anomaly
    categories). Our zero-shot Qwen managed class accuracy 0.200 and never emitted 8 of 11
    classes; Cosmos reports Top-1 46.4% / Macro F1 38.9% on a 24-way split.
  - It is ONE FORWARD PASS. No autoregressive decode, so it should run roughly 15-45x faster
    than the 4.65 s/window VLM - which matters for both the real-time claim and the latency
    bonus in the official scoring.
  - It emits per-class probabilities NATIVELY: video and text land in one 768-d space, and a
    softmax over cosine similarities to the class phrases gives exactly the continuous
    per-class score the Stage-3 aggregator wants. The VLM needed token-logprob surgery to
    produce the same thing.
  - It is open-vocabulary: the class list is just the text you embed, so the 12 labels are
    supplied as phrases rather than baked into a head.

Reads the same frame packs as colab_infer.py, so no re-export is needed - the model card
states arbitrary non-square resolutions are supported, and our 588x336 frames are fed
directly.

Output format matches colab_infer.py exactly (one JSON verdict per window), so the existing
aggregator, scorer and submission builder all work unchanged.
"""

import argparse
import json
import time
from pathlib import Path

MODEL_ID = "nvidia/Cosmos-Embed1-448p-anomaly-detection"

# Retrieval quality depends heavily on phrasing: these are matched to how the tuning data
# describes events, not to our terse label strings. The mapping back to official class names
# is explicit so the submission vocabulary stays exact.
CLASS_PROMPTS = {
    "traffic_accident": "a traffic accident or vehicle collision on a road",
    "traffic_congestion": "heavy traffic congestion with queued or slow moving vehicles",
    "stalled_or_broken_down_vehicle": "a stalled or broken down vehicle stopped on the road",
    "vehicle_blocking_traffic": "a vehicle illegally blocking or obstructing the roadway",
    "wrong_way_driving": "a vehicle driving the wrong way against traffic",
    "road_spill_or_debris": "spilled cargo or debris obstructing the road surface",
    "waterlogging_or_flood": "flooding or waterlogging covering the road",
    "fire": "an active fire with visible flames",
    "smoke": "a large plume of smoke rising",
    "fighting_or_violence": "people fighting or a violent physical altercation",
    "loitering_or_suspicious_presence": "a person loitering suspiciously for a prolonged time",
    "normal": "an ordinary everyday scene with nothing unusual happening",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", type=Path, default=Path("./export_test_l1"))
    ap.add_argument("--out", type=Path, default=Path("./verdicts_cosmos.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # T4 has no usable bf16 (emulated, ~9x slower than fp16 - measured), so pick fp16 there.
    dtype = torch.float16 if dev == "cuda" else torch.float32
    print(f"loading {MODEL_ID} on {dev}/{dtype} ...", flush=True)

    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(dev, dtype=dtype)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    names = list(CLASS_PROMPTS)
    prompts = [CLASS_PROMPTS[n] for n in names]
    with torch.inference_mode():
        t_in = proc(text=prompts).to(dev, dtype=dtype)
        t_emb = model.get_text_embeddings(**t_in).text_proj  # [12, 768], L2-normalised

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
        arr = np.stack([np.array(Image.open(p).convert("RGB")) for p in paths])  # [T,H,W,C]
        batch = np.transpose(arr[None, ...], (0, 1, 4, 2, 3))  # -> [B,T,C,H,W]

        with torch.inference_mode():
            v_in = proc(videos=batch).to(dev, dtype=dtype)
            v_emb = model.get_video_embeddings(**v_in).visual_proj
            probs = torch.softmax(model.logit_scale.exp() * v_emb @ t_emb.T, dim=-1)[0]

        p = probs.float().cpu().numpy()
        best = int(p.argmax())
        cls = names[best]
        # P(anomaly) = 1 - P(normal): a single calibrated quantity for the aggregator's
        # hysteresis, rather than the raw argmax margin.
        p_anom = float(1.0 - p[names.index("normal")])
        is_anom = cls != "normal"
        if is_anom:
            n_anom += 1

        out_f.write(json.dumps({
            "video_id": r["video_id"], "start_time_sec": r["start"], "end_time_sec": r["end"],
            "class_name": cls if is_anom else "normal",
            "score": p_anom if is_anom else 0.0,
            "window_ms": round((time.perf_counter() - t_win) * 1000, 1),
            "n_frames": len(paths),
            "top3": [[names[j], round(float(p[j]), 3)] for j in p.argsort()[::-1][:3]],
        }) + "\n")

        if (i + 1) % 25 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1}/{len(rows)}  {el/(i+1):.3f}s/win  anomalies={n_anom}", flush=True)

    out_f.close()
    el = time.perf_counter() - t0
    print(f"\ndone: {len(rows)} windows in {el/60:.2f} min ({el/max(len(rows),1):.3f}s/window)",
          flush=True)
    print(f"anomaly windows: {n_anom}/{len(rows)} ({n_anom/max(len(rows),1):.1%})", flush=True)


main()

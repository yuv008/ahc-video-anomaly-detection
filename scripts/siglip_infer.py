"""Score windows with a zero-shot SigLIP image-text classifier.

Same idea as cosmos_infer.py (one forward pass, no autoregressive decode, per-class
probabilities natively from cosine similarity) but with a frozen, generic vision-language
model instead of a VAD-tuned one. This was the original Stage-1 gate proposal
(docs/architecture.md Stage 1) - never built as a *trained* head, but a zero-shot version
costs nothing to try and gives a third independent signal alongside Qwen3-VL and Cosmos.

SigLIP is single-image, not single-video: each window's frames are embedded independently
and mean-pooled before comparing against the class text embeddings. SigLIP's own head uses
a sigmoid loss (each class independent), so mean image-text logits are also reported ungated
by softmax - useful for a genuinely open "is ANY class present" gate rather than a forced
argmax over an exhaustive class list.

Output format matches colab_infer.py / cosmos_infer.py exactly (one JSON verdict per window).
"""

import argparse
import json
import time
from pathlib import Path

MODEL_ID = "google/siglip-base-patch16-256"

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
    ap.add_argument("--export-dir", type=Path, default=Path("./export_test"))
    ap.add_argument("--out", type=Path, default=Path("./verdicts_siglip.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    print(f"loading {MODEL_ID} on {dev}/{dtype} ...", flush=True)

    model = AutoModel.from_pretrained(MODEL_ID).to(dev, dtype=dtype)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    def features(x):
        """Newer transformers wraps get_*_features() output in a ModelOutput."""
        return x.pooler_output if hasattr(x, "pooler_output") else x

    names = list(CLASS_PROMPTS)
    prompts = [CLASS_PROMPTS[n] for n in names]
    with torch.inference_mode():
        t_in = proc(text=prompts, padding="max_length", return_tensors="pt").to(dev)
        t_emb = features(model.get_text_features(**t_in))
        t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)  # [12, dim]

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
        imgs = [Image.open(p).convert("RGB") for p in paths]

        with torch.inference_mode():
            im_in = proc(images=imgs, return_tensors="pt").to(dev, dtype=dtype)
            im_emb = features(model.get_image_features(**im_in))
            im_emb = im_emb / im_emb.norm(dim=-1, keepdim=True)  # [T, dim]
            frame_emb = im_emb.mean(dim=0, keepdim=True)          # mean-pool frames -> [1, dim]
            frame_emb = frame_emb / frame_emb.norm(dim=-1, keepdim=True)

            logits = (model.logit_scale.exp() * frame_emb @ t_emb.T + model.logit_bias)[0]
            probs = torch.softmax(logits, dim=-1)  # forced-choice view for aggregator compat

        p = probs.float().cpu().numpy()
        best = int(p.argmax())
        cls = names[best]
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

        if (i + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1}/{len(rows)}  {el/(i+1):.3f}s/win  anomalies={n_anom}", flush=True)

    out_f.close()
    el = time.perf_counter() - t0
    print(f"\ndone: {len(rows)} windows in {el/60:.2f} min ({el/max(len(rows),1):.3f}s/window)",
          flush=True)
    print(f"anomaly windows: {n_anom}/{len(rows)} ({n_anom/max(len(rows),1):.1%})", flush=True)


main()

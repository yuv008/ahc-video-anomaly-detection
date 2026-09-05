"""Verify the surprising 3B-slower-than-7B result before acting on it.

First sweep measured Qwen2.5-VL-7B as FASTER than 3B at identical configs (e.g. 4f@128:
2.21s vs 3.08s). That is backwards, and the 3B happened to run first, so load order,
thermal state or clock ramp are plausible confounds.

Design: A -> B -> A. Bench 3B, then 7B, then 3B AGAIN. If the second 3B pass matches the
first, order/thermal is ruled out and the inversion is a real property of the checkpoints.
Also dumps quantization and module dtypes, since a 3B that silently failed to load in 4-bit
(or fell back to a slow dequant path) would explain it.
"""

import gc
import json
import statistics
import time

import numpy as np
import torch
from PIL import Image

PATCH, MERGE = 14, 2
PX_PER_TOKEN = PATCH * PATCH * MERGE * MERGE
CONFIGS = [(4, 420, 224), (8, 420, 224), (8, 588, 336)]  # (frames, w, h)

ANOMALY_CLASSES = [
    "traffic_accident", "traffic_congestion", "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic", "wrong_way_driving", "road_spill_or_debris",
    "waterlogging_or_flood", "fire", "smoke", "fighting_or_violence",
    "loitering_or_suspicious_presence",
]
SYSTEM_PROMPT = (
    "You are a real-time visual anomaly detector for city drone, CCTV and dashcam footage. "
    "Given a short sequence of frames from one time window, decide whether they show one of "
    "these anomalies: " + ", ".join(ANOMALY_CLASSES) + ", or normal if nothing of concern is "
    "happening. Most footage is ordinary and should be called normal. "
    'Reply with a single JSON object: {"is_anomaly": true|false, "class_name": "<label>"}.'
)
_rng = np.random.default_rng(0)


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def describe(model, label):
    """Confirm the model really is 4-bit and report where time could be going."""
    qcfg = getattr(getattr(model, "config", None), "quantization_config", None)
    print(f"    quantization_config: {qcfg}", flush=True)
    dtypes, n_4bit, n_lin = {}, 0, 0
    for _, m in model.named_modules():
        cls = type(m).__name__
        if "Linear4bit" in cls:
            n_4bit += 1
        if cls.startswith("Linear"):
            n_lin += 1
    for n, p in list(model.named_parameters())[:400]:
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + 1
    print(f"    Linear4bit modules: {n_4bit} / {n_lin} linear-ish", flush=True)
    print(f"    param dtypes (first 400): {dtypes}", flush=True)
    try:
        vt = model.model.visual if hasattr(model.model, "visual") else model.visual
        n_vis = sum(p.numel() for p in vt.parameters())
        n_all = sum(p.numel() for p in model.parameters())
        print(f"    vision tower params: {n_vis/1e6:.0f}M of {n_all/1e6:.0f}M total", flush=True)
    except Exception as e:
        print(f"    (vision tower introspection failed: {type(e).__name__})", flush=True)


def bench(model, processor, n_frames, w, h, n_reps=7, n_warmup=3):
    from qwen_vl_utils import process_vision_info

    free()
    frames = [Image.fromarray(_rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
              for _ in range(n_frames)]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": f} for f in frames]
                                     + [{"type": "text", "text": "What is happening in this window?"}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img, vid = process_vision_info(messages)
    inputs = processor(text=[text], images=img, videos=vid, padding=True,
                       return_tensors="pt").to(model.device)

    for _ in range(n_warmup):
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=24, do_sample=False)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=24, do_sample=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "n_frames": n_frames, "wh": f"{w}x{h}",
        "input_tok": int(inputs["input_ids"].shape[1]),
        "median": statistics.median(times), "min": min(times), "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def pass_over(model_id, label, pass_name, results, introspect=False):
    from unsloth import FastVisionModel

    free()
    print(f"\n[{pass_name}] loading {label} ...", flush=True)
    t0 = time.perf_counter()
    model, processor = FastVisionModel.from_pretrained(
        model_id, load_in_4bit=True, use_gradient_checkpointing=False
    )
    FastVisionModel.for_inference(model)
    print(f"    load took {time.perf_counter()-t0:.0f}s, "
          f"weights {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    if introspect:
        describe(model, label)

    for n_frames, w, h in CONFIGS:
        r = bench(model, processor, n_frames, w, h)
        r.update(model=label, pass_name=pass_name)
        results.append(r)
        print(f"    {n_frames}f {w}x{h}: median={r['median']:.3f}s "
              f"min={r['min']:.3f} max={r['max']:.3f} sd={r['stdev']:.3f} "
              f"tok={r['input_tok']}", flush=True)

    del model, processor
    free()


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} SM{p.major}{p.minor}", flush=True)
    try:
        print("clocks(MHz) sm/mem:", torch.cuda.clock_rate() if hasattr(torch.cuda, "clock_rate") else "n/a")
    except Exception:
        pass

    M3 = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
    M7 = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"

    results = []
    pass_over(M3, "3B", "A1-3B-first", results, introspect=True)
    pass_over(M7, "7B", "B-7B", results, introspect=True)
    pass_over(M3, "3B", "A2-3B-repeat", results)

    print("\n\n================ A/B/A COMPARISON ================")
    hdr = f"{'config':<16}{'A1 3B':>10}{'B 7B':>10}{'A2 3B':>10}{'A1 vs A2':>11}{'verdict':>22}"
    print(hdr); print("-" * len(hdr))
    for n_frames, w, h in CONFIGS:
        key = f"{n_frames}f {w}x{h}"
        def get(pn):
            for r in results:
                if r["pass_name"] == pn and r["n_frames"] == n_frames and r["wh"] == f"{w}x{h}":
                    return r["median"]
            return float("nan")
        a1, b, a2 = get("A1-3B-first"), get("B-7B"), get("A2-3B-repeat")
        drift = abs(a1 - a2) / a1 * 100 if a1 else float("nan")
        if drift > 10:
            verdict = "ORDER EFFECT"
        elif a2 > b:
            verdict = "3B genuinely slower"
        else:
            verdict = "3B faster (as expected)"
        print(f"{key:<16}{a1:>10.3f}{b:>10.3f}{a2:>10.3f}{drift:>10.1f}%{verdict:>22}")

    with open("/content/bench_verify.json", "w") as f:
        json.dump(results, f, indent=2)


main()

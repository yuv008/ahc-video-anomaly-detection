"""Stage-2 VLM throughput benchmark. Runs remotely on a T4 via `colab exec`.

Settles three parameters (docs/architecture.md 5): model size, frames per window, and the
per-frame resolution cap.

Resolution is swept because it dominates. At Qwen2.5-VL defaults one 1280x720 frame costs
1,196 vision tokens (patch 14, merge 2 -> 784 px/token), so an 8-frame window is 9,568
tokens before the prompt. Capping to 256 tok/frame gives 2,016. Benchmarking only the
default would measure a config we would never deploy.

Frames are pre-resized to exact multiples of 28 so the intended token count is what the
processor actually produces - the output reports expected vs actual so this is verified,
not assumed.

Content does not matter for timing, so frames are synthetic and no dataset is needed.
"""

import gc
import json
import math
import statistics
import time

import numpy as np
import torch
from PIL import Image

PATCH, MERGE = 14, 2
FACTOR = PATCH * MERGE                        # 28: image dims must be multiples of this
PX_PER_TOKEN = PATCH * PATCH * MERGE * MERGE  # 784 px per vision token
SRC_W, SRC_H = 1280, 720                      # dominant real resolution (121/144 clips)

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
USER_PROMPT = "What is happening in this window?"

_rng = np.random.default_rng(0)


def dims_for_token_budget(tok_per_frame):
    """Largest 28-aligned box at source aspect ratio fitting a token budget.
    None -> no cap (Qwen's own smart_resize of native 720p)."""
    if tok_per_frame is None:
        return round(SRC_W / FACTOR) * FACTOR, round(SRC_H / FACTOR) * FACTOR
    beta = math.sqrt((SRC_W * SRC_H) / (tok_per_frame * PX_PER_TOKEN))
    w = max(FACTOR, math.floor(SRC_W / beta / FACTOR) * FACTOR)
    h = max(FACTOR, math.floor(SRC_H / beta / FACTOR) * FACTOR)
    return w, h


def expected_tokens(w, h):
    return (w * h) // PX_PER_TOKEN


def fake_frames(n, w, h):
    """Structured noise: flat colour can compress through the vision path and understate cost."""
    return [Image.fromarray(_rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(n)]


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load(model_id):
    from unsloth import FastVisionModel

    free()
    model, processor = FastVisionModel.from_pretrained(
        model_id, load_in_4bit=True, use_gradient_checkpointing=False
    )
    FastVisionModel.for_inference(model)
    return model, processor, torch.cuda.max_memory_allocated() / 1e9


def bench_config(model, processor, n_frames, tok_per_frame, n_reps=5, n_warmup=2,
                 max_new_tokens=24):
    from qwen_vl_utils import process_vision_info

    free()
    w, h = dims_for_token_budget(tok_per_frame)
    frames = fake_frames(n_frames, w, h)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": f} for f in frames]
                                     + [{"type": "text", "text": USER_PROMPT}]},
    ]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        n_tok = int(inputs["input_ids"].shape[1])

        for _ in range(n_warmup):
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()

        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        return {
            "n_frames": n_frames,
            "tok_per_frame": tok_per_frame if tok_per_frame else "default",
            "frame_wh": f"{w}x{h}",
            "expected_vision_tok": expected_tokens(w, h) * n_frames,
            "actual_input_tok": n_tok,
            "median_sec": statistics.median(times),
            "min_sec": min(times),
            "max_sec": max(times),
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
    except torch.cuda.OutOfMemoryError:
        print(f"    OOM: {n_frames}f @ {tok_per_frame}", flush=True)
        free()
        return None
    except Exception as e:
        print(f"    fail: {type(e).__name__}: {str(e)[:160]}", flush=True)
        free()
        return None


def run_model(model_id, label, configs, results):
    print(f"\nloading {label} ...", flush=True)
    try:
        model, processor, weights_gb = load(model_id)
    except Exception as e:
        print(f"  load failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return
    print(f"  weights {weights_gb:.2f} GB", flush=True)

    for n_frames, tok in configs:
        print(f"  {label}: {n_frames}f @ {tok} tok/frame", flush=True)
        r = bench_config(model, processor, n_frames, tok)
        if r:
            r["model"] = label
            r["weights_gb"] = weights_gb
            results.append(r)
            print(f"    {r['median_sec']:.3f}s | {r['actual_input_tok']} tok | "
                  f"peak {r['peak_gb']:.1f}GB", flush=True)

    del model, processor
    free()


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} {p.total_memory/1e9:.1f}GB SM{p.major}{p.minor}", flush=True)

    for t in [None, 512, 256, 128]:
        w, h = dims_for_token_budget(t)
        print(f"  cap={str(t):>7} -> {w}x{h} = {expected_tokens(w,h)} tok/frame", flush=True)

    results = []
    run_model(
        "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit", "Qwen2.5-VL-3B",
        [(4, 128), (8, 128), (16, 128), (4, 256), (8, 256), (16, 256), (8, 512), (8, None)],
        results,
    )
    run_model(
        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit", "Qwen2.5-VL-7B",
        [(4, 128), (8, 128), (8, 256), (16, 256)],
        results,
    )

    print("\n\n================ RESULTS ================")
    hdr = (f"{'model':<16}{'frames':>7}{'tok/frm':>9}{'frame wh':>11}"
           f"{'in tok':>8}{'sec/win':>9}{'peak GB':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda x: (x["model"], x["median_sec"])):
        print(f"{r['model']:<16}{r['n_frames']:>7}{str(r['tok_per_frame']):>9}"
              f"{r['frame_wh']:>11}{r['actual_input_tok']:>8}"
              f"{r['median_sec']:>9.3f}{r['peak_gb']:>9.1f}")

    print("\ntoken-math check (actual should be expected + ~160 prompt tokens):")
    for r in results:
        d = r["actual_input_tok"] - r["expected_vision_tok"]
        print(f"  {r['model']} {r['n_frames']}f@{r['tok_per_frame']}: "
              f"expected={r['expected_vision_tok']} actual={r['actual_input_tok']} "
              f"delta={d} {'ok' if 0 < d < 400 else '<-- CHECK'}")

    # Public test = 3391s of video; 4s windows @2s stride -> 1696 windows.
    TEST_SECONDS, STRIDE = 3391, 2.0
    n_windows = TEST_SECONDS / STRIDE
    print(f"\n\n================ VERDICT ================")
    print(f"public test {TEST_SECONDS}s -> {n_windows:.0f} windows @ {STRIDE}s stride")
    print(f"real-time budget: {STRIDE:.1f}s per window per feed\n")
    hdr = f"{'config':<34}{'offline test':>14}{'realtime':>10}{'gate drop':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda x: x["median_sec"]):
        total_min = n_windows * r["median_sec"] / 60
        drop = max(0.0, 1 - STRIDE / r["median_sec"])
        name = f"{r['model']} {r['n_frames']}f@{r['tok_per_frame']}"
        print(f"{name:<34}{total_min:>11.1f} min"
              f"{('YES' if r['median_sec'] <= STRIDE else 'no'):>10}{drop*100:>10.0f}%")

    with open("/content/bench_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\ncollected {len(results)} configs")


main()

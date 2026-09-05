# AHC Visual Intelligence Hackathon — Real-Time Video Anomaly Detection

Prep repo for the 05 Sept 2026 AHC hackathon: detect anomalies (accidents, fire, congestion,
flooding, fighting, loitering, etc.) in drone/CCTV/dashcam video, in real time, using a
**small** vision-language model that can actually run on limited GPU capability across many
feeds at once. Full brief: [docs/brief.md](docs/brief.md).

**Read [docs/architecture.md](docs/architecture.md) first** — it contains the data analysis
that drives every design decision here, plus what an earlier design got wrong (§7) and the
caveats that remain unresolved (§8).

## Layout

```
docs/
  architecture.md       ** the design + evidence. Start here. **
  brief.md               problem, scope, constraints, schedule
  dataset.md             dataset layout, labels, ground_truth.csv schema, download mirrors
  primer.md              VAD reading list + fine-tuning framework notes
  setup_guide.md         compute/API setup checklist

scripts/
  download_dataset.py    pulls the train/test pack from a Google Drive mirror
  build_synth_set.py     generates the synthesized long-form dev set (arch 3.2)
  oracle_ceiling.py      max score the current window/aggregation config allows

src/ahc_vad/
  schema.py              label set + CSV column constants (single source of truth)
  data/
    dataset.py           loads train/<class>/ + test/ into Event records
    sampling.py          frame sampling + sliding-window helpers
    windows.py           sub-window generation with OVERLAP-BASED labeling (arch 3.1)
    synth.py             long-video synthesizer; manifest-based, renders no video (arch 3.2)
    build_sft_dataset.py window-level SFT data + class balancing (arch 3.3)
  train/
    finetune_unsloth.py  LoRA fine-tune, auto-selects fp16 on T4 / bf16 on Ampere+
    finetune_swift.sh    ms-swift CLI alternative for longer paid-GPU runs
  eval/
    matching.py          event-level matching (temporal IoU for L2/3, class-only for L1)
    evaluate.py          scores an EVENT LIST; per-level breakdown + false alarm rate
  infer/
    aggregate.py         Stage 3: window verdicts -> event list, hysteresis + merge
    realtime_infer.py    sliding-window runtime detector, small model only

data/raw/               the dataset (gitignored)
data/processed/         derived: train.jsonl, synth/ dev set (gitignored)
```

## Status

| Component | State |
|---|---|
| Dataset downloaded + verified | done — 3,173 train / 34 test, schema confirmed |
| Event-list contract (multi-event) | done, oracle-validated |
| Window labeling (arch 3.1) | done — mines 1,604 background windows from anomaly clips |
| Long-form synth dev set (arch 3.2) | done — 40 videos / 160 min / 6.0% coverage |
| Window-level SFT data | done — 14,157 examples |
| Logprob window scores | done — unit-tested, calibrated, monotonic |
| Multi-scale inference | done — 4s event scale + 32s dwell scale |
| Remote T4 via `google-colab-cli` | done — benchmarked, see decisions below |
| Frame-export pipeline | done — parallel, train/serve resolution locked |
| Stage 2 fine-tune | in progress |
| Stage 1 gate | not started (no longer blocking — see below) |

**Oracle ceiling (perfect model, current config): F1 0.978 @ IoU≥0.1** — L1 1.000,
L2 0.941, L3 1.000. Any real score is bounded by this; re-run `oracle_ceiling.py` after
changing window/stride/merge parameters.

## Measured decisions (Tesla T4)

Everything here is measured, not estimated. Details and tables in
[docs/architecture.md §5](docs/architecture.md).

| Decision | Value | Evidence |
|---|---|---|
| Model | **Qwen2.5-VL-7B** | measured *faster* than the 3B (2.23s vs 3.11s/window). The 3B is deeper/narrower — 36 layers vs 28 — and depth dominates latency at batch 1. Verified with an A→B→A control (0.3% drift). |
| Precision | **fp16** | `is_bf16_supported()` returns True on a T4 but counts *emulation*: fp16 20.98 TFLOP/s vs bf16 2.28. Selecting bf16 would be ~9× slower. |
| Resolution | **256 tok/frame** (588×336) | uncapped 720p is 1,196 tok/frame → 17.8s/window; capping is a **5× speedup**. |
| Window / stride | **4s / 4s** (no overlap) | 50% overlap is strictly dominated: same F1, *worse* boundary IoU (0.747 → 0.840), 2× compute. |
| Frames per window | **8** | one frame per 0.5s; accidents are ~1s events. |

Consequence: **real-time no longer requires the Stage-1 gate** — 4 configs clear the
4s/window budget on a single feed. The gate is now a throughput multiplier for running many
feeds per GPU, not a correctness dependency.

## Quickstart

```bash
pip install -e . && pip install -r requirements.txt

# dataset -> data/raw/{train,test}   (see docs/dataset.md if Drive rate-limits you)
python scripts/download_dataset.py --mirror 1 --dest data/raw

# build training data + the long-form dev set
python -m ahc_vad.data.build_sft_dataset --out data/processed/train.jsonl
python scripts/build_synth_set.py --n-videos 40

# what's the best achievable score with this window config?
python scripts/oracle_ceiling.py

# fine-tune (T4-safe precision is auto-detected)
python -m ahc_vad.train.finetune_unsloth --output-dir outputs/qwen2.5-vl-3b-lora

# predict -> score
python -m ahc_vad.infer.realtime_infer --model outputs/qwen2.5-vl-3b-lora --out preds.csv
python -m ahc_vad.eval.evaluate --predictions preds.csv --dataset-root data/raw

# tune thresholds against the SYNTH dev set, not the 34-video public test set
python -m ahc_vad.eval.evaluate --predictions preds.csv --dataset-root data/processed/synth
```

## Prediction format

One row per **detected event**, not per video. A video with no events contributes no rows.

```csv
video_id,class_name,start_time_sec,end_time_sec,score
T026,traffic_accident,12.0,38.5,0.91
T026,fighting_or_violence,148.0,205.0,0.77
```

## Next steps

1. Per-class scores from token logprobs — Stage 3 hysteresis does nothing without them.
2. Fine-tune the 3B baseline; score Level 1 first.
3. Benchmark real seconds/window on T4 (3B vs 7B, 4/8/16 frames) before committing model size.
4. Stage 1 SigLIP gate — required for the real-time claim, not for offline scoring (arch 5).
5. Fit aggregator thresholds on `data/processed/synth`, never on public test.

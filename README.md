# AHC Visual Intelligence Hackathon — Real-Time Video Anomaly Detection

Prep repo for the 05 Sept 2026 AHC hackathon: detect anomalies (accidents, fire, congestion,
flooding, fighting, loitering, etc.) in drone/CCTV/dashcam video, in real time, using a
**small** vision-language model that can actually run on limited GPU capability across many
feeds at once. Full brief: [docs/brief.md](docs/brief.md).

**Models: see [docs/architecture.md §0](docs/architecture.md) — Qwen3-VL-8B 4-bit LoRA r=32 is the
deployed classifier; no gate is built; Cosmos-Embed1 is blocked on a transformers version.**

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
| Stage 2 fine-tune | **validated** — L1 0.329 → 0.421 from 80 samples |
| Stage 1 gate | **not built** — real-time achieved without it (see below) |

**Oracle ceiling (perfect model, current config): F1 0.978 @ IoU≥0.1** — L1 1.000,
L2 0.941, L3 1.000. Any real score is bounded by this; re-run `oracle_ceiling.py` after
changing window/stride/merge parameters.

## Measured decisions (Tesla T4)

Everything here is measured, not estimated. Details and tables in
[docs/architecture.md §5](docs/architecture.md).

| Decision | Value | Evidence |
|---|---|---|
| Model | **Qwen3-VL-8B, 4-bit, LoRA r=32** | every AI City Track-3 top team used a ~8B Qwen VLM. Qwen3.5 rejected: no unsloth 4-bit build, and the leaderboard gap is 0.0009. Qwen2.5-VL-3B measured *slower* than the 7B (36 layers vs 28 — depth dominates at batch 1, A→B→A verified, 0.3% drift). |
| Precision | **fp16** | `is_bf16_supported()` returns True on a T4 but counts *emulation*: fp16 20.98 TFLOP/s vs bf16 2.28. Selecting bf16 would be ~9× slower. |
| Resolution | **256 tok/frame** (588×336) | uncapped 720p is 1,196 tok/frame → 17.8s/window; capping is a **5× speedup**. |
| Window / stride | **4s / 4s** (no overlap) | 50% overlap is strictly dominated: same F1, *worse* boundary IoU (0.747 → 0.840), 2× compute. |
| Frames per window | **8 train / 4 infer** | 8f = 4.64 s/window on T4 = 116% of the real-time budget; 4f halves tokens and clears it. |

Consequence: **real-time does not require the Stage-1 gate** — several configs clear the
4s/window budget on a single feed, so no gate was built. Its remaining value is multiplying
feeds per GPU: a throughput argument, not a correctness dependency.

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

# fine-tune Qwen3-VL-8B (T4-safe precision auto-detected: fp16, since bf16 is emulated)
python -m ahc_vad.train.finetune_unsloth --output-dir outputs/qwen3-vl-8b-lora

# predict -> score
python -m ahc_vad.infer.realtime_infer --model outputs/qwen3-vl-8b-lora --out preds.csv
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

## Results so far

| | Level 1 score | classes emitted |
|---|---|---|
| Zero-shot Qwen2.5-VL-7B | 0.329 | 3 / 11 |
| **Qwen3-VL-8B, checkpoint-20** (80 samples) | **0.421** | 5 / 11 |
| Oracle ceiling (perfect window model) | 1.000 | 11 / 11 |

Detection rate also corrected from badly under-firing (7.5% of windows against a true 29.4%)
to well calibrated (26.8%). The zero-shot failure was *silence* on 8 of 11 classes, not
confusion between them — which is exactly what the training windows fix.

## Next steps

1. Finish training and re-score; `traffic_accident` regressed at checkpoint-20 and should be
   re-checked at a later checkpoint.
2. Full 865-window inference at **4 frames** — halves runtime and is the config that clears
   the 4s real-time budget (8 frames measures 4.64s = 116% of it).
3. Sweep aggregator thresholds against the official metric on `data/processed/synth`, never
   on the 34-video public test set.
4. **Cosmos-Embed1-448p-anomaly-detection** in a pinned-transformers venv — purpose-built for
   VAD, one forward pass instead of autoregressive decode, per-class scores natively. Blocked
   only on `apply_chunking_to_forward` being removed in transformers 5.5. Most promising
   unexplored direction.
5. Stage 1 gate, if many-feeds-per-GPU throughput becomes the goal — it is not needed for the
   single-feed real-time claim (arch 5.3).

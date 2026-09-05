# Architecture — Real-Time Drone Video Anomaly Detection

**v2** — revised after adversarial review of v1 against the data. Section 7 lists what was
wrong in v1 and why; §8 lists residual caveats that remain unresolved by design.

Design decisions and their evidence, derived from analysis of the actual downloaded dataset
(2026-09-04/05), not from the brief alone.

**Locked constraints for this build:**
- Primary objective: **private leaderboard score**
- Runtime target: **free T4** (Colab/Kaggle) — 16GB, Turing SM75, **fp16 only, no bf16**
- Level 2/3 localization: **aggregator-derived + stitched synthetic training data**

---

## 1. Data analysis — what actually drives the design

### 1.1 The central problem: train/test distribution mismatch

| | Train clips | Test L1 | Test L2 | Test L3 |
|---|---|---|---|---|
| Count | 3,173 | 24 videos | 6 videos | 4 videos |
| Duration | 5–30s (median 6.5s) | 5–26s | 240s | 308–629s |
| **Event coverage** | **median 99.9%** | ~100% | **median 7.7%** | **median 7.7%** |
| Events per video | 1 | 1 | 1–6 | 1–4 |
| Max event length | 30s | 26s | 60s | **125s** |

Training clips are **pre-trimmed**: the labeled event spans essentially the entire clip, and
70% start at t=0. Level 2/3 test videos are 4–10 minutes in which anomalies occupy **~8% of
the timeline**.

**Consequence:** a model trained naively on trimmed clips learns the shortcut *"a clip was
shown to me ⇒ an anomaly is present"* — true ~100% of the time in training for 11 of 12
classes. Deployed on long footage it fires almost continuously. Per the brief, *"an alerting
system that fires regularly on ordinary activity stops being used."* This mismatch, not model
capacity, is the dominant risk.

### 1.2 Per-class temporal shape (train split) — and why it does NOT transfer

| Class | Median event (s) | Coverage of clip | Character |
|---|---|---|---|
| `traffic_accident` | 5.0 | 1.00 | instantaneous |
| `road_spill_or_debris` | 2.7 | **0.38** | static condition, sub-clip |
| `wrong_way_driving` | 5.0 | **0.45** | sub-clip, directional |
| `vehicle_blocking_traffic` | 5.0 | **0.50** | sub-clip |
| `waterlogging_or_flood` | 5.75 | 1.00 | static condition |
| `traffic_congestion` | 5.3 | 0.99 | gradual build-up |
| `smoke` / `fire` | 5.8 | 0.90–0.99 | gradual, persistent |
| `stalled_or_broken_down_vehicle` | 8.9 | 0.76 | requires dwell |
| `fighting_or_violence` | 29.0 | 0.95 | sustained |
| `loitering_or_suspicious_presence` | **30.0 (always)** | 1.00 | *see warning* |

> **⚠ Do not derive aggregator dwell thresholds from this table.**
> Train `loitering` durations span 28.8–30.0s (essentially a constant — an artifact of clip
> trimming, not of the phenomenon). Test `loitering` durations are **2.6, 9.4, 13.7, 19.5,
> 37.6s**. A dwell rule requiring ~30s of persistence — which v1 proposed — would miss 2 of
> the 5 test loitering events outright. Train durations reflect **how clips were cut**, not
> how long events last. Dwell parameters must be fit on long-form data (§3.2/§3.4), never on
> train clip lengths.

### 1.3 Supervision is categorical, not descriptive

`description_summary` is templated boilerplate: `loitering` 300 videos → **1** unique string;
`waterlogging` 95 → **1**; `fighting` 124 → 6; `normal` 973 → 6; `fire` 77 → 7.

**Consequence:** build a **classifier**, not a captioner. Training the model to emit prose
buys nothing (the "prose" is a restated class label) and costs decode latency at runtime.

### 1.4 Long test videos: Level 2 is constructed, Level 3 is not

| Video | Level | Event timestamps | Reading |
|---|---|---|---|
| T025 | 2 | 20–40, 60–80, 100–120, 140–160, 180–200, 220–240 | **stitched** (perfectly regular) |
| T028 | 2 | 30–35, 90–95, 150–155, 210–215 | **stitched** (60s spacing) |
| T026 | 2 | 10–40.4, 65–74.5, 105–125.8, 150–210 | semi-constructed |
| T027 | 2 | 40–45, 55–60, 65–125, 145–150 | semi-constructed |
| T031 | 3 | 235–360 | **natural** (single 125s event) |
| T032 | 3 | 29.3–66.9, 135.8–149.5, 194.8–197.4, 267.4–286.9 | **natural** (fractional, irregular) |
| T033 | 3 | 170–245, 490–535 | **natural** |

L2 durations are exactly 240.000s with round-number events — clearly assembled. L3 has
fractional, irregular timestamps and events up to **125s, longer than any training clip
(max 30s)**.

**Consequence:** the synthesizer (§3.2) faithfully reproduces **Level 2** only. **Level 3
remains genuinely out-of-distribution** and no amount of stitching fixes it. This is a real
limitation, not a solved problem — see §8.1.

### 1.5 Media properties

Mostly **1280×720 @ 15fps** (121/144 sampled), some 640×480. **Frames-per-window is the
primary latency dial**, not resolution.

### 1.6 Class imbalance (train)

`normal` 973 · `traffic_accident` 565 · `loitering` 300 · `traffic_congestion` 268 ·
`stalled` 223 · `wrong_way` 164 · `road_spill` 151 · `vehicle_blocking` 148 ·
`fighting` 124 · `waterlogging` 95 · `smoke` 85 · `fire` 77 — 7.3:1 imbalance.

### 1.7 The task is multi-event and multi-class per video

T026 alone contains `traffic_accident`, `vehicle_blocking_traffic`, `road_spill_or_debris`
and `fighting_or_violence`. T029/T030 are 240s videos with **zero** events (`normal`, NaN
timestamps) — "emit nothing" is a valid and required output.

**Consequence:** the output contract is **a list of events per video**, not one verdict per
video. v1 got this wrong; see §7.1.

---

## 2. System architecture

```
                 ┌────────────────── per drone feed ──────────────────┐
                 │                                                    │
 video stream ─► [0] Multi-scale window sampler                       │
                 │      short: 4s / 2s stride  (event classes)         │
                 │      long: 32s / 8s stride  (dwell classes)         │
                 ▼                                                    │
                 [1] Gate  (SigLIP encoder + linear head, 1–2 frames)  │
                 │      drops background; HIGH-RECALL operating point  │
                 ▼                                                    │
                 [2] Small VLM  (Qwen2.5-VL 3B, LoRA, 4-bit, fp16)     │
                 │      → per-class scores from token logprobs         │
                 ▼                                                    │
                 [3] Temporal aggregator                               │
                 │      hysteresis + per-class dwell + merge           │
                 ▼                                                    │
                 EVENT LIST: [(class, start_sec, end_sec, score), ...] │
                 └────────────────────────────────────────────────────┘
```

### Stage 0 — Multi-scale window sampler

Two concurrent scales, because **some classes are not decidable at 4s**:

| Scale | Window / stride | Frames | Serves |
|---|---|---|---|
| **Short** | **4s / 4s** (no overlap) | 8 | `accident`, `fire`, `smoke`, `flood`, `congestion`, `road_spill` |
| **Long** | 32s / 8s | 8 (spread) | `loitering`, `stalled_vehicle`, `fighting`, `wrong_way` |

Rationale: loitering is *defined* by persistence. A 4s window contains no evidence that a
person has lingered — asking the VLM to label it `loitering` from 4s is asking it to guess
(§7.3). The long scale costs little: 8s stride is 4× fewer invocations than the short scale.

**Stride = window, i.e. no overlap — measured, not assumed.** `scripts/oracle_ceiling.py`
sweep over the public test set:

| window / stride | windows on test set | F1@IoU0.1 | F1@IoU0.5 | mean IoU@0.5 |
|---|---|---|---|---|
| 4s / 2s | 1695 | 0.978 | 0.956 | 0.747 |
| **4s / 4s** | **847** | 0.978 | **0.956** | **0.840** |
| 6s / 6s | 565 | **0.989** | 0.945 | 0.822 |
| 8s / 8s | 423 | 0.978 | 0.911 | 0.772 |

50% overlap (the original spec) is **strictly dominated**: identical F1 at every threshold,
*worse* boundary IoU, and twice the compute. Overlapping windows widen the aggregated span
past the true event, which is why IoU drops. 6s/6s scores marginally better at the lenient
threshold and marginally worse at the strict one; since the private metric's IoU threshold
is unpublished (§8.3), 4s/4s is the safer pick — best-or-tied at strict, ~0.01 behind at
lenient. Re-run the sweep before changing any of these numbers.

### Stage 1 — Cheap always-on gate

- Frozen **SigLIP** image encoder + small trained head, **1–2 frames** per window, no decode.
- Trained binary normal-vs-anomalous on train clips **plus** synthesized long-form negatives
  (§3.2) — the negatives are what teach it real background.
- **Operating point chosen by measurement, not assertion**: sweep the threshold on the
  synthesized dev set and pick the knee. v1's "target ≥0.97 recall" was an invented number.
- **Recall is multiplicative** — end-to-end recall ≈ `gate_recall × VLM_recall`. The gate is a
  hard ceiling on system recall, so it is tuned conservatively and its recall is reported as a
  first-class metric, not an implementation detail.

### Stage 2 — Small VLM classifier

- **Qwen2.5-VL 3B Instruct, 4-bit, LoRA, vision tower frozen**, **fp16** (T4 has no bf16).
- 3B for the T4 target; 7B benchmarked as the accuracy fallback since leaderboard is priority.
- **Output: minimal JSON** — `{"is_anomaly": bool, "class_name": str}`.
- **Scores come from token logprobs**, not from the generated string. At the first decoded
  position of the class field we read the logprob of each of the 12 class tokens, giving a
  continuous per-class score. Without this, hysteresis thresholds in Stage 3 are
  unimplementable (§7.4).

### Stage 3 — Temporal aggregator

Converts per-window scores into an **event list**. The model never predicts timestamps — §1.1
shows training data carries almost no sub-clip localization signal, so asking the VLM for
timestamps would be asking it to hallucinate. The aggregator derives them from *where*
windows fired.

Mechanism:
- **Hysteresis** — open an event when score > `θ_high`, keep it open while score > `θ_low`.
  Prevents one event fragmenting into many.
- **Per-class minimum duration + merge gap** — parameters **fit on the synthesized dev set
  and long-form validation, NOT on train clip durations** (§1.2 warning).
- **Multi-event output** — emits every event found, with class and interval; emits an empty
  list for videos like T029/T030.
- **Scale routing** — dwell classes are aggregated from long-scale windows, event classes from
  short-scale.

Cheap, interpretable, independently tunable without retraining — the fastest knob on the day.

---

## 3. Training strategy

### 3.1 Sub-window sampling — labeled by interval overlap, never by folder

Sample 4s/32s sub-windows matching the runtime window shape. **Critical:** a sub-window's
label is determined by its **temporal overlap with the annotated interval**, not by which
class folder the clip came from.

For low-coverage classes (`road_spill` 0.38, `wrong_way` 0.45, `vehicle_blocking` 0.50) a
sub-window drawn outside the annotated interval contains no event and **must be labeled
`normal`**. Labeling it by folder name would teach the model to fire on ordinary traffic —
manufacturing exactly the false alarms the whole design exists to prevent (§7.5).

Rule: label = class if `IoU(window, interval) > τ` (τ ≈ 0.5 of the window), else `normal`.

### 3.2 Synthesized long-form videos (highest leverage)

Mirroring §1.4, build synthetic Level-2-style sequences: concatenate `normal` clips into 240s
backgrounds, insert anomaly clips at random offsets targeting ~8% coverage, emit ground truth
in the real schema.

Value: (a) genuine long-form **negatives** — background windows the model must call normal,
almost absent from raw training data; (b) an honest dev set for fitting gate and aggregator
thresholds; (c) directly attacks the false-alarm failure mode.

**Explicitly does not solve Level 3** (§1.4, §8.1) — synthesized videos inherit train's ≤30s
event lengths and cannot represent T031's 125s congestion event.

### 3.3 Class balance and negative mining

- Weighted sampling to correct the 7.3:1 imbalance (§1.6).
- Oversample `normal` **windows** relative to normal *clips* — at runtime background windows
  vastly outnumber event windows; the training set does not reflect that ratio.
- Hard-negative mining: windows the gate passes but that are truly normal are exactly the
  distribution Stage 2 must get right — recycle them into training.

### 3.4 Validation protocol

Split **by source video**, never by window (windows from one video leak).

| Set | Role |
|---|---|
| Held-out train slice | model selection, class-level accuracy |
| **Synthesized long videos** | **primary dev set** — gate + aggregator thresholds |
| Public test (34 videos) | final sanity check only, sparingly |

The public test set is 34 videos / 28 anomalous / 10 long. Fitting thresholds on it will
overfit — and it is not the private set. Treat every public-test number as a noisy estimate
with roughly ±1 video ≈ ±3% granularity.

---

## 4. Why not the alternatives

| Approach | Why not (here) |
|---|---|
| **YOLO / object detector** | Brief rules it out; data agrees — `stalled_vehicle` vs parked car is the same object, different context. |
| **Pure large hosted VLM** | Explicitly forbidden at runtime by the brief. |
| **Video-native model (VideoMAE etc.)** | Fixed class set, loses open-set/language-queryable property; no checkpoint matches this label set. |
| **Captioning / description generation** | §1.3 — supervision is boilerplate; nothing to distill, real decode cost. |
| **VLM predicts timestamps directly** | §1.1 — ~no sub-clip localization signal in training data. |
| **Single-stage VLM on every window** | Viable for **offline scoring** (§5), not for the real-time claim. |

---

## 5. Throughput — MEASURED on a real T4

All numbers below are measured on a Tesla T4 (15.6GB, SM75) via `scripts/bench_vlm.py`,
4-bit, median of 5 reps after 2 warmups, synthetic 720p frames. Not estimates.

### 5.1 Resolution is the dominant term

At Qwen2.5-VL defaults one 1280×720 frame costs **1,196 vision tokens** (patch 14, merge 2
→ 784 px/token; smart_resize maps 720p to 1288×728). Capping it is the biggest single lever:

| 3B config | input tokens | sec/window |
|---|---|---|
| 8f @ default (uncapped) | 9,742 | **17.81** |
| 8f @ 512 tok/frame | 4,014 | 7.28 |
| 8f @ 256 tok/frame | 2,190 | 4.90 |
| 8f @ 128 tok/frame | 1,134 | **3.51** |

**5× faster from the resolution cap alone**, at identical frame count. Benchmarking only the
default would have measured a config we would never deploy.

### 5.2 Full sweep (sec/window)

| config | 3B | 7B |
|---|---|---|
| 4f @ 128 | 3.08 | **2.21** |
| 8f @ 128 | 3.51 | **3.09** |
| 8f @ 256 | 4.90 | **4.59** |
| 16f @ 256 | 7.55 | 8.48 |

Peak VRAM: 3B 3.2–5.2GB, 7B 6.2–7.2GB — both fit a T4 with room to spare, so **memory is not
the binding constraint; latency is.**

The 3B being *slower* than the 7B at matched configs is counterintuitive; see §5.4.

### 5.3 Real-time is achievable WITHOUT the gate

This is the conclusion that changed. With stride = window = 4s (§Stage 0), the budget is
**4.0s per window per feed**, not the 2.0s implied by the discarded 50%-overlap design, and
the test set is 847 windows rather than 1,696:

| config | sec/window | offline test set | real-time (single feed)? |
|---|---|---|---|
| 7B 4f@128 | 2.21 | 31 min | **YES** — 55% of budget |
| 7B 8f@128 | 3.09 | 44 min | **YES** — 77% |
| 3B 4f@128 | 3.08 | 43 min | **YES** — 77% |
| 3B 8f@128 | 3.51 | 50 min | **YES** — 88% |
| 7B 8f@256 | 4.59 | 65 min | no — 115% |
| 3B 8f@256 | 4.90 | 69 min | no — 122% |

So the Stage-1 gate is **no longer required for the real-time claim on a single feed** — the
stride fix plus the resolution cap bought it outright. The gate's remaining value is
multiplying feeds per GPU (the brief's "many drone feeds at once"), which is a throughput
argument rather than a correctness one. That demotes it from blocking to valuable.

Two distinct requirements, still worth separating:
- **Leaderboard scoring is offline** — 31–50 min for the whole test set is fine.
- **The brief demands real-time** — now satisfied at 4f–8f @128 tok/frame.

### 5.4 The 3B really is slower than the 7B — verified, and explained

`scripts/bench_verify.py` ran A→B→A (3B, then 7B, then 3B again) to rule out load order and
clock ramp:

| config | A1 (3B first) | B (7B) | A2 (3B repeat) | A1↔A2 drift |
|---|---|---|---|---|
| 4f @128 | 3.109 | **2.232** | 3.119 | **0.3%** |
| 8f @128 | 3.658 | **3.021** | 3.515 | 3.9% |
| 8f @256 | 4.854 | **4.652** | 4.839 | 0.3% |

The repeat pass reproduces the first to 0.3%, so it is not an artefact. The model configs
explain why:

| | LLM layers | LLM hidden | vision tower |
|---|---|---|---|
| Qwen2.5-VL-3B | **36** | 2048 | hidden 1280, depth 32 |
| Qwen2.5-VL-7B | **28** | 3584 | hidden 1280, depth 32 — **identical** |

Both ship the *same* vision encoder, so image processing costs the same in each. The 3B is
**deeper and narrower**. At batch size 1, decode latency is governed by sequential depth and
per-layer kernel-launch overhead rather than FLOPs, and 2048-wide matmuls leave a T4's SMs
underutilised. The 3B therefore pays 36 sequential layers per token to do less useful work
than the 7B does in 28. "Smaller model = faster" does not hold at batch 1.

**Decision: Qwen2.5-VL-7B.** Faster, more capable, and fits comfortably (7.2GB peak of
15.6GB). Choosing the 3B would have cost accuracy *and* latency.

### 5.5 Chosen runtime configuration

| Parameter | Value | Why |
|---|---|---|
| Model | **Qwen2.5-VL-7B, 4-bit** | §5.4 — faster and stronger than 3B |
| Window / stride | **4s / 4s** | §Stage 0 — overlap is strictly dominated |
| Frames per window | **8** | one frame per 0.5s; 4 frames leaves 1.33s gaps, and the median event is 5.3s with accidents ~1s |
| Resolution cap | **256 tok/frame (588×336)** | leaderboard is the stated priority, so accuracy heads over the last of the latency |
| Precision | **fp16** | §5.4 of the bf16 finding — emulated bf16 is 9× slower on SM75 |

That is **4.65 s/window → ~65 min** for the whole public test set offline, which is fine.
It sits at 116% of the single-feed real-time budget; **8f @128 (3.02s, 76% of budget)** is
the documented real-time configuration if a live demo is needed. Whether dropping to 128
tok/frame costs accuracy is an empirical question to settle after the fine-tune, not by
assertion — train and evaluate at the resolution you intend to deploy.

---

## 5.6 Execution topology — why frames, not video, cross the wire

Training and inference run on a remote T4 driven by `google-colab-cli`; the dataset lives
locally. Two measurements shaped the pipeline:

- **Upload to the session is ~1.1 MB/s.** Shipping the 15GB dataset would take ~4 hours, so
  that is off the table.
- **gdown cannot fetch the dataset on the VM either** — Google Drive caps anonymous folder
  listings at 50 files per folder and the class folders hold hundreds
  (`FolderContentsMaximumLimitError`). `google.colab.auth.authenticate_user()` blocks on
  interactive input under `colab exec`, so the Drive API path is closed too.

So frames are pre-extracted **locally** and only JPEGs are uploaded
(`scripts/export_frames.py`). The public test set is ~85MB as 8-frame windows versus 1.5GB
of video. This is better engineering than decoding remotely regardless of the transfer cost:

1. **No train/serve skew.** Resolution is fixed once, here, at the same 256 tok/frame cap
   used at inference (§5.5). Training on full-res frames and serving downscaled ones would
   teach the model detail it never sees at runtime.
2. **The GPU never waits on video decode.** Decoding is CPU-bound; doing it once locally
   keeps the T4 saturated with actual training.
3. **The remote side needs no dataset logic.** Windows are already labelled by temporal
   overlap (§3.1), so the VM only reads JPEGs and an index.

Export is parallel across cores (video decode is CPU-bound and windows are independent).
`decord` has no Windows wheel, so `sample_frames` falls back to OpenCV.

**Transport gotchas, all hit in practice — see `tools/upload_chunked.sh`:**

| Symptom | Cause | Fix |
|---|---|---|
| `SSLEOFError` after ~9s on a 207MB upload | `colab upload` base64-encodes the whole file into ONE Jupyter contents-API POST; the server drops it past ~100MB | split into 20MB chunks, upload, `cat` back together on the VM |
| Remote path became `C:/Program Files/Git/content/...` | Git Bash MSYS path conversion rewrites `/content/...` | `MSYS_NO_PATHCONV=1` |
| `ModuleNotFoundError: termios` after setting that | the same flag stops `PYTHONPATH` being translated, so the shim is not found | pass `PYTHONPATH` as a native Windows path |
| `TooManyAssignmentsError` on a new session | stale sessions still hold assignment slots after a runtime is reclaimed | stop old sessions, or retry once they are pruned |

Colab reclaims idle GPU runtimes without warning (seen mid-run: `404/401`, session lost).
Anything long-running should be resumable, and packs re-uploadable, rather than assuming one
continuous session.

**The GPU pass emits raw per-window verdicts, not events.** Aggregation into events happens
locally (`scripts/score_verdicts.py`), so hysteresis and merge thresholds can be swept for
free instead of costing a fresh inference run per candidate. Fit them on the synthesized dev
set, never on the 34-video public test set (§3.4, §8.4).

## 6. Build order

1. **Fix the output contract to multi-event** (§7.1) — blocking; current code cannot score L2/L3.
2. Fix fp16/bf16 for T4. *(done)*
3. Sub-window sampler with overlap-based labeling (§3.1).
4. Long-video synthesizer (§3.2) + by-source validation split (§3.4).
5. Fine-tune Stage 2 baseline; score Level 1.
6. Temporal aggregator (§3) + score Level 2/3.
7. Stage 1 gate; measure throughput and false-alarm rate.
8. Fit thresholds on synthesized dev set; final public-test check.

Steps 1–6 produce a scoring end-to-end system. 7–8 make it competitive and substantiate
real-time. Cut line if time runs short: after step 6.

---

## 7. What v1 got wrong

| # | v1 claim | Reality | Fix |
|---|---|---|---|
| **7.1** | One verdict per video | Task is **multi-event, multi-class** (T026 has 4 classes); `realtime_infer.py` returns the *first* anomaly and stops; `evaluate.py` assumes one row per video. **Structurally cannot score L2/L3.** | Event-list contract + event-level matching metric. **Most severe gap.** |
| **7.2** | Dwell rules from train per-class durations | Train `loitering` ≡ 30s is a *trimming artifact*; test spans 2.6–37.6s. v1's rule would miss 2 of 5 test loitering events. | Fit dwell on long-form dev data only (§1.2 warning). |
| **7.3** | Single 4s window scale | `loitering`/`stalled` are **undecidable** in 4s — persistence is their definition. | Multi-scale sampler (§Stage 0). |
| **7.4** | Hysteresis with θ_high/θ_low | VLM emits discrete tokens; no continuous score existed to threshold. | Per-class scores from token logprobs (§Stage 2). |
| **7.5** | "Sample sub-windows from clips" | Unqualified, this labels event-free sub-windows of low-coverage classes (0.38–0.50) as anomalous — **teaching false alarms**. | Overlap-based labeling (§3.1). |
| **7.6** | "Gate is an optimization, not a correctness dependency" | True for offline scoring; **false** for the brief's real-time requirement — without it a single feed runs slower than real-time. | Offline/real-time split made explicit (§5). |
| **7.7** | "L2/L3 are stitched" | Holds for **L2 only**. L3 is natural footage with 125s events, longer than any train clip. | Claim scoped; L3 acknowledged as OOD (§8.1). |
| **7.8** | Gate "target ≥0.97 recall" | Invented number with no basis; also ignored that cascade recall is *multiplicative*. | Operating point set by measurement; gate recall reported as first-class metric. |
| **7.9** | — | Never stated that "emit zero events" (T029/T030) is a required valid output. | §1.7. |

---

## 8. Residual caveats — not solved by this design

**8.1 Level 3 is out-of-distribution and stays that way.** Training clips max out at 30s;
T031 has a single 125s congestion event and T033 a 75s accident. Nothing in the training data
or the synthesizer represents this. Expect weaker L3 performance; the aggregator's merge-gap
is the only mechanism that can span such events, and it is being asked to extrapolate ~4×
beyond anything it was fit on. **4 of 34 public-test videos are L3.**

**8.2 Annotation coarseness is irreducible.** Coverage ≈ 1.0 on most classes likely means
"the clip was cut around the event", not "the event is visible in every frame". A 5s accident
clip labeled 0–5s may show the collision only at 2–3s. Overlap-based labeling (§3.1) inherits
this coarseness — it fixes folder-name mislabeling, not annotation granularity. This argues
for keeping windows at 4s rather than shorter.

**8.3 RESOLVED — the official metric is now published** (Submission Format doc). See §9.
Two of the earlier guesses were wrong and have been corrected: the IoU gate is **0.5**, not
the 0.1 that was being reported as primary, and scoring is **per-video averaged**, not global
event-level P/R/F1. `src/ahc_vad/eval/official.py` implements the published rules. The only
part still unpublished is the exact weighting of *alert* / *matched events* / *timing* within
a Level-2/3 video; those weights are a documented default in `Weights`, so absolute scores are
indicative while rankings between configs remain reliable.

**8.4 Public test is too small to trust.** 34 videos, 10 of them long. One video ≈ 3%. Any
tuning decision resting on <2 videos of difference is noise.

**8.5 Viewpoint shift is unmeasured.** The brief centers *drone* footage; the data mixes
CCTV/dashcam/drone (descriptions say "aerial viewpoint" vs "fixed CCTV scene"). There is no
viewpoint column, and we have not stratified performance by it. A model strong on CCTV and
weak on aerial would score acceptably on this test set while failing the brief's actual use
case. Worth extracting viewpoint from description text and stratifying validation.

**8.6 Gate and VLM may fail in correlated ways.** Cascades assume stage independence; if both
were trained on the same trimmed-clip distribution they will share blind spots, so the gate's
measured recall on *held-out train* will overstate real-world recall. Measure gate recall on
synthesized long-form data specifically.

**8.7 No night/weather stratification.** Brief explicitly mentions night flights and difficult
visibility. We have not measured performance under those conditions and the data carries no
lighting label.

---

## 9. Official scoring and submission format

From the Submission Format doc. This supersedes the earlier inferred contract (§7.1) and
resolves §8.3.

### 9.1 The metric

| Level | Rule |
|---|---|
| **1** | Pooled over all Level-1 videos: **half** anomaly-vs-normal accuracy, **half** class accuracy. Repeating a class on one video earns nothing. |
| **2 / 3** | Scored **per video, then averaged**. Ground truth normal → predict nothing = **1**, predict anything = **0**. Ground truth has events → weighted mix of *did you alert*, *matched events*, *timing*. Timing weighs more at Level 3. |

**An event matches only when the class is right AND temporal IoU ≥ 0.5.** Not 0.1 — earlier
tuning targeted the wrong threshold. Only the best-overlapping prediction can match a given
ground-truth event; every other fragment counts as a false positive.

**Latency bonus** = total reported processing time ÷ total video duration, which is why
`runtime_metadata` is required per video.

### 9.2 What the metric punishes

These are design pressures, not trivia:

- **A single false alarm on a normal Level-2/3 video scores that whole video 0.** Emitting
  nothing when unsure is worth more than a speculative event. This is the strongest argument
  yet for a precision-biased operating point (§3.3's `normal_ratio > 1`).
- **Fragmentation is actively penalised** — several partial detections of one real event give
  at most one match and the rest become false positives. `merge_gap_sec` therefore matters
  more than it appeared to.
- **"The whole clip is anomalous" fails the 0.5 IoU gate**, so a lazy always-fire baseline
  scores far below a real attempt.

### 9.3 Oracle ceiling under the official metric

`scripts/oracle_ceiling.py` output re-scored with `eval/official.py`:

| | score |
|---|---|
| Level 1 | **1.000** (binary 1.000, class 1.000) |
| Level 2 | 0.929 — alert 1.00, match_f1 0.92, **timing 0.78** |
| Level 3 | 0.926 — alert 1.00, match_f1 0.94, **timing 0.88** |
| **Overall** | **0.952** |

Timing is the binding constraint even with a *perfect* window-level model: 4s windows cannot
align exactly to event boundaries. Finer boundary refinement is the main headroom left at
Levels 2/3 once classification is working.

### 9.4 Submission mechanics

`src/ahc_vad/submit.py` builds and validates the JSON. Rules that are hard rejections or
silent zeros rather than style choices:

1. `"class_name": "normal"` is **rejected** — a normal video is `"events": []`.
2. Level-1 events **must** have `null` timestamps; Level-2/3 require `end > start ≥ 0`.
3. Level 1 wants **one** label per clip.
4. `runtime_metadata` is required on **every** video; `average_time_ms` must equal
   `total_time_ms / call_count` within 2%, and `call_times_ms` must have exactly
   `call_count` entries.
5. `explanation` is optional and bonus-only, valid at 20–500 characters.
6. Max file size **5 MB**.

Operationally: uploads are **partial-update** (a file only touches the videos it mentions,
omitted videos keep their previous answer, and a never-answered video is scored as normal),
and **each upload replaces the score outright — there is no "best of"**. So a run that scores
worse permanently lowers your standing. Validate locally with `submit.validate()` before
spending an upload, and do not submit a configuration that has not beaten the current one on
the local official metric.

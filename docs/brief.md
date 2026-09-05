# AHC Visual Intelligence Hackathon — Real-Time Video Anomaly Detection

Source: `AHC Visual Intelligence Hackathon.pdf`

## Problem

Drones fly over a city (highways, streets, parks, railway stations, bus terminals, public
gatherings, utility sites — day and night). Most footage is routine; a small fraction contains
events that need a response. Goal: detect those events **while the drone is still overhead**,
not in a later review.

- Standard object detectors (YOLO etc.) fail because "anomaly" is contextual, not tied to an
  object class (a stationary car is fine in a parking bay, a problem on a highway shoulder).
- VLMs fit better (language-queryable, open-set) but large hosted VLMs are too slow/expensive
  to run continuously across many drone feeds.
- **Core question: can a small VLM do this reliably in real time?**

## Scope (event types — not a fixed list)

1. Traffic congestion
2. Vehicle breakdown / stopped or illegally parked
3. Vehicle blocking traffic
4. Accidents
5. Smoke and fire
6. Waterlogging / flood
7. Loitering or suspicious presence
8. Road spill or debris
9. Fighting or violence
10. Wrong-way driving

Events differ in temporal shape: an accident is ~1s, congestion builds gradually, a stopped
vehicle only becomes anomalous after a dwell time, an open drain is a static condition not an
event. **False alarms matter as much as misses** — a system that fires on routine activity
stops being used.

## Constraints

- Must run **in real time on limited GPU capability**, cheap enough to run across many drone
  feeds concurrently.
- Large hosted models are allowed for dev/comparison/training-data generation only — **not**
  allowed as part of the runtime detector.
- Open approach: fine-tune a small VLM, distill a larger model, cascade a lightweight
  always-on stage with a heavier verifier, train something purpose-built, or implement recent
  published work.

## What's provided

Unannotated drone/CCTV/dashcam footage (day + night) plus public benchmark datasets
pre-downloaded. See [dataset.md](dataset.md).

## Schedule

| Time | Session |
|---|---|
| 9:00–9:30 | Breakfast and introductions |
| 9:30–11:00 | State-of-the-art talk + FlytBase demos |
| 11:00–18:00 | Build |
| 18:00–19:00 | Demos and results |

**Date:** 05 September 2026 · **Venue:** FlytBase Labs or Online

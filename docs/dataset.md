# Dataset — Training and Public Test Data

Source: `AHC Visual Intelligence Hackathon (1).pdf`

## Download (Google Drive mirrors — pick one, they're identical)

1. https://drive.google.com/drive/folders/1sEFKR7ctd5GfFw-nMlYd_MnTw1VVYz9K?usp=sharing
2. https://drive.google.com/drive/folders/13E_CePn14lcbwMA_yZEiHpAVx6i09UIG?usp=sharing
3. https://drive.google.com/drive/folders/13V8JqgZRMzn2TCF0HTsCqVgUH0UOMmpb?usp=sharing
4. https://drive.google.com/drive/folders/1fS_i7QKXRDI6mnaI6UWqYzKSOYWG8rFv?usp=sharing
5. https://drive.google.com/drive/folders/1efhUZhB6Kyvpw3RulZJSwd0brb8KhuZf?usp=sharing

Full pack is ~15–17 GB. Use `scripts/download_dataset.py` (wraps `gdown`) to pull one mirror
into `data/raw/`.

### If you hit "Cannot retrieve the public link... too many accesses"

Google Drive rate-limits anonymous bulk downloads (likely from many participants hitting the
same shared links). Fix: authenticate `gdown` with your own Google account so downloads count
against your quota, not the shared anonymous one.

1. Install a "cookies.txt export" browser extension while signed into the Google account you
   want to use (e.g. "Get cookies.txt LOCALLY" for Chrome/Edge, "cookies.txt" for Firefox).
2. Visit `drive.google.com` in that browser (make sure you're logged in and can open one of
   the mirror folder links above).
3. Export cookies for the `drive.google.com` / `google.com` domain in Netscape format.
4. Save the exported file to `C:\Users\<you>\.cache\gdown\cookies.txt` (the directory already
   exists in this repo's setup — `~/.cache/gdown/`).
5. Re-run `python scripts/download_dataset.py --mirror 1 --dest data/raw`. `gdown` picks the
   cookie file up automatically (`use_cookies=True`, the script's default).

If that's not an option, the fallback is manually opening the folder in a browser and using
Drive's "Download all" (server-side zip), then unzipping into `data/raw/`.

## Layout

```
train/
  <class_name>/
    videos/*.mp4
    videos.csv
    ground_truth.csv
test/
  videos/*.mp4
  videos.csv
  ground_truth.csv
```

- Training videos are real samples (short event clips, normal contextual clips, longer
  temporal videos). Each label — including `normal` — has its own folder.
- Same media supports three uses: (1) raw video for custom pipelines, (2) `ground_truth.csv`
  for anomaly/class/temporal supervision, (3) `description_summary` column for VLM
  fine-tuning/distillation.
- No synthetic anomaly footage.
- Public test set: 34 videos, ~56 minutes, with `ground_truth.csv` included so you can
  validate your scoring pipeline locally before the private leaderboard.
- Training sources are separated from public test and private eval at the
  source-video/source-sequence level.

## Camera domains

CCTV, dashcam, drone; highways, streets, intersections, campuses, open areas; day/night/bad
weather; short isolated events and longer multi-event videos.

## Label set (12 classes)

1. `normal`
2. `traffic_accident`
3. `traffic_congestion`
4. `stalled_or_broken_down_vehicle`
5. `vehicle_blocking_traffic`
6. `wrong_way_driving`
7. `road_spill_or_debris`
8. `waterlogging_or_flood`
9. `fire`
10. `smoke`
11. `fighting_or_violence`
12. `loitering_or_suspicious_presence`

## `ground_truth.csv` columns

| Column | Notes |
|---|---|
| `video_id` | Repeats — one video can hold several events |
| `level` | 1, 2 or 3 — task tier |
| `is_anomaly` | Binary label |
| `class_name` | One of the 12 strings above — match exactly |
| `start_time_sec` / `end_time_sec` | Empty on Level 1, populated on Levels 2–3 |
| `description_summary` | Short natural-language description; sometimes blank |

Normal videos have exactly one row: `class_name=normal`, empty timestamps.

`videos.csv` (per folder) maps `video_id` → filename.

## Intended use

Hackathon development, training, fine-tuning, distillation, local validation. Teams remain
responsible for the usage terms of the underlying source datasets (no synthetic footage is
distributed, but sources are third-party CCTV/dashcam/drone collections).

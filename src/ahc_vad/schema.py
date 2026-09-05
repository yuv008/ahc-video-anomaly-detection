"""Dataset constants shared across loading, training and evaluation code.

Mirrors the schema in docs/dataset.md exactly - keep the two in sync.
"""

CLASS_NAMES = [
    "normal",
    "traffic_accident",
    "traffic_congestion",
    "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic",
    "wrong_way_driving",
    "road_spill_or_debris",
    "waterlogging_or_flood",
    "fire",
    "smoke",
    "fighting_or_violence",
    "loitering_or_suspicious_presence",
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ANOMALY_CLASSES = [c for c in CLASS_NAMES if c != "normal"]

GROUND_TRUTH_COLUMNS = [
    "video_id",
    "is_anomaly",
    "class_name",
    "start_time_sec",
    "end_time_sec",
    "description_summary",
]

# Present in test/ground_truth.csv (1/2/3 task tier) but absent from every train/<class>/
# ground_truth.csv - confirmed against the real download, not just the doc.
GROUND_TRUTH_OPTIONAL_COLUMNS = ["level"]

VIDEOS_CSV_COLUMNS = ["video_id", "filename"]  # filename is relative to the split dir,
# e.g. "videos/T001.mp4" - it already includes the videos/ prefix.

#!/usr/bin/env bash
# Fetch the private evaluation set (28 videos, ~1.27GB) into data/eval/.
# Mirrors the public test layout so the existing export/inference path needs no changes.
set -u
B="https://visual-intelligence-hackathon-5-sep-2026-evaluation-dataset.aihackerscollective.com"
ROOT="data/eval"

for L in L1 L2 L3; do
  mkdir -p "$ROOT/$L/videos"
  curl -sS -o "$ROOT/$L/videos.csv" "$B/$L/videos.csv"
done

# Resume-friendly (-C -) and retrying: a partial file here silently corrupts frame export.
grep '\.mp4' "$1" | while IFS=$'\t' read -r path size; do
  dest="$ROOT/$path"
  if [ -f "$dest" ] && [ "$(stat -c%s "$dest" 2>/dev/null || echo 0)" = "$size" ]; then
    continue
  fi
  curl -sS --retry 3 --retry-delay 2 -C - -o "$dest" "$B/$path" \
    && echo "ok   $path" || echo "FAIL $path"
done

echo "--- verifying sizes ---"
bad=0
grep '\.mp4' "$1" | while IFS=$'\t' read -r path size; do
  got=$(stat -c%s "$ROOT/$path" 2>/dev/null || echo 0)
  [ "$got" = "$size" ] || { echo "SIZE MISMATCH $path want=$size got=$got"; bad=1; }
done
echo "download pass complete"

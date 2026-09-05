#!/usr/bin/env bash
# Upload a large file to a Colab session in chunks, then reassemble remotely.
#
# `colab upload` posts the whole file base64-encoded in ONE request to the Jupyter contents
# API. Past roughly 100MB the server drops the connection (SSLEOFError), so a 207MB pack
# fails in ~9s. A 20MB chunk uploads fine, so we split, send the parts, and cat them back
# together on the VM.
#
# Also note: the remote path must NOT be mangled by MSYS path conversion, hence
# MSYS_NO_PATHCONV=1 - and because that also stops PYTHONPATH being translated, PYTHONPATH
# is given as a native Windows path.
#
# Usage: tools/upload_chunked.sh <local_file> <remote_path> <session> [chunk_mb]

set -euo pipefail

LOCAL="$1"
REMOTE="$2"
SESSION="$3"
CHUNK_MB="${4:-20}"

export MSYS_NO_PATHCONV=1
export PYTHONPATH="C:/Users/yuvra/OneDrive/Desktop/AHC/tools/winshim"

BASE="$(basename "$REMOTE")"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# The contents API will not create intermediate directories, and an upload into a missing
# directory fails with no useful message. Make it first.
MKDIR_SCRIPT="$WORK/mkdir.py"
cat > "$MKDIR_SCRIPT" <<'PYEOF'
import pathlib
pathlib.Path("/content/chunks").mkdir(parents=True, exist_ok=True)
print("chunks dir ready")
PYEOF
colab exec -s "$SESSION" -f "$(cygpath -w "$MKDIR_SCRIPT" 2>/dev/null || echo "$MKDIR_SCRIPT")" --timeout 120 >/dev/null 2>&1

echo "splitting $(basename "$LOCAL") into ${CHUNK_MB}MB chunks ..."
split -b "${CHUNK_MB}m" -d -a 3 "$LOCAL" "$WORK/${BASE}.part"
PARTS=("$WORK/${BASE}".part*)
echo "  ${#PARTS[@]} chunks"

i=0
for p in "${PARTS[@]}"; do
    i=$((i + 1))
    name="$(basename "$p")"
    printf "  [%2d/%2d] %s ... " "$i" "${#PARTS[@]}" "$name"
    # Convert the local path to Windows form for the CLI, since path conversion is off.
    win="$(cygpath -w "$p" 2>/dev/null || echo "$p")"
    for attempt in 1 2 3; do
        err="$(colab upload -s "$SESSION" "$win" "/content/chunks/$name" 2>&1)" && { echo "ok"; break; }
        if [ "$attempt" = 3 ]; then
            echo "FAILED after 3 attempts"
            echo "$err" | tail -3
            exit 1
        fi
        sleep 5
    done
done

echo "reassembling on the VM ..."
CAT_SCRIPT="$WORK/reassemble.py"
cat > "$CAT_SCRIPT" <<PYEOF
import pathlib, hashlib
chunks = sorted(pathlib.Path("/content/chunks").glob("${BASE}.part*"))
out = pathlib.Path("${REMOTE}")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("wb") as f:
    for c in chunks:
        f.write(c.read_bytes())
        c.unlink()
print(f"reassembled {len(chunks)} chunks -> {out} ({out.stat().st_size/1e6:.0f} MB)")
PYEOF
colab exec -s "$SESSION" -f "$(cygpath -w "$CAT_SCRIPT" 2>/dev/null || echo "$CAT_SCRIPT")" --timeout 600 2>&1 | tail -3

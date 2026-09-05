"""Report the state of a detached VM job: alive/dead plus the tail of its log.

Deliberately short-lived and chatty so `colab exec` always gets output immediately and never
trips the websocket inter-message timeout that killed the inline approach.
"""

import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/content/job.log")
    ap.add_argument("--pidfile", default="/content/job.pid")
    ap.add_argument("--tail", type=int, default=25)
    args = ap.parse_args()

    pidfile = Path(args.pidfile)
    alive = False
    pid = None
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text().strip())
            os.kill(pid, 0)  # signal 0 = existence check, does not kill
            alive = True
        except (ProcessLookupError, ValueError):
            alive = False
        except PermissionError:
            alive = True  # exists but owned by another user

    log = Path(args.log)
    size = log.stat().st_size if log.exists() else 0
    print(f"pid={pid} alive={alive} log_bytes={size}")

    if log.exists():
        lines = log.read_text(errors="replace").splitlines()
        for line in lines[-args.tail:]:
            print(line)


main()

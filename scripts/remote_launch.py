"""Launch a long-running job on the Colab VM detached, then return immediately.

`colab exec` streams kernel output over a websocket and the client raises
`TimeoutError: Timeout waiting for output` if nothing arrives for a while. Any job that is
quiet for minutes - model download, a training step, buffered subprocess output - trips it,
and the client disconnects even though the GPU is still working.

So long jobs are started with Popen, detached, writing to a log file, and this script
returns straight away. Progress is then read with short `colab exec` calls that just tail
the log (scripts/remote_poll.py). Side benefit: the job survives a client disconnect, which
matters because Colab reclaims runtimes without warning.

Edit CMD below (or pass --cmd) before running.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", required=True, help="Shell command to run detached on the VM")
    ap.add_argument("--log", default="/content/job.log")
    ap.add_argument("--pidfile", default="/content/job.pid")
    args = ap.parse_args()

    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    # Truncate so a poll never shows stale output from a previous run.
    log.write_text("")

    with log.open("ab") as f:
        p = subprocess.Popen(
            shlex.split(args.cmd),
            stdout=f, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,          # detach from the kernel's process group
            env={**os.environ, "PYTHONUNBUFFERED": "1"},  # or the log stays empty
        )

    Path(args.pidfile).write_text(str(p.pid))
    print(f"launched pid={p.pid}")
    print(f"log={args.log}")
    sys.stdout.flush()


main()

"""Drive a Lightning AI Studio: upload once, train on a real GPU, fetch results.

Why Lightning over Colab for this project: Studios have **persistent storage**. Colab
reclaimed three GPU sessions mid-run and each loss meant re-uploading 728MB of frame packs.
Here the packs upload once and survive stops, restarts and machine switches, so a lost
session costs minutes of compute rather than half an hour of transfer.

Two more things it gets right for this workload:
  - `upload_folder` sends a directory directly, so none of the tar/split/reassemble
    machinery the Colab contents API forced on us is needed.
  - `switch_machine` means you can upload and prep on a cheap CPU box and pay GPU credits
    only while training actually runs.

Auth is headless via environment variables - set them once, never paste a key into a shell
that gets logged:

    LIGHTNING_API_KEY, LIGHTNING_USER_ID   (Account settings -> API key)

Usage:
    python scripts/lightning_run.py setup                # create studio, upload packs
    python scripts/lightning_run.py train --machine L4   # switch to GPU, train detached
    python scripts/lightning_run.py status
    python scripts/lightning_run.py infer  --machine L4
    python scripts/lightning_run.py fetch                # pull verdicts/adapter back
    python scripts/lightning_run.py stop                 # release the GPU (stops billing)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STUDIO_NAME = "ahc-vad"
REMOTE = "/teamspace/studios/this_studio"


def require_auth() -> None:
    missing = [k for k in ("LIGHTNING_API_KEY", "LIGHTNING_USER_ID") if not os.environ.get(k)]
    if missing:
        sys.exit(
            f"Missing {', '.join(missing)}.\n"
            "Get them from Lightning AI -> Account settings -> API key, then set as USER\n"
            "environment variables (PowerShell):\n"
            '  [Environment]::SetEnvironmentVariable("LIGHTNING_API_KEY","<key>","User")\n'
            '  [Environment]::SetEnvironmentVariable("LIGHTNING_USER_ID","<id>","User")\n'
            "Open a new shell afterwards so the variables are inherited."
        )


def get_studio(machine=None, teamspace=None):
    from lightning_sdk import Machine, Studio

    kwargs = {"name": STUDIO_NAME, "create_ok": True}
    if teamspace:
        kwargs["teamspace"] = teamspace
    studio = Studio(**kwargs)
    if studio.status != "Running":
        target = getattr(Machine, machine) if machine else Machine.CPU
        print(f"starting studio on {machine or 'CPU'} ...", flush=True)
        studio.start(target)
    return studio


def cmd_setup(args) -> None:
    studio = get_studio(teamspace=args.teamspace)
    print(f"studio status: {studio.status}")

    for pack in ("export_test", "export_train"):
        local = REPO / "data" / "processed" / pack
        if not local.exists():
            print(f"  skip {pack} (not exported locally)")
            continue
        print(f"uploading {pack} ... (once - it persists across sessions)", flush=True)
        studio.upload_folder(str(local), remote_path=f"{REMOTE}/{pack}")

    for script in ("colab_train.py", "colab_infer.py"):
        studio.upload_file(str(REPO / "scripts" / script), remote_path=f"{REMOTE}/{script}")

    print(studio.run(
        "pip install -q unsloth qwen-vl-utils 2>&1 | tail -2; "
        f"ls {REMOTE}; python -c \"import torch;print('cuda',torch.cuda.is_available())\""
    ))


def cmd_train(args) -> None:
    studio = get_studio(machine=args.machine, teamspace=args.teamspace)
    from lightning_sdk import Machine

    if args.machine:
        target = getattr(Machine, args.machine)
        print(f"switching to {args.machine} ...", flush=True)
        studio.switch_machine(target)

    cmd = (
        f"cd {REMOTE} && nohup python colab_train.py "
        f"--export-dir {REMOTE}/export_train --output-dir {REMOTE}/qwen7b-lora "
        f"--max-steps {args.max_steps} --grad-accum 4 --lora-r 32 --save-steps 50 "
        f"> {REMOTE}/train.log 2>&1 & echo started"
    )
    print(studio.run(cmd))
    print("training detached; poll with: python scripts/lightning_run.py status")


def cmd_infer(args) -> None:
    studio = get_studio(machine=args.machine, teamspace=args.teamspace)
    limit = f"--limit {args.limit}" if args.limit else ""
    cmd = (
        f"cd {REMOTE} && nohup python colab_infer.py "
        f"--model {REMOTE}/{args.model} --export-dir {REMOTE}/export_test "
        f"--out {REMOTE}/window_verdicts.jsonl {limit} "
        f"> {REMOTE}/infer.log 2>&1 & echo started"
    )
    print(studio.run(cmd))


def cmd_status(args) -> None:
    studio = get_studio(teamspace=args.teamspace)
    print(f"studio: {studio.status}")
    print(studio.run(
        f"tail -3 {REMOTE}/train.log 2>/dev/null; "
        f"tail -3 {REMOTE}/infer.log 2>/dev/null; "
        f"ls -d {REMOTE}/qwen7b-lora/checkpoint-* 2>/dev/null | tail -3; "
        f"wc -l {REMOTE}/window_verdicts.jsonl 2>/dev/null"
    ))


def cmd_fetch(args) -> None:
    studio = get_studio(teamspace=args.teamspace)
    out = REPO / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    try:
        studio.download_file(f"{REMOTE}/window_verdicts.jsonl",
                             str(out / "window_verdicts_lightning.jsonl"))
        print(f"fetched verdicts -> {out / 'window_verdicts_lightning.jsonl'}")
    except Exception as e:
        print(f"no verdicts yet: {type(e).__name__}")


def cmd_stop(args) -> None:
    """Stop the Studio. Files persist; GPU billing does not."""
    studio = get_studio(teamspace=args.teamspace)
    studio.stop()
    print("studio stopped - storage persists, credits no longer burning")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",
                    choices=["setup", "train", "infer", "status", "fetch", "stop"])
    ap.add_argument("--machine", default=None,
                    help="L4 | L40S | A100 | T4 (omit to stay on the current machine)")
    ap.add_argument("--teamspace", default=os.environ.get("LIGHTNING_TEAMSPACE"),
                    help="owner/teamspace, if not inferable from the account")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--model", default="qwen7b-lora")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    require_auth()
    {"setup": cmd_setup, "train": cmd_train, "infer": cmd_infer,
     "status": cmd_status, "fetch": cmd_fetch, "stop": cmd_stop}[args.command](args)


if __name__ == "__main__":
    main()

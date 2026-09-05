# Ship frame packs to the Colab T4 and run training + inference there.
#
# Uploads a single TAR per pack, not individual files: the CLI uploads one file per
# invocation and a pack holds ~7,000 JPEGs, so per-file transfers would be hopeless.
# Measured link speed is ~1.1 MB/s, so an 85MB test pack is ~75s.
#
# Usage:
#   .\tools\push_and_train.ps1 -Stage push
#   .\tools\push_and_train.ps1 -Stage train
#   .\tools\push_and_train.ps1 -Stage infer

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("push", "train", "infer", "fetch")]
    [string]$Stage,
    [string]$Session = "bench"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$colab = Join-Path $PSScriptRoot "colab.ps1"

function Invoke-Colab { & $colab @args }

switch ($Stage) {
    "push" {
        foreach ($pack in @("export_train", "export_test")) {
            $dir = Join-Path $repo "data\processed\$pack"
            if (-not (Test-Path $dir)) { Write-Output "skip $pack (not exported yet)"; continue }

            $tar = Join-Path $env:TEMP "$pack.tar"
            Write-Output "packing $pack ..."
            # -C so paths inside the tar are relative to the pack directory
            tar -cf $tar -C (Split-Path $dir -Parent) (Split-Path $dir -Leaf)
            $mb = (Get-Item $tar).Length / 1MB
            Write-Output ("  {0:N0} MB -> uploading (~{1:N1} min at 1.1 MB/s)" -f $mb, ($mb / 1.1 / 60))

            $t = Measure-Command { Invoke-Colab upload -s $Session $tar "/content/$pack.tar" }
            Write-Output ("  uploaded in {0:N0}s" -f $t.TotalSeconds)
            Remove-Item $tar
        }

        $unpack = Join-Path $env:TEMP "unpack.py"
        @'
import subprocess, pathlib
for pack in ["export_train", "export_test"]:
    tar = pathlib.Path(f"/content/{pack}.tar")
    if not tar.exists():
        print(f"{pack}: no tar"); continue
    subprocess.run(["tar", "-xf", str(tar), "-C", "/content"], check=True)
    d = pathlib.Path(f"/content/{pack}")
    n = len(list((d / "frames").iterdir())) if (d / "frames").exists() else 0
    print(f"{pack}: {n} windows unpacked")
'@ | Set-Content -Path $unpack -Encoding UTF8
        Invoke-Colab exec -s $Session -f $unpack --timeout 900
    }

    "train" {
        Invoke-Colab upload -s $Session (Join-Path $repo "scripts\colab_train.py") /content/colab_train.py
        Invoke-Colab exec -s $Session -f (Join-Path $repo "scripts\colab_train.py") --timeout 7200
    }

    "infer" {
        Invoke-Colab exec -s $Session -f (Join-Path $repo "scripts\colab_infer.py") --timeout 7200
    }

    "fetch" {
        Invoke-Colab download -s $Session /content/window_verdicts.jsonl `
            (Join-Path $repo "data\processed\window_verdicts.jsonl")
        Write-Output "fetched -> data\processed\window_verdicts.jsonl"
    }
}

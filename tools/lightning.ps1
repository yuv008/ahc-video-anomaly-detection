# Wrapper for scripts/lightning_run.py.
#
# Lightning credentials are set as USER-scope environment variables, but an already-running
# shell inherited its environment before they existed and cannot see them. This pulls them
# from the user environment at call time and injects them into THIS process only.
#
# The key is never printed - only whether it was found - so it does not end up in logs or
# terminal scrollback.
#
# Usage:
#   .\tools\lightning.ps1 setup
#   .\tools\lightning.ps1 train --machine L4
#   .\tools\lightning.ps1 status

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent

foreach ($name in @("LIGHTNING_API_KEY", "LIGHTNING_USER_ID", "LIGHTNING_TEAMSPACE",
                    "LIGHTNING_USERNAME", "LIGHTNING_ORG", "LIGHTNING_CLOUD_URL")) {
    $val = [Environment]::GetEnvironmentVariable($name, "User")
    if (-not $val) { $val = [Environment]::GetEnvironmentVariable($name, "Machine") }
    if ($val) { Set-Item -Path "Env:$name" -Value $val }
}

$have = @("LIGHTNING_API_KEY", "LIGHTNING_USER_ID") | Where-Object { Test-Path "Env:$_" }
Write-Output ("credentials found: " + ($have -join ", "))

& python (Join-Path $repo "scripts\lightning_run.py") @args

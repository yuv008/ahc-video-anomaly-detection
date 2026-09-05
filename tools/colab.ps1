# Wrapper for `colab` (google-colab-cli) on Windows.
#
# The CLI is documented as Linux/macOS only: it imports the Unix-only `termios` module at
# load time. In practice termios is touched in exactly ONE function - the interactive
# attached-console path, guarded by `is_tty` - which never runs when driving the CLI
# non-interactively. So satisfying the import is enough.
#
# The stubs live in tools/winshim and are added to PYTHONPATH for THIS PROCESS ONLY.
# They are deliberately NOT installed into site-packages: a global `termios` that imports
# successfully on Windows would break other libraries that use
#     try: import termios
#     except ImportError:  # -> "we are on Windows"
# to detect the platform.
#
# Usage:
#   .\tools\colab.ps1 sessions
#   .\tools\colab.ps1 run script.py --gpu T4

$ErrorActionPreference = "Stop"
$shim = Join-Path $PSScriptRoot "winshim"

if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$shim;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $shim
}

& colab @args

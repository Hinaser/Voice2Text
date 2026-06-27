<#
  M0 setup: create an isolated Python venv and install the GPU STT stack.

  Installs:
    - torch (CUDA 12.8 / cu128 wheel)  -> GPU canary, sm_120 support
    - faster-whisper (+ ctranslate2)   -> the STT backend we validate

  Safe to re-run; skips venv creation if it already exists.
#>
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Creating venv at $venv ..."
    python -m venv $venv
}

Write-Host "Upgrading pip ..."
& $py -m pip install --upgrade pip --quiet

# torch first, from the CUDA 12.8 index (default PyPI torch is CPU-only).
Write-Host "Installing torch (cu128) — this is a large download (~2.5 GB) ..."
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128

Write-Host "Installing faster-whisper + ctranslate2 ..."
& $py -m pip install -r (Join-Path $root "stt-sidecar\requirements.txt")

Write-Host ""
Write-Host "Done. Next:"
Write-Host "  1. .\scripts\Make-TestWav.ps1        # generate local test clip"
Write-Host "  2. .\.venv\Scripts\python.exe .\scripts\m0_validate.py"

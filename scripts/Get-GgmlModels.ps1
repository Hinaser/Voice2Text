<#
  Downloads whisper.cpp GGML model files for M0.5.
  These are a DIFFERENT format than the CTranslate2 weights M0 cached.

  By default fetches the quantized q5_0 (~1.1 GB) — fast to download and the
  likely bundle choice. Pass -Full to also get full large-v3 (~3.1 GB) for the
  accuracy comparison.
#>
param(
    [switch]$Full,
    [string]$OutDir = "$PSScriptRoot\..\models"
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

$base = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
$files = @("ggml-large-v3-q5_0.bin")
if ($Full) { $files += "ggml-large-v3.bin" }

foreach ($f in $files) {
    $dest = Join-Path $OutDir $f
    if (Test-Path $dest) { Write-Host "exists: $f (skip)"; continue }
    Write-Host "downloading $f ..."
    # BITS-free, resumable-ish straight download.
    Invoke-WebRequest -Uri "$base/$f" -OutFile $dest
    Write-Host "  -> $dest ($([math]::Round((Get-Item $dest).Length/1MB)) MB)"
}
Write-Host "done."

<#
  Downloads a sherpa-onnx streaming (online) Zipformer English ASR model and
  extracts it under models/. These are the streaming transducer (RNN-T) models —
  the same family as Google Live Transcribe — for low-latency partials.
#>
param(
    [string]$Model = "sherpa-onnx-streaming-zipformer-en-2023-06-26",
    [string]$Tag = "asr-models",   # e.g. "punctuation-models" for punctuation
    [string]$OutDir = "$PSScriptRoot\..\models"
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

$dest = Join-Path $OutDir $Model
if (Test-Path $dest) { Write-Host "exists: $dest (skip)"; return }

$url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/$Tag/$Model.tar.bz2"
$tar = Join-Path $OutDir "$Model.tar.bz2"
Write-Host "downloading $Model ..."
Invoke-WebRequest -Uri $url -OutFile $tar
Write-Host "  -> $([math]::Round((Get-Item $tar).Length/1MB)) MB; extracting ..."
tar -xf $tar -C $OutDir
Remove-Item $tar
Write-Host "extracted to $dest"
Get-ChildItem $dest -Filter *.onnx | Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}} | Format-Table -Auto

<#
  Download every model Voice2Text needs into models/. The weights are NOT in the
  git repo (too large + third-party licenses — see THIRD-PARTY-MODELS.md).

  Usage:
    .\scripts\Fetch-Models.ps1            # everything (incl. GPU models, ~3.5 GB)
    .\scripts\Fetch-Models.ps1 -CpuOnly   # only the streaming/punct/diar models
                                          # (skips Whisper + the LLM; ~0.5 GB)

  Requires `tar` (built into Windows 10/11) for the sherpa archives.
#>
param(
    [switch]$CpuOnly,
    [string]$OutDir = "$PSScriptRoot\..\models"
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

function Get-SherpaArchive($name, $tag) {
    $dest = Join-Path $OutDir $name
    if (Test-Path $dest) { Write-Host "exists: $name (skip)"; return }
    $url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/$tag/$name.tar.bz2"
    $tar = Join-Path $OutDir "$name.tar.bz2"
    Write-Host "downloading $name ..."
    Invoke-WebRequest -Uri $url -OutFile $tar
    tar -xf $tar -C $OutDir
    Remove-Item $tar
    Write-Host "  -> $dest"
}

function Get-File($url, $name) {
    $dest = Join-Path $OutDir $name
    if (Test-Path $dest) { Write-Host "exists: $name (skip)"; return }
    Write-Host "downloading $name ..."
    Invoke-WebRequest -Uri $url -OutFile $dest
    Write-Host ("  -> {0} ({1:N0} MB)" -f $dest, ((Get-Item $dest).Length / 1MB))
}

# --- Always needed: live captions (streaming ASR), punctuation, diarization ---
Get-SherpaArchive 'sherpa-onnx-streaming-zipformer-en-2023-06-26' 'asr-models'
Get-SherpaArchive 'sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12' 'punctuation-models'
Get-File 'https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx' '3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx'

if ($CpuOnly) {
    Write-Host "`nCPU-only models fetched. (Whisper clean transcript + LLM summary skipped.)"
    return
}

# --- GPU features: Whisper clean transcript + local-LLM summary ---
Get-File 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin' 'ggml-large-v3-q5_0.bin'
Get-File 'https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf' 'Qwen2.5-3B-Instruct-Q4_K_M.gguf'

Write-Host "`nAll models fetched into $OutDir"

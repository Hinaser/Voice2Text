<#
  Assemble a self-contained portable Voice2Text distribution: the exe (frontend
  is baked into the binary by Tauri), every runtime DLL it needs (sherpa,
  onnxruntime, and the CUDA runtime for Whisper), and the models. The result
  folder runs on a fresh Windows 11 box with no CUDA toolkit / Python / Rust
  installed (WebView2 is already present on Win11).

  Usage:
    .\scripts\Stage-Portable.ps1            # build + stage into dist\Voice2Text
    .\scripts\Stage-Portable.ps1 -Zip       # also produce dist\Voice2Text-portable.zip
    .\scripts\Stage-Portable.ps1 -SkipBuild # stage from an existing release build
#>
param(
    [string]$OutDir,
    [switch]$Zip,
    [switch]$SkipBuild,
    # CPU-only build: no Whisper sidecar, CUDA DLLs, or large GGML model.
    # Live captions + streaming-saved transcript only; ~300 MB.
    [switch]$Slim
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
# All workspace members build into the shared app/target/release.
$release = Join-Path $repo 'app\target\release'
$models = Join-Path $repo 'models'
if (-not $OutDir) { $OutDir = Join-Path $repo ($Slim ? 'dist\Voice2Text-slim' : 'dist\Voice2Text') }

if (-not $SkipBuild) {
    if ($Slim) {
        Write-Host "==> Building app (CPU-only)..."
        & (Join-Path $PSScriptRoot 'Build-Cuda.ps1') -CrateDir '..\app\src-tauri'
    } else {
        Write-Host "==> Building app + GPU sidecars (whole workspace)..."
        & (Join-Path $PSScriptRoot 'Build-Cuda.ps1') -CrateDir '..\app'
    }
}

$exe = Join-Path $release 'voice2text.exe'
if (-not (Test-Path $exe)) { throw "exe not found: $exe (run without -SkipBuild)" }

Write-Host "==> Staging into $OutDir"
if (Test-Path $OutDir) { Remove-Item $OutDir -Recurse -Force }
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$dstModels = Join-Path $OutDir 'models'
New-Item -ItemType Directory -Path $dstModels -Force | Out-Null

# 1. exe
Copy-Item $exe (Join-Path $OutDir 'Voice2Text.exe')

# 2. app runtime DLLs (produced beside the exe by the build)
$appDlls = @(
    'sherpa-onnx-c-api.dll', 'sherpa-onnx-cxx-api.dll',
    'onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'cargs.dll'
)
foreach ($d in $appDlls) {
    $src = Join-Path $release $d
    if (Test-Path $src) { Copy-Item $src $OutDir } else { Write-Warning "missing app DLL: $d" }
}

# 3. GPU sidecars + CUDA runtime DLLs (skipped in slim/CPU-only builds)
if (-not $Slim) {
    $sidecar = Join-Path $release 'whisper-sidecar.exe'
    if (Test-Path $sidecar) { Copy-Item $sidecar $OutDir } else { Write-Warning "missing whisper-sidecar.exe (build it first)" }
    $llama = Join-Path $release 'llama-sidecar.exe'
    if (Test-Path $llama) { Copy-Item $llama $OutDir } else { Write-Warning "missing llama-sidecar.exe (build it first)" }

    $cudaBin = Join-Path $env:CUDA_PATH 'bin'
    $cudaDlls = @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')
    foreach ($d in $cudaDlls) {
        $src = Join-Path $cudaBin $d
        if (Test-Path $src) { Copy-Item $src $OutDir } else { Write-Warning "missing CUDA DLL: $d (Whisper transcript will be disabled)" }
    }
}

# 4. models (only the ones the app loads — skip the m1 capture-test junk).
# The 1 GB Whisper model is only needed for the GPU clean transcript.
$modelItems = @(
    'sherpa-onnx-streaming-zipformer-en-2023-06-26',
    'sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12',
    '3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx'
)
if (-not $Slim) {
    $modelItems += 'ggml-large-v3-q5_0.bin'            # Whisper clean transcript
    $modelItems += 'Qwen2.5-3B-Instruct-Q4_K_M.gguf'  # local-LLM summary
}
foreach ($m in $modelItems) {
    $src = Join-Path $models $m
    if (Test-Path $src) {
        Write-Host "    + models\$m"
        Copy-Item $src $dstModels -Recurse
    } else {
        Write-Warning "missing model: $m"
    }
}

# 5. a short README
$whisperNote = if ($Slim) {
    "This is the SLIM (CPU-only) build: live captions + a streaming-quality saved`r`ntranscript, no GPU needed. For the clean Whisper transcript, use the full build."
} else {
    "Requires a CUDA GPU for the clean Whisper transcript (bundled, runs locally);`r`nwithout one, turn that off in Settings and live captions still work on CPU."
}
@"
Voice2Text — portable build
Run Voice2Text.exe. Live meeting captions appear in a floating overlay; the
transcript is saved to Documents\Voice2Text (configurable via the gear icon).
Everything runs locally.
$whisperNote
"@ | Set-Content (Join-Path $OutDir 'README.txt')

$size = (Get-ChildItem $OutDir -Recurse | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("==> Staged {0:N0} MB at {1}" -f $size, $OutDir)

if ($Zip) {
    $zipPath = "$OutDir-portable.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Write-Host "==> Zipping to $zipPath ..."
    Compress-Archive -Path "$OutDir\*" -DestinationPath $zipPath
    Write-Host "==> Done: $zipPath"
}

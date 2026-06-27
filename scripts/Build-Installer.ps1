<#
  Build the slim (CPU-only) NSIS installer via Tauri's bundler. The main app no
  longer links Whisper-CUDA, so this installer bundles only the small runtime
  DLLs + streaming/punct/diar models (~650 MB). The Whisper sidecar + GPU bits
  ship separately (full portable) for users who want the clean transcript.

  Usage:  .\scripts\Build-Installer.ps1
  Output: app\src-tauri\target\release\bundle\nsis\*-setup.exe

  KNOWN BLOCKER (2026-06-28): on this machine `tauri build` fails inside the
  tauri_build build script with "Access denied (os error 5)" as soon as ANY
  bundle.resources entry is present (even a single in-crate file) — reproduced
  with files, directories, and external paths. Until that's resolved, ship the
  verified self-contained distribution from Stage-Portable.ps1 instead
  (dist\Voice2Text-slim for CPU-only, dist\Voice2Text for the full GPU build).
#>
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$appDir = Join-Path $repo 'app'

# Same toolchain env as Build-Cuda (vcvars INCLUDE for sherpa bindgen, Ninja,
# libclang). CUDA vars are harmless for this CPU-only build.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $vcvars) {
        cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
            if ($_ -match '^(.*?)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
        }
        $ninja = Get-ChildItem "$vsPath\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 -Expand FullName
        if ($ninja) { $env:Path = "$(Split-Path -Parent $ninja);$env:Path" }
    }
}
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
$env:Path = "C:\Program Files\CMake\bin;$env:Path"
Remove-Item env:CMAKE_GENERATOR_INSTANCE -ErrorAction SilentlyContinue
Remove-Item env:CMAKE_GENERATOR_PLATFORM -ErrorAction SilentlyContinue
Remove-Item env:CMAKE_GENERATOR_TOOLSET -ErrorAction SilentlyContinue

# Produce the runtime DLLs, then stage them OUTSIDE target/ so Tauri's resource
# copy doesn't copy them onto themselves (the os-error-5 we hit referencing
# target/release directly).
$srcTauri = Join-Path $appDir 'src-tauri'
$release = Join-Path $appDir 'target\release'   # workspace target
Write-Host "==> Pre-build to produce runtime DLLs..."
& (Join-Path $PSScriptRoot 'Build-Cuda.ps1') -CrateDir '..\app\src-tauri'

# Tauri rejects resource paths that escape the crate root (../../models -> os
# error 5), so everything the installer bundles is staged under src-tauri\payload.
$payload = Join-Path $srcTauri 'payload'
Remove-Item $payload -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $payload -Force | Out-Null
foreach ($d in 'sherpa-onnx-c-api.dll','sherpa-onnx-cxx-api.dll','onnxruntime.dll','onnxruntime_providers_shared.dll','cargs.dll') {
    Copy-Item (Join-Path $release $d) $payload -Force
}
$payloadModels = Join-Path $payload 'models'
New-Item -ItemType Directory -Path $payloadModels -Force | Out-Null
$models = Join-Path $repo 'models'
foreach ($m in 'sherpa-onnx-streaming-zipformer-en-2023-06-26','sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12','3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx') {
    Copy-Item (Join-Path $models $m) $payloadModels -Recurse -Force
}

Write-Host "==> Building NSIS installer (Tauri bundler; first run downloads NSIS)..."
Push-Location $appDir
try {
    & npx --no-install tauri build --bundles nsis --config src-tauri/installer.conf.json
    if ($LASTEXITCODE -ne 0) { throw "tauri build failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}

$out = Get-ChildItem (Join-Path $appDir 'target\release\bundle\nsis') -Filter *-setup.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if ($out) {
    Write-Host ("==> Installer: {0} ({1:N0} MB)" -f $out.FullName, ($out.Length / 1MB))
} else {
    Write-Warning "No installer produced — check the tauri build output above."
}

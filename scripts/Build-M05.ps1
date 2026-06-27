<#
  Builds the M0.5 native whisper.cpp + CUDA harness.

  whisper-rs compiles whisper.cpp from source via CMake, and the CUDA backend
  needs nvcc + the MSVC C++ toolchain on PATH. So we:
    1. locate VS 2022 via vswhere and import its x64 dev environment (vcvars64)
    2. put CMake on PATH
    3. cargo build --release
#>
$ErrorActionPreference = 'Stop'
$crate = Join-Path (Split-Path -Parent $PSScriptRoot) "m05-whisper-cuda"

# --- 1. import the VS x64 build environment into this session ---
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere not found at $vswhere" }
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw "No VS install with the C++ toolchain found." }
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

Write-Host "Importing VS env from: $vcvars"
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
    if ($_ -match '^(.*?)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}

# --- 2. CMake + libclang on PATH/env ---
$cmakeBin = "C:\Program Files\CMake\bin"
if (Test-Path $cmakeBin) { $env:Path = "$cmakeBin;$env:Path" }

# bindgen (whisper-rs-sys) needs libclang from LLVM.
$llvmBin = "C:\Program Files\LLVM\bin"
if (Test-Path (Join-Path $llvmBin "libclang.dll")) {
    $env:LIBCLANG_PATH = $llvmBin
} else {
    throw "libclang.dll not found in $llvmBin — install LLVM first."
}

# Build CUDA kernels specifically for Blackwell sm_120. CMake reads CUDAARCHS as
# the default for CMAKE_CUDA_ARCHITECTURES, so this avoids a 'no kernel image'
# runtime failure on the RTX 5080.
$env:CUDAARCHS = "120"

# This machine's MSVC (14.50) is newer than CUDA 12.8 officially supports; allow
# nvcc to proceed past its host-compiler version gate.
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler"

# Use the Ninja generator instead of the Visual Studio generator. The VS
# generator needs CUDA's MSBuild integration installed into VS ("No CUDA toolset
# found"), which the CUDA 12.8 installer did NOT add to this new VS 18. Ninja
# invokes nvcc directly and avoids that entirely.
$ninjaExe = Get-ChildItem "$vsPath\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 -Expand FullName
if (-not $ninjaExe) { $ninjaExe = (Get-Command ninja -ErrorAction SilentlyContinue)?.Source }
if (-not $ninjaExe) { throw "ninja.exe not found (needed for the CUDA build)." }
$env:Path = "$(Split-Path -Parent $ninjaExe);$env:Path"
$env:CMAKE_GENERATOR = "Ninja"
# vcvars exports these VS-generator-only vars; they conflict with Ninja
# ("generator Ninja does not support instance specification"). Clear them.
Remove-Item env:CMAKE_GENERATOR_INSTANCE -ErrorAction SilentlyContinue
Remove-Item env:CMAKE_GENERATOR_PLATFORM -ErrorAction SilentlyContinue
Remove-Item env:CMAKE_GENERATOR_TOOLSET -ErrorAction SilentlyContinue

Write-Host "cl   : $((Get-Command cl).Source)"
Write-Host "cmake: $((Get-Command cmake).Source)  ($(cmake --version | Select-Object -First 1))"
Write-Host "nvcc : $((Get-Command nvcc).Source)"
Write-Host "ninja: $ninjaExe"
Write-Host "CUDA_PATH    : $env:CUDA_PATH"
Write-Host "LIBCLANG_PATH: $env:LIBCLANG_PATH"
Write-Host "CUDAARCHS    : $env:CUDAARCHS"
Write-Host "CMAKE_GENERATOR: $env:CMAKE_GENERATOR"

# --- 3. build ---
Write-Host "`nBuilding (this compiles whisper.cpp + CUDA kernels; first build is slow)..."
Push-Location $crate
try {
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed with exit code $LASTEXITCODE" }
    Write-Host "`nBUILD OK"
} finally {
    Pop-Location
}

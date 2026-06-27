<#
  Generic CUDA-crate builder — same toolchain recipe proven in M0.5
  (Build-M05.ps1), parameterized by crate directory so it can build any crate
  that links whisper-rs + CUDA (m05-whisper-cuda, live-stt, eventually src-tauri).

  Usage:  .\scripts\Build-Cuda.ps1 -CrateDir ..\live-stt
#>
param(
    [Parameter(Mandatory = $true)][string]$CrateDir,
    # cargo subcommand + args to run in the crate dir (default: a release build).
    # e.g. -CargoArgs test,--release
    [string[]]$CargoArgs = @('build', '--release')
)
$ErrorActionPreference = 'Stop'
$crate = Resolve-Path (Join-Path $PSScriptRoot $CrateDir) -ErrorAction SilentlyContinue
if (-not $crate) { $crate = Resolve-Path $CrateDir }

# 1. VS x64 build environment
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
    if ($_ -match '^(.*?)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}

# 2. CMake + libclang
$env:Path = "C:\Program Files\CMake\bin;$env:Path"
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"

# 3. Blackwell sm_120 + new-MSVC bypass + Ninja generator (see Build-M05.ps1)
$env:CUDAARCHS = "120"
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler"
$ninjaExe = Get-ChildItem "$vsPath\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 -Expand FullName
$env:Path = "$(Split-Path -Parent $ninjaExe);$env:Path"
$env:CMAKE_GENERATOR = "Ninja"
Remove-Item env:CMAKE_GENERATOR_INSTANCE -ErrorAction SilentlyContinue
Remove-Item env:CMAKE_GENERATOR_PLATFORM -ErrorAction SilentlyContinue
Remove-Item env:CMAKE_GENERATOR_TOOLSET -ErrorAction SilentlyContinue

Write-Host "Running 'cargo $($CargoArgs -join ' ')' in $crate (CUDA; first build compiles kernels, ~minutes)..."
Push-Location $crate
try {
    cargo @CargoArgs
    if ($LASTEXITCODE -ne 0) { throw "cargo $($CargoArgs -join ' ') failed with exit code $LASTEXITCODE" }
    Write-Host "`nBUILD OK"
} finally {
    Pop-Location
}

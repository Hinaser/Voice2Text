<#
  Runs the M1 dual-track capture harness.
  Usage:  .\scripts\Run-M1.ps1 [seconds]
  Records mic (raw) + mic (AEC) + system loopback to models\m1\*.wav.

  Unlike the STT harness, this needs no CUDA DLLs — it uses only system audio
  APIs, so the exe runs standalone.
#>
param([int]$Seconds = 60)
$exe = "$PSScriptRoot\..\m1-capture\target\release\m1-capture.exe"
if (-not (Test-Path $exe)) { throw "Build first: cd m1-capture; cargo build --release" }
& $exe $Seconds

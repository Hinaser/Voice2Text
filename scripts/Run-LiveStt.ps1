<#
  Runs the live system-audio -> text demo.
  Usage:  .\scripts\Run-LiveStt.ps1 [seconds] [model.bin] [transcript.txt]

  Puts the CUDA runtime DLLs on PATH so the exe can load cudart/cublas (in the
  shipped app these will sit next to the exe).
#>
param(
    [int]$Seconds = 120,
    [string]$Model = "",
    [string]$Transcript = ""
)
$cudaBin = if ($env:CUDA_PATH) { "$env:CUDA_PATH\bin" } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin" }
$env:Path = "$cudaBin;$env:Path"

$exe = "$PSScriptRoot\..\live-stt\target\release\live-stt.exe"
if (-not (Test-Path $exe)) { throw "Build first: .\scripts\Build-Cuda.ps1 -CrateDir ..\live-stt" }

$argList = @($Seconds)
if ($Model) { $argList += $Model } elseif ($Transcript) { $argList += "$PSScriptRoot\..\models\ggml-large-v3-q5_0.bin" }
if ($Transcript) { $argList += $Transcript }
& $exe @argList

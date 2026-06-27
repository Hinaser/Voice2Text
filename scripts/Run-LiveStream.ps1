<#
  Runs the Google-style live streaming captions (CPU-only).
  Usage:  .\scripts\Run-LiveStream.ps1 [seconds] [transcript.txt]

  Adds the sherpa-onnx + onnxruntime DLLs (next to the build) to PATH.
#>
param(
    [int]$Seconds = 300,
    [string]$Transcript = ""
)
$root = Split-Path -Parent $PSScriptRoot
$deps = Join-Path $root "live-stream\target\release\deps"
$env:Path = "$deps;$env:Path"

$exe = Join-Path $root "live-stream\target\release\live-stream.exe"
if (-not (Test-Path $exe)) { throw "Build first: .\scripts\Build-Cuda.ps1 -CrateDir ..\live-stream" }

$model = Join-Path $root "models\sherpa-onnx-streaming-zipformer-en-2023-06-26"
$argList = @($Seconds, $model)
if ($Transcript) { $argList += $Transcript }
& $exe @argList

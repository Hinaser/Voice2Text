<#
  Runs the M0.5 harness. Puts the CUDA runtime DLLs on PATH so the binary can
  load cudart/cublas at runtime (in the shipped app these DLLs sit next to the
  exe; here we borrow them from the CUDA toolkit).
#>
param(
    [string]$Model = "$PSScriptRoot\..\models\ggml-large-v3-q5_0.bin",
    [string]$Wav   = "$PSScriptRoot\..\models\test.wav"
)
$ErrorActionPreference = 'Stop'
$cudaBin = "$env:CUDA_PATH\bin"
if (-not $env:CUDA_PATH) { $cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin" }
$env:Path = "$cudaBin;$env:Path"

$exe = "$PSScriptRoot\..\m05-whisper-cuda\target\release\m05-whisper-cuda.exe"
& $exe $Model $Wav

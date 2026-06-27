<#
  Generates a local test WAV using the Windows built-in speech synthesizer.
  Fully offline — no network, no audio leaves the machine.

  We synthesize a known sentence so the M0 validator can check transcription
  accuracy (word error) against ground truth, not just that *something* ran.
#>
param(
    [string]$OutFile = "$PSScriptRoot\..\models\test.wav",
    [string]$Text = "Let's review the quarterly roadmap and align on the deliverables for the next sprint before we schedule the follow-up meeting."
)

$ErrorActionPreference = 'Stop'

$outDir = Split-Path -Parent $OutFile
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }

Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

# 16 kHz, 16-bit, mono — matches what the STT pipeline will feed in production.
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth.SetOutputToWaveFile($OutFile, $fmt)
$synth.Speak($Text)
$synth.Dispose()

# Save the ground-truth text next to the WAV for the accuracy check.
$Text | Set-Content -Path ([System.IO.Path]::ChangeExtension($OutFile, ".txt")) -Encoding UTF8

Write-Host "Wrote $OutFile"
Write-Host "Ground truth: $Text"

<#
  Build, stage and publish a Voice2Text portable bundle to a GitHub Release,
  creating and pushing the version tag in the process.

  The full (GPU) bundle is ALWAYS built and attached. Pass -IncludeSlim to also
  build and attach the CPU-only bundle alongside it.

  Releases ship BINARIES ONLY (exe + runtime/CUDA DLLs + GPU sidecars +
  Fetch-Models.ps1) — not the model weights. Keeps each asset well under
  GitHub's 2 GB/file limit and avoids redistributing the Qwen weights
  (non-commercial license). Users run the bundled Fetch-Models.ps1 once.

  Requires the GitHub CLI (`gh`), authenticated (`gh auth login`).

  Usage:
    .\scripts\Publish-Release.ps1 -Tag v0.1.0                 # GPU bundle only
    .\scripts\Publish-Release.ps1 -Tag v0.1.0 -IncludeSlim    # GPU bundle + CPU-only bundle
    .\scripts\Publish-Release.ps1 -Tag v0.1.0 -Draft          # create as a draft to review first
    .\scripts\Publish-Release.ps1 -Tag v0.1.0 -SkipBuild      # reuse the last build
    .\scripts\Publish-Release.ps1 -Tag v0.1.0 -SkipTag        # tag already exists; don't create/push it
    .\scripts\Publish-Release.ps1 -Tag v0.1.0 -NotesFile notes.md
#>
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [string]$Title,
    [string]$Notes,
    [string]$NotesFile,
    # Also build+attach the CPU-only bundle. The full GPU bundle is built either
    # way — this only ADDS the slim one; it never replaces the full bundle.
    [switch]$IncludeSlim,
    # Reuse an existing release build instead of recompiling.
    [switch]$SkipBuild,
    # Skip creating/pushing the git tag (use when the tag already exists).
    [switch]$SkipTag,
    [switch]$Draft,
    [switch]$Prerelease
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI 'gh' not found. Install it and run 'gh auth login' first."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git not found on PATH."
}
if (-not $Title) { $Title = "Voice2Text $Tag" }

# --- Create + push the version tag (annotated, at the current HEAD) -----------
if (-not $SkipTag) {
    $dirty = git status --porcelain
    if ($dirty) {
        Write-Warning "Working tree has uncommitted changes; the tag will point at the last commit, NOT these edits. Commit first if you want them in the release."
    }
    $exists = git tag --list $Tag
    if ($exists) {
        Write-Host "==> Tag $Tag already exists locally; reusing it."
    } else {
        Write-Host "==> Creating annotated tag $Tag at HEAD..."
        git tag -a $Tag -m $Title
        if ($LASTEXITCODE -ne 0) { throw "git tag failed ($LASTEXITCODE)" }
    }
    Write-Host "==> Pushing tag $Tag to origin..."
    git push origin $Tag
    if ($LASTEXITCODE -ne 0) { throw "git push of tag failed ($LASTEXITCODE)" }
}

# --- Stage the bundle(s): binaries only, zipped -------------------------------
$stage = Join-Path $PSScriptRoot 'Stage-Portable.ps1'
$assets = @()

$common = @{ NoModels = $true; Zip = $true }
if ($SkipBuild) { $common.SkipBuild = $true }

Write-Host "==> Staging full (GPU) bundle..."
& $stage @common
$assets += Join-Path $repo 'dist\Voice2Text-portable.zip'

if ($IncludeSlim) {
    Write-Host "==> Staging slim (CPU-only) bundle..."
    & $stage @common -Slim
    $assets += Join-Path $repo 'dist\Voice2Text-slim-portable.zip'
}

foreach ($a in $assets) {
    if (-not (Test-Path $a)) { throw "expected asset not found: $a" }
    $mb = (Get-Item $a).Length / 1MB
    if ($mb -ge 2048) { throw ("asset {0} is {1:N0} MB — exceeds GitHub's 2 GB/file limit" -f $a, $mb) }
    Write-Host ("    asset: {0} ({1:N0} MB)" -f (Split-Path $a -Leaf), $mb)
}

# --- Create the release and upload the assets ---------------------------------
$ghArgs = @('release', 'create', $Tag, '--title', $Title)
if ($NotesFile) {
    $ghArgs += @('--notes-file', $NotesFile)
} elseif ($Notes) {
    $ghArgs += @('--notes', $Notes)
} else {
    $ghArgs += @('--generate-notes')   # from commits since the last tag
}
if ($Draft) { $ghArgs += '--draft' }
if ($Prerelease) { $ghArgs += '--prerelease' }
$ghArgs += $assets

Write-Host "==> gh $($ghArgs -join ' ')"
& gh @ghArgs
if ($LASTEXITCODE -ne 0) { throw "gh release create failed ($LASTEXITCODE)" }
Write-Host "`n==> Release $Tag published."

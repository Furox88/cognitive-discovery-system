# Local PyPI publish script.
#
# Usage:
#   .\scripts\publish.ps1                  # interactive, builds + uploads
#   .\scripts\publish.ps1 -Version 0.8.1   # explicit version check
#   .\scripts\publish.ps1 -DryRun          # build but don't upload
#
# Requirements: build, twine (both `pip install build twine`).
# Token: stored at %USERPROFILE%\.pypi-token (PyPI API token, pypi-...).
#        Create the file with: Set-Content "$env:USERPROFILE\.pypi-token" -Value "pypi-Ag..."
#
# Why local-only: this project does not auto-publish from CI. Tag pushes
# stay clean. We push the wheel + sdist manually after a green test pass.

[CmdletBinding()]
param(
    [string]$Version,
    [switch]$DryRun,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# 1. Sanity: clean working tree, correct branch
$status = git status --porcelain
if ($status) {
    Write-Host "Working tree is dirty. Commit/stash first." -ForegroundColor Red
    git status --short
    exit 1
}

$branch = git branch --show-current
if ($branch -ne "main") {
    Write-Host "Not on main (currently on $branch). Publish from main only." -ForegroundColor Red
    exit 1
}

# 2. Verify local version matches what we expect
$pyprojectVersion = (Select-String -Path pyproject.toml -Pattern '^version\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
if ($Version -and $Version -ne $pyprojectVersion) {
    Write-Host "Version mismatch: pyproject=$pyprojectVersion, requested=$Version" -ForegroundColor Red
    exit 1
}
Write-Host "Version: $pyprojectVersion" -ForegroundColor Cyan

# 3. Run tests
if (-not $SkipTests) {
    Write-Host "Running tests..." -ForegroundColor Cyan
    python -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Tests failed. Fix first, then publish." -ForegroundColor Red
        exit 1
    }
}

# 4. Clean old dist artifacts, build fresh
if (Test-Path dist) {
    Get-ChildItem dist | Remove-Item -Recurse -Force
}
Write-Host "Building sdist + wheel..." -ForegroundColor Cyan
python -m build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed." -ForegroundColor Red
    exit 1
}

# 5. Show what we're about to upload
Write-Host ""
Write-Host "Built artifacts:" -ForegroundColor Cyan
Get-ChildItem dist | ForEach-Object { Write-Host "  $($_.Name) ($([math]::Round($_.Length/1KB, 1)) kB)" }

# 6. Upload (unless dry-run)
if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run: skipping upload." -ForegroundColor Yellow
    exit 0
}

$tokenFile = Join-Path $env:USERPROFILE ".pypi-token"
if (-not (Test-Path $tokenFile)) {
    Write-Host "No token file at $tokenFile. See script header for setup." -ForegroundColor Red
    exit 1
}
$token = Get-Content $tokenFile -Raw | ForEach-Object { $_.Trim() }

$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = $token
Write-Host ""
Write-Host "Uploading to PyPI..." -ForegroundColor Cyan
python -m twine upload dist/*

# 7. Cleanup env (defense in depth)
Remove-Item Env:\TWINE_USERNAME -ErrorAction SilentlyContinue
Remove-Item Env:\TWINE_PASSWORD -ErrorAction SilentlyContinue

if ($LASTEXITCODE -ne 0) {
    Write-Host "Upload failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Done. v$pyprojectVersion published." -ForegroundColor Green
Write-Host "Next: tag the release and create a GitHub release." -ForegroundColor Cyan
Write-Host "  git tag -a v$pyprojectVersion -m 'Release v$pyprojectVersion'"
Write-Host "  git push origin v$pyprojectVersion"
Write-Host "  gh release create v$pyprojectVersion --generate-notes"

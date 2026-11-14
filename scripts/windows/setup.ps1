[CmdletBinding()]
param(
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

& (Join-Path $PSScriptRoot "check_system.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "Preflight failed. Fix the failed checks above, then run setup again."
}

Push-Location $repoRoot
try {
    Write-Host ""
    Write-Host "Installing the pinned Isaac Lab environment..." -ForegroundColor Cyan
    & uv sync --extra isaacsim
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed with exit code $LASTEXITCODE."
    }

    $urllib3Version = (& uv run --extra isaacsim python -c "import urllib3; print(urllib3.__version__)")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not verify the resolved urllib3 version."
    }
    Write-Host "Resolved urllib3 $urllib3Version (Isaac Lab requires 2.8.0 or newer)." -ForegroundColor Green

    if ($SmokeTest) {
        Write-Host ""
        Write-Host "Running the 16-environment smoke test..." -ForegroundColor Cyan
        & uv run --extra isaacsim isaaclab zero_agent --task SO101-Grasp-Cube-State-v0 --num_envs 16 --max_steps 200 --visualizer none presets=isaacsim_physx
        if ($LASTEXITCODE -ne 0) {
            throw "Isaac Lab smoke test failed with exit code $LASTEXITCODE."
        }
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
if (-not $SmokeTest) {
    Write-Host "Next: run .\scripts\windows\setup.ps1 -SmokeTest"
}

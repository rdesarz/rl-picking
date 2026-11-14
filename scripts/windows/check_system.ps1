[CmdletBinding()]
param(
    [int]$MinimumFreeDiskGb = 100
)

$ErrorActionPreference = "Stop"
$script:Failures = 0
$script:Warnings = 0

function Write-Result {
    param(
        [string]$Name,
        [ValidateSet("PASS", "WARN", "FAIL")]
        [string]$Status,
        [string]$Details
    )

    $color = switch ($Status) {
        "PASS" { "Green" }
        "WARN" { "Yellow" }
        "FAIL" { "Red" }
    }

    Write-Host ("{0,-5} {1,-16} {2}" -f $Status, $Name, $Details) -ForegroundColor $color

    if ($Status -eq "FAIL") {
        $script:Failures += 1
    }
    elseif ($Status -eq "WARN") {
        $script:Warnings += 1
    }
}

Write-Host "SO-101 Isaac Lab Windows preflight" -ForegroundColor Cyan
Write-Host ""

$os = Get-CimInstance Win32_OperatingSystem
$isWindows11 = ([version]$os.Version -ge [version]"10.0.22000")
Write-Result "Windows" $(if ($isWindows11) { "PASS" } else { "FAIL" }) "$($os.Caption), build $($os.BuildNumber) (Windows 11 required)"

$isAmd64 = $env:PROCESSOR_ARCHITECTURE -eq "AMD64"
Write-Result "Architecture" $(if ($isAmd64) { "PASS" } else { "FAIL" }) "$($env:PROCESSOR_ARCHITECTURE) (AMD64 required)"

$computer = Get-CimInstance Win32_ComputerSystem
$logicalCores = [int]$computer.NumberOfLogicalProcessors
Write-Result "CPU threads" $(if ($logicalCores -ge 4) { "PASS" } else { "FAIL" }) "$logicalCores logical processors (4 minimum)"

$ramGb = [math]::Round($computer.TotalPhysicalMemory / 1GB, 1)
$ramStatus = if ($ramGb -ge 32) { "PASS" } else { "FAIL" }
Write-Result "System RAM" $ramStatus "$ramGb GB (32 GB minimum)"

$longPaths = (Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled
Write-Result "Long paths" $(if ($longPaths -eq 1) { "PASS" } else { "WARN" }) $(if ($longPaths -eq 1) { "enabled" } else { "disabled; enable before installing" })

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$driveName = (Split-Path -Qualifier $repoRoot).TrimEnd("\").TrimEnd(":")
$drive = Get-PSDrive -Name $driveName
$freeDiskGb = [math]::Round($drive.Free / 1GB, 1)
$diskStatus = if ($freeDiskGb -ge $MinimumFreeDiskGb) { "PASS" } elseif ($freeDiskGb -ge 50) { "WARN" } else { "FAIL" }
Write-Result "Free disk" $diskStatus "$freeDiskGb GB on $($drive.Name): ($MinimumFreeDiskGb GB recommended)"

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($null -eq $nvidiaSmi) {
    Write-Result "NVIDIA GPU" "FAIL" "nvidia-smi was not found; install the NVIDIA notebook driver"
}
else {
    $gpuRow = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | Select-Object -First 1)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($gpuRow)) {
        Write-Result "NVIDIA GPU" "FAIL" "nvidia-smi could not query the GPU"
    }
    else {
        $gpuParts = $gpuRow.Split(",") | ForEach-Object { $_.Trim() }
        $gpuName = $gpuParts[0]
        $driverVersion = $gpuParts[1]
        $vramMb = [int]$gpuParts[2]
        $vramGb = [math]::Round($vramMb / 1024, 1)

        $isRtx = $gpuName -match "RTX"
        Write-Result "NVIDIA GPU" $(if ($isRtx) { "PASS" } else { "WARN" }) "$gpuName"

        $driverOk = ([version]$driverVersion -ge [version]"595.97")
        Write-Result "GPU driver" $(if ($driverOk) { "PASS" } else { "WARN" }) "$driverVersion (595.97 is NVIDIA's tested Windows version)"

        $vramStatus = if ($vramGb -ge 16) { "PASS" } else { "FAIL" }
        Write-Result "GPU memory" $vramStatus "$vramGb GB (16 GB minimum)"
    }
}

$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uv) {
    Write-Result "uv" "FAIL" "not found; install uv and open a new PowerShell window"
}
else {
    $uvVersion = (& uv --version)
    Write-Result "uv" "PASS" $uvVersion
}

Write-Host ""
if ($script:Failures -gt 0) {
    Write-Host "$($script:Failures) required check(s) failed and $($script:Warnings) warning(s) were found." -ForegroundColor Red
    exit 1
}

Write-Host "Preflight passed with $($script:Warnings) warning(s)." -ForegroundColor Green
exit 0

# PowerShell script to start webcam streaming for WSL2
# Run this on Windows to stream webcam via RTSP

<#
.SYNOPSIS
    Start webcam streaming from Windows to WSL2 via RTSP.

.DESCRIPTION
    This script:
    1. Lists available cameras
    2. Starts MediaMTX RTSP server
    3. Starts ffmpeg to stream webcam

.PARAMETER CameraName
    Name of the camera to use. If not specified, lists available cameras.

.PARAMETER MediaMTXPath
    Path to MediaMTX executable. Default: C:\mediamtx\mediamtx.exe

.EXAMPLE
    .\start_webcam_stream.ps1
    Lists available cameras

.EXAMPLE
    .\start_webcam_stream.ps1 -CameraName "Integrated Camera"
    Starts streaming with the specified camera
#>

param(
    [string]$CameraName = "",
    [string]$MediaMTXPath = "C:\mediamtx\mediamtx.exe",
    [int]$Width = 640,
    [int]$Height = 480,
    [int]$FPS = 30
)

$ErrorActionPreference = "Stop"

function Test-Prerequisites {
    Write-Host "`n=== Checking Prerequisites ===" -ForegroundColor Cyan

    # Check ffmpeg
    try {
        $null = Get-Command ffmpeg -ErrorAction Stop
        Write-Host "[OK] ffmpeg found" -ForegroundColor Green
    }
    catch {
        Write-Host "[MISSING] ffmpeg not found" -ForegroundColor Red
        Write-Host "  Download from: https://www.gyan.dev/ffmpeg/builds/"
        Write-Host "  Extract to C:\ffmpeg and add C:\ffmpeg\bin to PATH"
        return $false
    }

    # Check MediaMTX
    if (Test-Path $MediaMTXPath) {
        Write-Host "[OK] MediaMTX found at $MediaMTXPath" -ForegroundColor Green
    }
    else {
        Write-Host "[MISSING] MediaMTX not found at $MediaMTXPath" -ForegroundColor Red
        Write-Host "  Download from: https://github.com/bluenviron/mediamtx/releases"
        Write-Host "  Extract to C:\mediamtx"
        return $false
    }

    return $true
}

function Get-AvailableCameras {
    Write-Host "`n=== Available Cameras ===" -ForegroundColor Cyan

    $output = ffmpeg -list_devices true -f dshow -i dummy 2>&1 | Out-String

    # Extract video devices
    $cameras = @()
    $inVideoSection = $false

    foreach ($line in $output -split "`n") {
        if ($line -match "DirectShow video devices") {
            $inVideoSection = $true
            continue
        }
        if ($line -match "DirectShow audio devices") {
            $inVideoSection = $false
        }
        if ($inVideoSection -and $line -match '"([^"]+)"') {
            $cameras += $Matches[1]
        }
    }

    if ($cameras.Count -eq 0) {
        Write-Host "No cameras found!" -ForegroundColor Red
        return @()
    }

    Write-Host "Found cameras:" -ForegroundColor Green
    for ($i = 0; $i -lt $cameras.Count; $i++) {
        Write-Host "  [$i] $($cameras[$i])"
    }

    return $cameras
}

function Start-WebcamStream {
    param(
        [string]$Camera,
        [int]$Width,
        [int]$Height,
        [int]$FPS
    )

    Write-Host "`n=== Starting Webcam Stream ===" -ForegroundColor Cyan
    Write-Host "Camera: $Camera"
    Write-Host "Resolution: ${Width}x${Height}"
    Write-Host "FPS: $FPS"

    # Start MediaMTX in background
    Write-Host "`nStarting MediaMTX RTSP server..." -ForegroundColor Yellow
    $mediamtxProcess = Start-Process -FilePath $MediaMTXPath -PassThru -WindowStyle Minimized

    Start-Sleep -Seconds 2

    if ($mediamtxProcess.HasExited) {
        Write-Host "[ERROR] MediaMTX failed to start" -ForegroundColor Red
        return
    }

    Write-Host "[OK] MediaMTX started (PID: $($mediamtxProcess.Id))" -ForegroundColor Green

    # Build ffmpeg command
    $ffmpegArgs = @(
        "-f", "dshow",
        "-i", "video=$Camera",
        "-framerate", "$FPS",
        "-video_size", "${Width}x${Height}",
        "-vcodec", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-f", "rtsp",
        "rtsp://localhost:8554/webcam"
    )

    Write-Host "`nStarting ffmpeg stream..." -ForegroundColor Yellow
    Write-Host "ffmpeg $($ffmpegArgs -join ' ')"

    Write-Host "`n=== Stream Active ===" -ForegroundColor Green
    Write-Host "RTSP URL: rtsp://localhost:8554/webcam"
    Write-Host "`nIn WSL2, run:"
    Write-Host '  export WEBCAM_RTSP_URL="rtsp://<host_ip>:8554/webcam"' -ForegroundColor Cyan
    Write-Host "  (Get host IP with: ip route list default | awk '{print `$3}')"
    Write-Host "`nPress Ctrl+C to stop streaming..."

    try {
        # Run ffmpeg (this will block until Ctrl+C)
        & ffmpeg @ffmpegArgs
    }
    finally {
        Write-Host "`nStopping MediaMTX..."
        Stop-Process -Id $mediamtxProcess.Id -ErrorAction SilentlyContinue
        Write-Host "Stream stopped."
    }
}

# Main
if (-not (Test-Prerequisites)) {
    exit 1
}

$cameras = Get-AvailableCameras

if ($cameras.Count -eq 0) {
    exit 1
}

if ([string]::IsNullOrEmpty($CameraName)) {
    Write-Host "`nUsage:" -ForegroundColor Yellow
    Write-Host '  .\start_webcam_stream.ps1 -CameraName "Camera Name"'
    Write-Host "`nExample:"
    Write-Host "  .\start_webcam_stream.ps1 -CameraName `"$($cameras[0])`""
    exit 0
}

# Verify camera exists
if ($cameras -notcontains $CameraName) {
    Write-Host "`n[ERROR] Camera '$CameraName' not found" -ForegroundColor Red
    Write-Host "Available cameras:"
    foreach ($cam in $cameras) {
        Write-Host "  - $cam"
    }
    exit 1
}

Start-WebcamStream -Camera $CameraName -Width $Width -Height $Height -FPS $FPS

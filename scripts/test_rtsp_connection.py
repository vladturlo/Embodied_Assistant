#!/usr/bin/env python3
"""Test RTSP connection from WSL2 to Windows webcam stream.

This script tests the RTSP connection to verify that the webcam stream
from Windows is accessible in WSL2.

Prerequisites (Windows side):
1. MediaMTX running: cd C:\mediamtx && .\mediamtx.exe
2. ffmpeg streaming: ffmpeg -f dshow -i video="Camera Name" -vcodec libx264 \
   -preset ultrafast -tune zerolatency -f rtsp rtsp://localhost:8554/webcam

Usage:
    # Auto-detect host IP
    python scripts/test_rtsp_connection.py

    # Specify custom RTSP URL
    python scripts/test_rtsp_connection.py rtsp://172.25.192.1:8554/webcam
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2


def get_wsl_host_ip() -> str:
    """Get the Windows host IP address from WSL2.

    Returns:
        The IP address of the Windows host.
    """
    try:
        result = subprocess.run(
            ["ip", "route", "list", "default"],
            capture_output=True,
            text=True,
            check=True
        )
        # Output format: "default via 172.25.192.1 dev eth0"
        parts = result.stdout.strip().split()
        if len(parts) >= 3:
            return parts[2]
    except Exception as e:
        print(f"Warning: Could not get host IP: {e}")

    # Fallback: try reading from resolv.conf
    try:
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                if line.startswith("nameserver"):
                    return line.split()[1]
    except Exception:
        pass

    return "172.25.192.1"  # Common default


def test_rtsp_connection(rtsp_url: str) -> bool:
    """Test RTSP connection and capture a frame.

    Args:
        rtsp_url: The RTSP URL to connect to.

    Returns:
        True if connection and frame capture succeed.
    """
    print(f"\n{'='*60}")
    print("RTSP Connection Test")
    print(f"{'='*60}")
    print(f"URL: {rtsp_url}")

    print("\nConnecting...")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("\n[FAIL] Could not connect to RTSP stream")
        print("\nTroubleshooting:")
        print("  1. Is MediaMTX running on Windows?")
        print("  2. Is ffmpeg streaming the webcam?")
        print("  3. Is the IP address correct?")
        print(f"     Expected host IP: {get_wsl_host_ip()}")
        return False

    print("Connected! Reading frame...")

    # Try to read a frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("\n[FAIL] Connected but could not read frame")
        print("The stream may not be producing frames yet.")
        return False

    # Save the frame
    temp_path = Path(tempfile.mkdtemp()) / "rtsp_test_capture.jpg"
    cv2.imwrite(str(temp_path), frame)

    print(f"\n[PASS] Successfully captured frame!")
    print(f"  Frame shape: {frame.shape}")
    print(f"  Frame saved: {temp_path}")

    return True


def test_rtsp_continuous(rtsp_url: str, num_frames: int = 10) -> bool:
    """Test continuous frame reading from RTSP stream.

    Args:
        rtsp_url: The RTSP URL to connect to.
        num_frames: Number of frames to read.

    Returns:
        True if all frames are read successfully.
    """
    print(f"\n{'='*60}")
    print(f"Continuous Reading Test ({num_frames} frames)")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("[FAIL] Could not connect")
        return False

    success_count = 0
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            success_count += 1
            print(f"  Frame {i+1}: OK ({frame.shape})")
        else:
            print(f"  Frame {i+1}: FAILED")

    cap.release()

    print(f"\nRead {success_count}/{num_frames} frames")

    if success_count == num_frames:
        print("[PASS] All frames read successfully")
        return True
    elif success_count > 0:
        print("[WARN] Some frames failed")
        return True
    else:
        print("[FAIL] No frames could be read")
        return False


def show_setup_instructions():
    """Print setup instructions for Windows."""
    host_ip = get_wsl_host_ip()

    print(f"""
{'='*60}
WINDOWS SETUP INSTRUCTIONS
{'='*60}

Your Windows host IP (from WSL2): {host_ip}

Step 1: Install ffmpeg
  - Download from https://www.gyan.dev/ffmpeg/builds/
  - Extract to C:\\ffmpeg and add C:\\ffmpeg\\bin to PATH

Step 2: Install MediaMTX (RTSP server)
  - Download from https://github.com/bluenviron/mediamtx/releases
  - Extract mediamtx.exe to C:\\mediamtx

Step 3: Find your camera name (PowerShell)
  ffmpeg -list_devices true -f dshow -i dummy

Step 4: Start MediaMTX (PowerShell #1)
  cd C:\\mediamtx
  .\\mediamtx.exe

Step 5: Start webcam stream (PowerShell #2)
  ffmpeg -f dshow -i video="Integrated Camera" -framerate 30 ^
    -video_size 640x480 -vcodec libx264 -preset ultrafast ^
    -tune zerolatency -f rtsp rtsp://localhost:8554/webcam

  (Replace "Integrated Camera" with your camera name from Step 3)

Step 6: Test from WSL2
  python scripts/test_rtsp_connection.py

Step 7: Use in your app
  export WEBCAM_RTSP_URL="rtsp://{host_ip}:8554/webcam"
  python tests/test_webcam_capture.py
""")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            show_setup_instructions()
            return 0
        rtsp_url = sys.argv[1]
    else:
        # Auto-detect host IP
        host_ip = get_wsl_host_ip()
        rtsp_url = f"rtsp://{host_ip}:8554/webcam"

    print(f"Windows host IP detected: {get_wsl_host_ip()}")

    # Run tests
    if test_rtsp_connection(rtsp_url):
        test_rtsp_continuous(rtsp_url)
        print(f"\n{'='*60}")
        print("SUCCESS! RTSP streaming is working.")
        print(f"{'='*60}")
        print(f"\nTo use webcam in your app, set:")
        print(f'  export WEBCAM_RTSP_URL="{rtsp_url}"')
        return 0
    else:
        print(f"\n{'='*60}")
        print("RTSP connection failed.")
        print(f"{'='*60}")
        show_setup_instructions()
        return 1


if __name__ == "__main__":
    sys.exit(main())

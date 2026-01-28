"""Test webcam capture functionality.

This test verifies that the webcam can capture images and videos.

Supports two modes:
1. Direct webcam access (native Linux or Windows)
2. RTSP streaming from Windows to WSL2 (set WEBCAM_RTSP_URL env var)

Usage:
    # Direct webcam
    python tests/test_webcam_capture.py

    # RTSP from Windows (WSL2)
    export WEBCAM_RTSP_URL="rtsp://172.25.192.1:8554/webcam"
    python tests/test_webcam_capture.py
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.webcam import get_video_capture, get_capture_source_info, RTSP_URL


def test_webcam_available() -> bool:
    """Test if webcam/RTSP stream is accessible.

    Returns:
        True if video source opens successfully.
    """
    print(f"\n{'='*60}")
    print("Test 1: Video Source Availability")
    print(f"{'='*60}")

    source_info = get_capture_source_info()
    print(f"Capture source: {source_info}")

    cap = get_video_capture()
    available = cap.isOpened()

    if available:
        # Try to read a frame to verify connection
        ret, frame = cap.read()
        if ret:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Video source connected!")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}")
            print(f"  Frame shape: {frame.shape}")
            print(f"\n[PASS] Video source is available")
        else:
            print("Connected but failed to read frame.")
            print("For RTSP: Check if ffmpeg is streaming on Windows.")
            print(f"\n[FAIL] Could not read frame")
            available = False
    else:
        print("Could not connect to video source.")
        if RTSP_URL:
            print(f"RTSP URL: {RTSP_URL}")
            print("Check that:")
            print("  1. MediaMTX is running on Windows")
            print("  2. ffmpeg is streaming the webcam")
            print("  3. The IP address is correct")
        else:
            print("No webcam found or could not open.")
            print("For WSL2: Set WEBCAM_RTSP_URL environment variable")
        print(f"\n[FAIL] Video source not available")

    cap.release()
    return available


def test_capture_image() -> bool:
    """Test single image capture.

    Returns:
        True if image is captured and saved successfully.
    """
    print(f"\n{'='*60}")
    print("Test 2: Image Capture")
    print(f"{'='*60}")

    cap = get_video_capture()
    if not cap.isOpened():
        print(f"Cannot open {get_capture_source_info()}")
        print(f"\n[FAIL] Video source not available")
        return False

    # Allow camera to warm up
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture image")
        print(f"\n[FAIL] Image capture failed")
        return False

    # Save image
    temp_path = Path(tempfile.mkdtemp()) / "test_capture.jpg"
    cv2.imwrite(str(temp_path), frame)

    print(f"Image captured successfully!")
    print(f"  Shape: {frame.shape}")
    print(f"  Dtype: {frame.dtype}")
    print(f"  Saved to: {temp_path}")
    print(f"  File size: {temp_path.stat().st_size} bytes")

    if temp_path.exists() and temp_path.stat().st_size > 0:
        print(f"\n[PASS] Image capture successful")
        return True
    else:
        print(f"\n[FAIL] Image file not created properly")
        return False


def test_capture_video(duration: float = 2.0, fps: int = 15) -> bool:
    """Test video capture.

    Args:
        duration: Duration in seconds.
        fps: Target frames per second.

    Returns:
        True if video is captured and saved successfully.
    """
    print(f"\n{'='*60}")
    print("Test 3: Video Capture")
    print(f"{'='*60}")
    print(f"Duration: {duration}s, Target FPS: {fps}")

    cap = get_video_capture()
    if not cap.isOpened():
        print(f"Cannot open {get_capture_source_info()}")
        print(f"\n[FAIL] Video source not available")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_path = Path(tempfile.mkdtemp()) / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))

    if not out.isOpened():
        print("Failed to create video writer")
        cap.release()
        print(f"\n[FAIL] Video writer creation failed")
        return False

    import time
    start = time.time()
    frame_count = 0

    print(f"Recording...")
    while time.time() - start < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frame_count += 1
        else:
            break

    cap.release()
    out.release()

    actual_fps = frame_count / duration
    print(f"Video captured successfully!")
    print(f"  Frames captured: {frame_count}")
    print(f"  Actual FPS: {actual_fps:.1f}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Saved to: {temp_path}")
    print(f"  File size: {temp_path.stat().st_size} bytes")

    if temp_path.exists() and frame_count > 0:
        print(f"\n[PASS] Video capture successful")
        # Store path for frame extraction test
        print(f"\nVideo saved for frame extraction test: {temp_path}")
        return True
    else:
        print(f"\n[FAIL] Video capture failed")
        return False


def test_multiple_captures() -> bool:
    """Test multiple consecutive captures.

    Returns:
        True if all captures succeed.
    """
    print(f"\n{'='*60}")
    print("Test 4: Multiple Captures")
    print(f"{'='*60}")

    cap = get_video_capture()
    if not cap.isOpened():
        print(f"Cannot open {get_capture_source_info()}")
        print(f"\n[FAIL] Video source not available")
        return False

    success_count = 0
    total_captures = 5

    print(f"Capturing {total_captures} images...")
    for i in range(total_captures):
        ret, frame = cap.read()
        if ret:
            success_count += 1
            print(f"  Capture {i+1}: OK ({frame.shape})")
        else:
            print(f"  Capture {i+1}: FAILED")

    cap.release()

    print(f"\nSuccessful captures: {success_count}/{total_captures}")

    if success_count == total_captures:
        print(f"\n[PASS] All captures successful")
        return True
    else:
        print(f"\n[FAIL] Some captures failed")
        return False


def test_webcam_properties() -> bool:
    """Test reading webcam properties.

    Returns:
        True if properties can be read.
    """
    print(f"\n{'='*60}")
    print("Test 5: Video Source Properties")
    print(f"{'='*60}")

    cap = get_video_capture()
    if not cap.isOpened():
        print(f"Cannot open {get_capture_source_info()}")
        print(f"\n[FAIL] Video source not available")
        return False

    properties = {
        "Width": cv2.CAP_PROP_FRAME_WIDTH,
        "Height": cv2.CAP_PROP_FRAME_HEIGHT,
        "FPS": cv2.CAP_PROP_FPS,
        "Brightness": cv2.CAP_PROP_BRIGHTNESS,
        "Contrast": cv2.CAP_PROP_CONTRAST,
        "Saturation": cv2.CAP_PROP_SATURATION,
        "Exposure": cv2.CAP_PROP_EXPOSURE,
        "Auto Exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    }

    print("Webcam properties:")
    for name, prop_id in properties.items():
        value = cap.get(prop_id)
        print(f"  {name}: {value}")

    cap.release()
    print(f"\n[PASS] Properties read successfully")
    return True


def run_all_tests():
    """Run all video capture tests."""
    print("\n" + "="*60)
    print("VIDEO CAPTURE TESTS")
    print("="*60)
    print(f"Source: {get_capture_source_info()}")
    if RTSP_URL:
        print(f"Mode: RTSP streaming (WSL2)")
    else:
        print(f"Mode: Direct webcam access")

    results = {
        "webcam_available": test_webcam_available(),
        "image_capture": test_capture_image(),
        "video_capture": test_capture_video(),
        "multiple_captures": test_multiple_captures(),
        "properties": test_webcam_properties(),
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(results.values())
    total = len(results)
    print(f"Passed: {passed}/{total}")

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    if passed == total:
        print("\nAll tests passed! Webcam capture is ready.")
        return 0
    elif results["webcam_available"]:
        print("\nSome tests failed but webcam is available.")
        return 1
    else:
        print("\nWebcam not available. Cannot run capture tests.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

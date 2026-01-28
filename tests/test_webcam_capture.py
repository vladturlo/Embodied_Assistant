"""Test webcam capture functionality.

This test verifies that the webcam can capture images and videos.

Usage:
    python tests/test_webcam_capture.py
"""

import sys
import tempfile
from pathlib import Path

import cv2


def test_webcam_available() -> bool:
    """Test if webcam is accessible.

    Returns:
        True if webcam opens successfully.
    """
    print(f"\n{'='*60}")
    print("Test 1: Webcam Availability")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(0)
    available = cap.isOpened()

    if available:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Webcam found!")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"\n[PASS] Webcam is available")
    else:
        print("No webcam found or could not open.")
        print("Check if a camera is connected and not in use by another app.")
        print(f"\n[FAIL] Webcam not available")

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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        print(f"\n[FAIL] Webcam not available")
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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        print(f"\n[FAIL] Webcam not available")
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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        print(f"\n[FAIL] Webcam not available")
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
    print("Test 5: Webcam Properties")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        print(f"\n[FAIL] Webcam not available")
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
    """Run all webcam tests."""
    print("\n" + "="*60)
    print("WEBCAM CAPTURE TESTS")
    print("="*60)

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

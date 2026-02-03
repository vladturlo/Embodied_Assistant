"""Tests for the Live Vision capture service.

Run with:
    python tests/test_live_capture.py
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.live_capture import (
    LiveCaptureService,
    FrameBuffer,
    TimestampedFrame,
    frames_to_images,
)
from PIL import Image


def test_frame_buffer():
    """Test FrameBuffer circular buffer functionality."""
    print("\n=== Testing FrameBuffer ===")

    buffer = FrameBuffer(max_seconds=2.0, max_frames=10)

    # Create test images
    test_img = Image.new('RGB', (100, 100), color='red')

    # Add frames
    print("Adding 5 frames...")
    for i in range(5):
        buffer.add_frame(test_img)
        time.sleep(0.1)

    print(f"  Frame count: {buffer.frame_count}")
    assert buffer.frame_count == 5, "Expected 5 frames"

    # Get recent frames
    recent = buffer.get_recent_frames(seconds=1.0, max_count=10)
    print(f"  Frames in last 1 second: {len(recent)}")
    assert len(recent) == 5, "Expected 5 frames in recent"

    # Get latest frame
    latest = buffer.get_latest_frame()
    print(f"  Latest frame exists: {latest is not None}")
    assert latest is not None, "Expected latest frame"

    # Test frame expiration
    print("\nTesting frame expiration...")
    time.sleep(2.5)  # Wait for frames to expire
    buffer._cleanup_old_frames(time.time())
    recent = buffer.get_recent_frames(seconds=1.0, max_count=10)
    print(f"  Frames after expiration wait: {len(recent)}")
    assert len(recent) == 0, "Expected 0 frames after expiration"

    # Clear buffer
    buffer.add_frame(test_img)  # Add one frame
    buffer.clear()
    print(f"  Frame count after clear: {buffer.frame_count}")
    assert buffer.frame_count == 0, "Expected 0 frames after clear"

    print("FrameBuffer tests PASSED")


def test_live_capture_service_init():
    """Test LiveCaptureService initialization."""
    print("\n=== Testing LiveCaptureService Init ===")

    service = LiveCaptureService(
        capture_fps=2.0,
        buffer_seconds=3.0,
        max_buffer_frames=20,
        add_timestamp_overlay=True,
        camera_index=0
    )

    print(f"  is_running (before start): {service.is_running}")
    assert not service.is_running, "Should not be running before start"

    print(f"  inactivity_timeout: {service.inactivity_timeout}")
    assert service.inactivity_timeout == 5 * 60, "Default timeout should be 5 minutes"

    # Test setter
    service.inactivity_timeout = 60
    print(f"  inactivity_timeout (after set): {service.inactivity_timeout}")
    assert service.inactivity_timeout == 60, "Timeout should be 60 after set"

    print("LiveCaptureService init tests PASSED")


def test_live_capture_service_capture():
    """Test LiveCaptureService frame capture.

    NOTE: This test requires a connected webcam on Windows.
    """
    print("\n=== Testing LiveCaptureService Capture ===")

    service = LiveCaptureService(
        capture_fps=2.0,
        buffer_seconds=3.0,
        add_timestamp_overlay=True,
    )

    # Start capture
    print("Starting capture service...")
    success = service.start()

    if not success:
        print(f"  ERROR: Failed to start - {service.error_message}")
        print("  (This may be expected if no webcam is connected)")
        return False

    print(f"  Started successfully: {service.is_running}")
    assert service.is_running, "Should be running after start"

    # Wait for some frames
    print("Waiting for frames to capture...")
    time.sleep(2.0)

    # Check frame count
    frame_count = service.frame_count
    print(f"  Frames captured: {frame_count}")
    assert frame_count > 0, "Should have captured some frames"

    # Get recent frames
    recent = service.get_recent_frames(seconds=2.0, max_count=5)
    print(f"  Recent frames retrieved: {len(recent)}")
    assert len(recent) > 0, "Should have recent frames"

    # Check frame has timestamp
    frame = recent[0]
    print(f"  First frame timestamp: {frame.datetime_str}")
    assert frame.datetime_str, "Frame should have timestamp string"
    assert frame.image is not None, "Frame should have image"

    # Test snapshot
    print("Testing snapshot...")
    snapshot = service.take_snapshot()
    assert snapshot is not None, "Snapshot should return data"
    img, ts = snapshot
    print(f"  Snapshot timestamp: {ts}")
    print(f"  Snapshot size: {img.size}")

    # Stop capture
    print("Stopping capture service...")
    service.stop()
    print(f"  is_running (after stop): {service.is_running}")
    assert not service.is_running, "Should not be running after stop"

    print("LiveCaptureService capture tests PASSED")
    return True


def test_frames_to_images():
    """Test frames_to_images utility function."""
    print("\n=== Testing frames_to_images ===")

    # Create test timestamped frames
    test_img = Image.new('RGB', (100, 100), color='blue')
    frames = [
        TimestampedFrame(image=test_img, timestamp=time.time(), datetime_str="12:00:00.000"),
        TimestampedFrame(image=test_img, timestamp=time.time(), datetime_str="12:00:01.000"),
    ]

    images = frames_to_images(frames)
    print(f"  Input frames: {len(frames)}")
    print(f"  Output images: {len(images)}")
    assert len(images) == 2, "Should have 2 images"
    assert all(isinstance(img, Image.Image) for img in images), "All should be PIL Images"

    print("frames_to_images tests PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Live Capture Service Tests")
    print("=" * 50)

    test_frame_buffer()
    test_live_capture_service_init()
    test_frames_to_images()

    # Camera-dependent test (may fail without webcam)
    webcam_ok = test_live_capture_service_capture()

    print("\n" + "=" * 50)
    if webcam_ok:
        print("All tests PASSED!")
    else:
        print("Tests completed (webcam test skipped - no camera)")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()

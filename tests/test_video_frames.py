"""Test video frame extraction for vision model.

This test verifies that frames can be extracted from video files
for analysis by the vision model.

Usage:
    python tests/test_video_frames.py
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage


def create_test_video(path: str, duration: float = 2.0, fps: int = 10) -> bool:
    """Create a synthetic test video with changing colors.

    Args:
        path: Output video path.
        duration: Duration in seconds.
        fps: Frames per second.

    Returns:
        True if video is created successfully.
    """
    width, height = 320, 240
    total_frames = int(duration * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    if not out.isOpened():
        return False

    # Create frames with changing colors
    for i in range(total_frames):
        # Cycle through colors
        t = i / total_frames
        r = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * t)))
        g = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * t + 2)))
        b = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * t + 4)))

        frame = np.full((height, width, 3), [b, g, r], dtype=np.uint8)

        # Add frame number text
        cv2.putText(frame, f"Frame {i+1}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    return Path(path).exists()


def extract_frames(video_path: str, num_frames: int = 5) -> list:
    """Extract evenly-spaced frames from video.

    Args:
        video_path: Path to video file.
        num_frames: Number of frames to extract.

    Returns:
        List of PIL Image objects.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    # Calculate frame indices
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb_frame)
            frames.append(pil_img)

    cap.release()
    return frames


def test_create_test_video() -> bool:
    """Test creating a synthetic test video.

    Returns:
        True if video is created successfully.
    """
    print(f"\n{'='*60}")
    print("Test 1: Create Test Video")
    print(f"{'='*60}")

    temp_path = Path(tempfile.mkdtemp()) / "test_video.mp4"
    print(f"Creating test video: {temp_path}")

    success = create_test_video(str(temp_path))

    if success:
        file_size = temp_path.stat().st_size
        print(f"Video created successfully!")
        print(f"  Path: {temp_path}")
        print(f"  Size: {file_size} bytes")
        print(f"\n[PASS] Test video created")

        # Store path for other tests
        global TEST_VIDEO_PATH
        TEST_VIDEO_PATH = str(temp_path)
        return True
    else:
        print(f"\n[FAIL] Could not create test video")
        return False


def test_extract_frames() -> bool:
    """Test extracting frames from video.

    Returns:
        True if frames are extracted successfully.
    """
    print(f"\n{'='*60}")
    print("Test 2: Extract Frames")
    print(f"{'='*60}")

    # Use test video from previous test or create new one
    video_path = globals().get('TEST_VIDEO_PATH')
    if not video_path or not Path(video_path).exists():
        video_path = Path(tempfile.mkdtemp()) / "test_video.mp4"
        create_test_video(str(video_path))

    print(f"Extracting frames from: {video_path}")

    frames = extract_frames(str(video_path), num_frames=5)

    if len(frames) == 5:
        print(f"Extracted {len(frames)} frames:")
        for i, frame in enumerate(frames):
            print(f"  Frame {i+1}: {frame.size}, mode={frame.mode}")
        print(f"\n[PASS] Frame extraction successful")
        return True
    else:
        print(f"Expected 5 frames, got {len(frames)}")
        print(f"\n[FAIL] Frame extraction failed")
        return False


def test_frame_to_agimage() -> bool:
    """Test converting frames to AGImage objects.

    Returns:
        True if conversion is successful.
    """
    print(f"\n{'='*60}")
    print("Test 3: Convert to AGImage")
    print(f"{'='*60}")

    try:
        from autogen_core import Image as AGImage

        # Create a simple test frame
        pil_img = PILImage.new('RGB', (100, 100), color='blue')
        ag_image = AGImage(pil_img)

        print(f"Converted PIL Image to AGImage:")
        print(f"  Type: {type(ag_image)}")
        print(f"  Base64 length: {len(ag_image.to_base64())}")

        print(f"\n[PASS] Conversion successful")
        return True

    except ImportError as e:
        print(f"\n[FAIL] Import error: {e}")
        print("Make sure autogen-core is installed")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


def test_extract_and_convert() -> bool:
    """Test extracting frames and converting to AGImage.

    Returns:
        True if the full pipeline works.
    """
    print(f"\n{'='*60}")
    print("Test 4: Extract and Convert Pipeline")
    print(f"{'='*60}")

    try:
        from autogen_core import Image as AGImage

        # Create test video
        video_path = Path(tempfile.mkdtemp()) / "pipeline_test.mp4"
        create_test_video(str(video_path))

        print(f"Extracting and converting frames from: {video_path}")

        # Extract frames
        frames = extract_frames(str(video_path), num_frames=3)

        if len(frames) == 0:
            print(f"\n[FAIL] No frames extracted")
            return False

        # Convert to AGImage
        ag_images = []
        for i, frame in enumerate(frames):
            ag_image = AGImage(frame)
            ag_images.append(ag_image)
            print(f"  Frame {i+1}: PIL {frame.size} -> AGImage ({len(ag_image.to_base64())} bytes)")

        if len(ag_images) == 3:
            print(f"\n[PASS] Pipeline successful")
            return True
        else:
            print(f"\n[FAIL] Not all frames converted")
            return False

    except ImportError as e:
        print(f"\n[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_variable_frame_count() -> bool:
    """Test extracting different numbers of frames.

    Returns:
        True if various frame counts work correctly.
    """
    print(f"\n{'='*60}")
    print("Test 5: Variable Frame Count")
    print(f"{'='*60}")

    # Create test video
    video_path = Path(tempfile.mkdtemp()) / "variable_test.mp4"
    create_test_video(str(video_path), duration=3.0, fps=15)

    test_counts = [1, 3, 5, 10]
    results = []

    for count in test_counts:
        frames = extract_frames(str(video_path), num_frames=count)
        success = len(frames) == count
        results.append(success)
        status = "OK" if success else "FAIL"
        print(f"  {count} frames: {status} (got {len(frames)})")

    if all(results):
        print(f"\n[PASS] All frame counts extracted correctly")
        return True
    else:
        print(f"\n[FAIL] Some frame counts failed")
        return False


def test_video_metadata() -> bool:
    """Test reading video metadata.

    Returns:
        True if metadata is read successfully.
    """
    print(f"\n{'='*60}")
    print("Test 6: Video Metadata")
    print(f"{'='*60}")

    # Create test video
    video_path = Path(tempfile.mkdtemp()) / "metadata_test.mp4"
    create_test_video(str(video_path), duration=2.0, fps=10)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"\n[FAIL] Could not open video")
        return False

    metadata = {
        "Frame Count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "FPS": cap.get(cv2.CAP_PROP_FPS),
        "Width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "Height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "Duration (s)": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
    }

    cap.release()

    print("Video metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    print(f"\n[PASS] Metadata read successfully")
    return True


def run_all_tests():
    """Run all video frame extraction tests."""
    print("\n" + "="*60)
    print("VIDEO FRAME EXTRACTION TESTS")
    print("="*60)

    results = {
        "create_test_video": test_create_test_video(),
        "extract_frames": test_extract_frames(),
        "frame_to_agimage": test_frame_to_agimage(),
        "extract_and_convert": test_extract_and_convert(),
        "variable_frame_count": test_variable_frame_count(),
        "video_metadata": test_video_metadata(),
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
        print("\nAll tests passed! Video frame extraction is ready.")
        return 0
    else:
        print("\nSome tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

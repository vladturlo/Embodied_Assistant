"""Webcam capture tool for the multimodal agent.

This module provides functionality to capture images and short videos
from the webcam, display them in the Chainlit chat, and return file paths
for the vision model to analyze.

Supports two modes:
1. Direct webcam access (native Linux or Windows)
2. RTSP streaming from Windows to WSL2 (set WEBCAM_RTSP_URL env var)
"""

import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from PIL import Image as PILImage


def center_crop_square(frame: np.ndarray) -> np.ndarray:
    """Center-crop frame to the largest centered square.

    This removes the left/right sides (for landscape) or top/bottom (for portrait)
    to create a square image, which is optimal for vision models like Ministral-3.

    Args:
        frame: Input image as numpy array (H, W, C).

    Returns:
        Square-cropped image as numpy array.
    """
    h, w = frame.shape[:2]
    size = min(h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    return frame[top:top + size, left:left + size]


# Optional: Import chainlit for display in chat
try:
    import chainlit as cl
    CHAINLIT_AVAILABLE = True
except ImportError:
    CHAINLIT_AVAILABLE = False

def _get_ffmpeg_path() -> str:
    """Find ffmpeg executable cross-platform.

    Returns:
        Path to ffmpeg executable.
    """
    # Check if ffmpeg is in PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path

    # Windows common install locations
    if platform.system() == "Windows":
        common_paths = [
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            os.path.expandvars(r'%LOCALAPPDATA%\ffmpeg\bin\ffmpeg.exe'),
        ]
        for path in common_paths:
            if Path(path).exists():
                return path

    # Fallback - assume it's in PATH
    return 'ffmpeg'


# RTSP URL for WSL2 mode (set via environment variable)
# Example: export WEBCAM_RTSP_URL="rtsp://172.25.192.1:8554/webcam"
RTSP_URL = os.environ.get("WEBCAM_RTSP_URL", None)


def get_video_capture() -> cv2.VideoCapture:
    """Get video capture from webcam or RTSP stream.

    If WEBCAM_RTSP_URL environment variable is set, uses RTSP stream
    (for WSL2 streaming from Windows). Otherwise, uses direct webcam access.

    Returns:
        OpenCV VideoCapture object.
    """
    if RTSP_URL:
        # WSL2 mode: use RTSP stream from Windows
        return cv2.VideoCapture(RTSP_URL)
    else:
        # Native mode: direct webcam access
        return cv2.VideoCapture(0)


def get_capture_source_info() -> str:
    """Get information about the current capture source.

    Returns:
        String describing the capture source (RTSP URL or 'device 0').
    """
    if RTSP_URL:
        return f"RTSP stream ({RTSP_URL})"
    else:
        return "local webcam (device 0)"


async def capture_webcam(
    mode: str = "image",
    duration: float = 3.0
) -> str:
    """Capture an image or short video from the webcam.

    This tool captures media from the default webcam (device 0) and saves it
    to a temporary file. The captured media is also displayed in the Chainlit
    chat so both the user and agent can see what was captured.

    Args:
        mode: Either "image" for a single frame capture, or "video" for a
            short video clip. Defaults to "image".
        duration: Duration in seconds for video capture. Only used when
            mode is "video". Must be between 1 and 10 seconds.
            Defaults to 3.0 seconds.

    Returns:
        Path to the captured file (JPEG for images, MP4 for videos), or an
        error message if capture failed.
    """
    # Validate duration
    duration = max(1.0, min(10.0, duration))

    # Open video capture (webcam or RTSP stream)
    cap = get_video_capture()
    source_info = get_capture_source_info()

    if not cap.isOpened():
        return f"Error: Could not access {source_info}. Please check connection."

    try:
        if mode == "video":
            return await _capture_video(cap, duration)
        else:
            return await _capture_image(cap)
    finally:
        cap.release()


async def _capture_image(cap: cv2.VideoCapture) -> str:
    """Capture a single image from the webcam.

    Args:
        cap: OpenCV VideoCapture object.

    Returns:
        Path to the captured image file or error message.
    """
    # Read frame
    ret, frame = cap.read()
    if not ret:
        return "Error: Failed to capture image from webcam."

    # Create temp file
    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / f"webcam_capture_{int(time.time())}.jpg"

    # Save image
    cv2.imwrite(str(temp_path), frame)

    # Note: Image display is handled in app.py after tool execution
    return str(temp_path)


async def _capture_video(cap: cv2.VideoCapture, duration: float) -> str:
    """Capture a short video from the webcam.

    Args:
        cap: OpenCV VideoCapture object.
        duration: Duration in seconds.

    Returns:
        Path to the captured video file or error message.
    """
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Get actual FPS from capture device, fallback to 30 if not available
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 60:
        fps = 30  # Default for most webcams/RTSP streams

    # Create temp file
    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / f"webcam_video_{int(time.time())}.mp4"

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))

    if not out.isOpened():
        return "Error: Failed to initialize video writer."

    # Capture frames
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frame_count += 1
        else:
            break

    out.release()

    if frame_count == 0:
        return "Error: No frames captured."

    # Re-encode to H.264 for browser compatibility (mp4v codec not supported by browsers)
    h264_path = temp_dir / f"webcam_video_{int(time.time())}_h264.mp4"
    result = subprocess.run([
        _get_ffmpeg_path(), '-y', '-i', str(temp_path),
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
        str(h264_path)
    ], capture_output=True)

    if result.returncode == 0 and h264_path.exists():
        temp_path.unlink()  # Remove original non-browser-compatible file
        temp_path = h264_path

    # Display in Chainlit chat if available
    if CHAINLIT_AVAILABLE:
        try:
            video_element = cl.Video(
                path=str(temp_path),
                name="webcam_video"
            )
            await cl.Message(
                content=f"🎥 Captured {duration:.1f}s video from webcam ({frame_count} frames):",
                elements=[video_element]
            ).send()
        except Exception as e:
            # If Chainlit display fails, continue anyway
            pass

    return str(temp_path)


def extract_video_frames(
    video_path: str,
    frames_per_second: float = 5.0,
    max_frames: int = 50
) -> list:
    """Extract frames from a video file at a specified rate.

    This is useful for analyzing video content with vision models that
    process images rather than video directly.

    Args:
        video_path: Path to the video file.
        frames_per_second: How many frames to extract per second of video.
            Defaults to 5.0 (e.g., a 5-second video yields 25 frames).
        max_frames: Maximum number of frames to extract. Prevents excessive
            frame extraction for long videos. Defaults to 50.

    Returns:
        List of PIL Image objects representing the extracted frames.
        Returns empty list if extraction fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames == 0 or video_fps <= 0:
        cap.release()
        return []

    # Calculate video duration and number of frames to extract
    duration = total_frames / video_fps
    num_frames = int(duration * frames_per_second)

    # Apply limits
    num_frames = max(1, min(num_frames, max_frames))

    # Calculate frame indices (evenly spaced)
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb_frame)
            frames.append(pil_img)

    cap.release()
    return frames


def test_webcam_availability() -> bool:
    """Test if a webcam/RTSP stream is available.

    Returns:
        True if video source is accessible, False otherwise.
    """
    cap = get_video_capture()
    available = cap.isOpened()
    if available:
        # Try to read a frame to verify connection
        ret, _ = cap.read()
        available = ret
    cap.release()
    return available


# Synchronous versions for testing without async
def capture_frame_bytes(
    max_size: int = 16,
    jpeg_quality: int = 70
) -> Optional[bytes]:
    """Capture a single frame, center-crop to square, resize, and return as JPEG bytes.

    This is used by the embodied control loop to get image data
    that can be sent directly to the model as AGImage.

    Args:
        max_size: Output image size (square, default 240x240 for optimal Ministral-3).
        jpeg_quality: JPEG compression quality 0-100 (default 70).

    Returns:
        JPEG image bytes or None if capture failed.
    """
    cap = get_video_capture()

    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    # Center-crop to square (optimal for 1:1 aspect ratio models like Ministral-3)
    frame = center_crop_square(frame)

    # Resize to target size
    frame = cv2.resize(frame, (max_size, max_size), interpolation=cv2.INTER_AREA)

    # Flip horizontally to correct mirror effect (selfie-style webcams)
    frame = cv2.flip(frame, 1)

    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return buffer.tobytes()


def capture_frame_from_cap(
    cap: cv2.VideoCapture,
    max_size: int = 16,
    jpeg_quality: int = 70
) -> Optional[bytes]:
    """Capture frame from an existing VideoCapture (no reconnection overhead).

    This is optimized for the embodied control loop where we want to avoid
    the 3+ second RTSP connection overhead on every frame.

    Args:
        cap: Open VideoCapture object (must already be connected).
        max_size: Output image size (square, default 240x240 for optimal Ministral-3).
        jpeg_quality: JPEG compression quality 0-100 (default 70).

    Returns:
        JPEG image bytes or None if read failed.
    """
    ret, frame = cap.read()
    if not ret:
        return None

    # Center-crop to square (optimal for 1:1 aspect ratio models like Ministral-3)
    frame = center_crop_square(frame)

    # Resize to target size
    frame = cv2.resize(frame, (max_size, max_size), interpolation=cv2.INTER_AREA)

    # Flip horizontally to correct mirror effect (selfie-style webcams)
    frame = cv2.flip(frame, 1)

    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return buffer.tobytes()


def capture_image_sync() -> str:
    """Synchronous version of image capture for testing.

    Returns:
        Path to the captured image file or error message.
    """
    cap = get_video_capture()
    source_info = get_capture_source_info()

    if not cap.isOpened():
        return f"Error: Could not access {source_info}."

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Error: Failed to capture image."

    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / f"test_capture_{int(time.time())}.jpg"
    cv2.imwrite(str(temp_path), frame)

    return str(temp_path)


def capture_video_sync(duration: float = 2.0) -> str:
    """Synchronous version of video capture for testing.

    Args:
        duration: Duration in seconds.

    Returns:
        Path to the captured video file or error message.
    """
    cap = get_video_capture()
    source_info = get_capture_source_info()

    if not cap.isOpened():
        return f"Error: Could not access {source_info}."

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 15

    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / f"test_video_{int(time.time())}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frame_count += 1

    cap.release()
    out.release()

    if frame_count == 0:
        return "Error: No frames captured."

    return str(temp_path)

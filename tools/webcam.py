"""Webcam capture tool for the multimodal agent.

This module provides functionality to capture images and short videos
from the webcam, display them in the Chainlit chat, and return file paths
for the vision model to analyze.

Supports two modes:
1. Direct webcam access (native Linux or Windows)
2. RTSP streaming from Windows to WSL2 (set WEBCAM_RTSP_URL env var)
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from PIL import Image as PILImage

# Optional: Import chainlit for display in chat
try:
    import chainlit as cl
    CHAINLIT_AVAILABLE = True
except ImportError:
    CHAINLIT_AVAILABLE = False

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
    fps = 15  # Target FPS for recording

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
    num_frames: int = 5
) -> list:
    """Extract evenly-spaced frames from a video file.

    This is useful for analyzing video content with vision models that
    process images rather than video directly.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract. Defaults to 5.

    Returns:
        List of PIL Image objects representing the extracted frames.
        Returns empty list if extraction fails.
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

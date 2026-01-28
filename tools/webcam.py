"""Webcam capture tool for the multimodal agent.

This module provides functionality to capture images and short videos
from the webcam, display them in the Chainlit chat, and return file paths
for the vision model to analyze.
"""

import tempfile
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image as PILImage

# Optional: Import chainlit for display in chat
try:
    import chainlit as cl
    CHAINLIT_AVAILABLE = True
except ImportError:
    CHAINLIT_AVAILABLE = False


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

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Could not access webcam. Please check if a camera is connected."

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

    # Display in Chainlit chat if available
    if CHAINLIT_AVAILABLE:
        try:
            image_element = cl.Image(path=str(temp_path), name="webcam_capture")
            await cl.Message(
                content="Captured image from webcam:",
                elements=[image_element]
            ).send()
        except Exception as e:
            # If Chainlit display fails, continue anyway
            pass

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
            video_element = cl.Video(path=str(temp_path), name="webcam_video")
            await cl.Message(
                content=f"Captured {duration:.1f}s video from webcam ({frame_count} frames):",
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
    """Test if a webcam is available.

    Returns:
        True if webcam is accessible, False otherwise.
    """
    cap = cv2.VideoCapture(0)
    available = cap.isOpened()
    cap.release()
    return available


# Synchronous versions for testing without async
def capture_image_sync() -> str:
    """Synchronous version of image capture for testing.

    Returns:
        Path to the captured image file or error message.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Could not access webcam."

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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Could not access webcam."

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

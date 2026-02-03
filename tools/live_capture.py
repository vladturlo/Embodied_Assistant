"""Live camera capture service for continuous vision mode.

This module provides continuous frame capture from webcam with a circular buffer,
enabling "live vision" mode where the model can see current camera state without
explicit tool calls.

Windows-only implementation using direct webcam access.
"""

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont


@dataclass
class TimestampedFrame:
    """A frame with its capture timestamp."""
    image: PILImage.Image
    timestamp: float  # Unix timestamp
    datetime_str: str  # Human-readable timestamp


class FrameBuffer:
    """Thread-safe circular buffer for storing recent frames.

    Stores frames with timestamps for retrieval of recent N seconds of footage.
    """

    def __init__(self, max_seconds: float = 10.0, max_frames: int = 100):
        """Initialize the frame buffer.

        Args:
            max_seconds: Maximum age of frames to keep (in seconds).
            max_frames: Maximum number of frames to store (hard limit).
        """
        self._buffer: deque[TimestampedFrame] = deque(maxlen=max_frames)
        self._max_seconds = max_seconds
        self._lock = threading.Lock()

    def add_frame(self, frame: PILImage.Image) -> None:
        """Add a new frame to the buffer.

        Args:
            frame: PIL Image to store.
        """
        now = time.time()
        dt = datetime.fromtimestamp(now)
        datetime_str = dt.strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm

        timestamped = TimestampedFrame(
            image=frame,
            timestamp=now,
            datetime_str=datetime_str
        )

        with self._lock:
            self._buffer.append(timestamped)
            self._cleanup_old_frames(now)

    def _cleanup_old_frames(self, current_time: float) -> None:
        """Remove frames older than max_seconds.

        Must be called with lock held.
        """
        cutoff = current_time - self._max_seconds
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def get_recent_frames(self, seconds: float = 3.0, max_count: int = 10) -> List[TimestampedFrame]:
        """Get frames from the last N seconds.

        Args:
            seconds: How many seconds of recent frames to retrieve.
            max_count: Maximum number of frames to return.

        Returns:
            List of TimestampedFrame objects, oldest first.
        """
        now = time.time()
        cutoff = now - seconds

        with self._lock:
            self._cleanup_old_frames(now)
            recent = [f for f in self._buffer if f.timestamp >= cutoff]

        # If more frames than requested, sample evenly
        if len(recent) > max_count:
            indices = [int(i * len(recent) / max_count) for i in range(max_count)]
            recent = [recent[i] for i in indices]

        return recent

    def get_latest_frame(self) -> Optional[TimestampedFrame]:
        """Get the most recent frame.

        Returns:
            The latest TimestampedFrame, or None if buffer is empty.
        """
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
        return None

    def clear(self) -> None:
        """Clear all frames from the buffer."""
        with self._lock:
            self._buffer.clear()

    @property
    def frame_count(self) -> int:
        """Number of frames currently in buffer."""
        with self._lock:
            return len(self._buffer)


class LiveCaptureService:
    """Background service for continuous webcam frame capture.

    Captures frames at a specified rate and stores them in a circular buffer.
    Runs in a background thread to avoid blocking the main async loop.
    """

    def __init__(
        self,
        capture_fps: float = 2.0,
        buffer_seconds: float = 5.0,
        max_buffer_frames: int = 50,
        add_timestamp_overlay: bool = True,
        camera_index: int = 0
    ):
        """Initialize the capture service.

        Args:
            capture_fps: Target frames per second to capture.
            buffer_seconds: How many seconds of frames to keep.
            max_buffer_frames: Maximum frames in buffer.
            add_timestamp_overlay: Whether to add timestamp text to frames.
            camera_index: OpenCV camera device index.
        """
        self._capture_fps = capture_fps
        self._add_timestamp = add_timestamp_overlay
        self._camera_index = camera_index

        self._buffer = FrameBuffer(
            max_seconds=buffer_seconds,
            max_frames=max_buffer_frames
        )

        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_activity_time = time.time()
        self._error_message: Optional[str] = None

        # Inactivity timeout (auto-stop after N minutes of no requests)
        self._inactivity_timeout = 5 * 60  # 5 minutes default

    def _add_timestamp_overlay(self, frame: np.ndarray, timestamp_str: str) -> np.ndarray:
        """Add timestamp overlay to a frame.

        Args:
            frame: OpenCV BGR frame.
            timestamp_str: Timestamp text to overlay.

        Returns:
            Frame with timestamp overlay.
        """
        # Create a copy to avoid modifying original
        frame = frame.copy()

        # Add semi-transparent background for text
        height, width = frame.shape[:2]
        overlay = frame.copy()

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text = timestamp_str

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Position: bottom-left corner with padding
        padding = 10
        x = padding
        y = height - padding

        # Draw background rectangle
        bg_rect_start = (x - 5, y - text_height - 5)
        bg_rect_end = (x + text_width + 5, y + baseline + 5)
        cv2.rectangle(overlay, bg_rect_start, bg_rect_end, (0, 0, 0), -1)

        # Blend overlay with original
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

        return frame

    def _capture_loop(self) -> None:
        """Background thread capture loop."""
        frame_interval = 1.0 / self._capture_fps

        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            self._error_message = f"Failed to open camera {self._camera_index}"
            self._running = False
            return

        self._error_message = None

        while self._running:
            loop_start = time.time()

            # Check inactivity timeout
            if time.time() - self._last_activity_time > self._inactivity_timeout:
                self._error_message = "Auto-stopped due to inactivity"
                break

            # Capture frame
            ret, frame = self._cap.read()
            if not ret:
                # Try to reconnect
                self._cap.release()
                time.sleep(1.0)
                self._cap = cv2.VideoCapture(self._camera_index)
                continue

            # Get timestamp
            now = time.time()
            dt = datetime.fromtimestamp(now)
            timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # Add timestamp overlay if enabled
            if self._add_timestamp:
                frame = self._add_timestamp_overlay(frame, timestamp_str)

            # Convert BGR to RGB and create PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_frame)

            # Add to buffer
            self._buffer.add_frame(pil_image)

            # Sleep to maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Cleanup
        if self._cap:
            self._cap.release()
            self._cap = None

    def start(self) -> bool:
        """Start the capture service.

        Returns:
            True if started successfully, False if already running or error.
        """
        if self._running:
            return True

        self._running = True
        self._last_activity_time = time.time()
        self._error_message = None
        self._buffer.clear()

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        # Wait briefly to check if capture started OK
        time.sleep(0.5)

        if not self._running:
            return False

        return True

    def stop(self) -> None:
        """Stop the capture service."""
        self._running = False

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        self._buffer.clear()

    def get_recent_frames(self, seconds: float = 3.0, max_count: int = 10) -> List[TimestampedFrame]:
        """Get recent frames from the buffer.

        Also updates the activity timestamp to prevent auto-stop.

        Args:
            seconds: How many seconds of recent frames.
            max_count: Maximum number of frames to return.

        Returns:
            List of TimestampedFrame objects.
        """
        self._last_activity_time = time.time()
        return self._buffer.get_recent_frames(seconds, max_count)

    def get_latest_frame(self) -> Optional[TimestampedFrame]:
        """Get the most recent frame.

        Also updates the activity timestamp.

        Returns:
            Latest TimestampedFrame or None.
        """
        self._last_activity_time = time.time()
        return self._buffer.get_latest_frame()

    def take_snapshot(self) -> Optional[Tuple[PILImage.Image, str]]:
        """Take a snapshot (current frame) for saving.

        Returns:
            Tuple of (PIL Image, timestamp string) or None if no frame available.
        """
        frame = self.get_latest_frame()
        if frame:
            return (frame.image, frame.datetime_str)
        return None

    @property
    def is_running(self) -> bool:
        """Check if capture service is currently running."""
        return self._running

    @property
    def error_message(self) -> Optional[str]:
        """Get the last error message, if any."""
        return self._error_message

    @property
    def frame_count(self) -> int:
        """Number of frames currently in buffer."""
        return self._buffer.frame_count

    @property
    def inactivity_timeout(self) -> float:
        """Get inactivity timeout in seconds."""
        return self._inactivity_timeout

    @inactivity_timeout.setter
    def inactivity_timeout(self, seconds: float) -> None:
        """Set inactivity timeout in seconds."""
        self._inactivity_timeout = seconds

    def reset_activity(self) -> None:
        """Reset the activity timer to prevent auto-stop."""
        self._last_activity_time = time.time()


# Convenience functions for getting PIL images without timestamps
def frames_to_images(frames: List[TimestampedFrame]) -> List[PILImage.Image]:
    """Extract just the PIL Images from a list of TimestampedFrames.

    Args:
        frames: List of TimestampedFrame objects.

    Returns:
        List of PIL Image objects.
    """
    return [f.image for f in frames]

"""Tools package for the multimodal agent."""

from .webcam import (
    capture_webcam,
    extract_video_frames,
    get_video_capture,
    get_capture_source_info,
    test_webcam_availability,
    RTSP_URL,
)

from .live_capture import (
    LiveCaptureService,
    FrameBuffer,
    TimestampedFrame,
    frames_to_images,
)

__all__ = [
    "capture_webcam",
    "extract_video_frames",
    "get_video_capture",
    "get_capture_source_info",
    "test_webcam_availability",
    "RTSP_URL",
    "LiveCaptureService",
    "FrameBuffer",
    "TimestampedFrame",
    "frames_to_images",
]

"""Tools package for the multimodal agent."""

from .webcam import (
    capture_webcam,
    extract_video_frames,
    get_video_capture,
    get_capture_source_info,
    test_webcam_availability,
    RTSP_URL,
)

from .mouse import (
    move_mouse,
    get_mouse_position,
    get_screen_size,
)

__all__ = [
    "capture_webcam",
    "extract_video_frames",
    "get_video_capture",
    "get_capture_source_info",
    "test_webcam_availability",
    "RTSP_URL",
    "move_mouse",
    "get_mouse_position",
    "get_screen_size",
]

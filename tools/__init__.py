"""Tools package for the multimodal agent."""

from .webcam import (
    capture_webcam,
    capture_frame_bytes,
    capture_frame_from_cap,
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

from .profiler import (
    EmbodiedProfiler,
    IterationTimings,
)

__all__ = [
    "capture_webcam",
    "capture_frame_bytes",
    "capture_frame_from_cap",
    "extract_video_frames",
    "get_video_capture",
    "get_capture_source_info",
    "test_webcam_availability",
    "RTSP_URL",
    "move_mouse",
    "get_mouse_position",
    "get_screen_size",
    "EmbodiedProfiler",
    "IterationTimings",
]

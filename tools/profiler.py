"""Latency profiler for embodied control mode.

Measures timing at each stage of the embodied control loop to identify bottlenecks.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class IterationTimings:
    """Timings for a single embodied loop iteration."""

    iteration: int
    capture_ms: float = 0.0
    file_write_ms: float = 0.0
    ui_display_ms: float = 0.0
    msg_build_ms: float = 0.0
    inference_ttft_ms: float = 0.0  # Time to first token
    inference_tool_ms: float = 0.0  # Tool request → execution
    inference_total_ms: float = 0.0
    stop_check_ms: float = 0.0
    delay_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class StageSummary:
    """Summary statistics for a timing stage."""

    avg_ms: float
    min_ms: float
    max_ms: float
    pct: float  # Percentage of total time


class EmbodiedProfiler:
    """Profiles latency in the embodied control loop."""

    # Mapping of mark pairs to timing field names
    STAGE_MAPPINGS = [
        ("capture_start", "capture_end", "capture_ms"),
        ("file_write_start", "file_write_end", "file_write_ms"),
        ("ui_display_start", "ui_display_end", "ui_display_ms"),
        ("msg_build_start", "msg_build_end", "msg_build_ms"),
        ("inference_start", "inference_ttft", "inference_ttft_ms"),
        ("tool_request", "tool_executed", "inference_tool_ms"),
        ("inference_start", "inference_end", "inference_total_ms"),
        ("stop_check_start", "stop_check_end", "stop_check_ms"),
        ("delay_start", "delay_end", "delay_ms"),
        ("iteration_start", "iteration_end", "total_ms"),
    ]

    def __init__(self, model: str = "", image_settings: Optional[dict] = None):
        """Initialize the profiler.

        Args:
            model: Model name for logging.
            image_settings: Image configuration for logging.
        """
        self.iterations: list[IterationTimings] = []
        self._marks: dict[str, float] = {}
        self._current_iteration: int = 0
        self._session_start: datetime = datetime.now()
        self._model = model
        self._image_settings = image_settings or {}

    def start_iteration(self, iteration: int) -> None:
        """Begin timing a new iteration.

        Args:
            iteration: The iteration number (0-indexed).
        """
        self._current_iteration = iteration
        self._marks = {"iteration_start": time.perf_counter()}

    def mark(self, name: str) -> None:
        """Record a timestamp for a stage.

        Args:
            name: The name of the timing mark (e.g., "capture_start", "capture_end").
        """
        self._marks[name] = time.perf_counter()

    def _duration_ms(self, start_mark: str, end_mark: str) -> float:
        """Calculate duration in milliseconds between two marks.

        Args:
            start_mark: The starting mark name.
            end_mark: The ending mark name.

        Returns:
            Duration in milliseconds, or 0.0 if marks not found.
        """
        start = self._marks.get(start_mark)
        end = self._marks.get(end_mark)
        if start is not None and end is not None:
            return (end - start) * 1000
        return 0.0

    def end_iteration(self) -> IterationTimings:
        """Complete timing for current iteration and calculate durations.

        Returns:
            IterationTimings with all calculated durations.
        """
        self.mark("iteration_end")

        timings = IterationTimings(iteration=self._current_iteration)

        # Calculate all stage durations
        for start_mark, end_mark, field_name in self.STAGE_MAPPINGS:
            duration = self._duration_ms(start_mark, end_mark)
            setattr(timings, field_name, round(duration, 1))

        self.iterations.append(timings)
        return timings

    def get_summary(self) -> dict[str, StageSummary]:
        """Calculate summary statistics for each timing stage.

        Returns:
            Dictionary mapping stage names to their StageSummary.
        """
        if not self.iterations:
            return {}

        # Fields to summarize (excluding iteration number)
        fields = [
            "capture_ms",
            "file_write_ms",
            "ui_display_ms",
            "msg_build_ms",
            "inference_ttft_ms",
            "inference_tool_ms",
            "inference_total_ms",
            "stop_check_ms",
            "delay_ms",
            "total_ms",
        ]

        # Calculate total average for percentage calculation
        total_avg = sum(t.total_ms for t in self.iterations) / len(self.iterations)

        summary = {}
        for field_name in fields:
            values = [getattr(t, field_name) for t in self.iterations]
            avg = sum(values) / len(values)
            summary[field_name] = StageSummary(
                avg_ms=round(avg, 1),
                min_ms=round(min(values), 1),
                max_ms=round(max(values), 1),
                pct=round((avg / total_avg * 100) if total_avg > 0 else 0, 1),
            )

        return summary

    def format_ui_table(self) -> str:
        """Format timing summary as markdown table for Chainlit UI.

        Returns:
            Markdown-formatted timing summary table.
        """
        if not self.iterations:
            return "No timing data collected."

        summary = self.get_summary()
        n_iterations = len(self.iterations)

        # Stage display names (order matters for display)
        stage_names = [
            ("capture_ms", "capture"),
            ("file_write_ms", "file_write"),
            ("ui_display_ms", "ui_display"),
            ("msg_build_ms", "msg_build"),
            ("inference_ttft_ms", "inference_ttft"),
            ("inference_tool_ms", "inference_tool"),
            ("inference_total_ms", "inference_total"),
            ("stop_check_ms", "stop_check"),
            ("delay_ms", "delay"),
        ]

        lines = [
            f"### Embodied Mode Timing ({n_iterations} iterations)",
            "",
            "| Stage | Avg | Min | Max | % |",
            "|-------|-----|-----|-----|---|",
        ]

        for field_name, display_name in stage_names:
            s = summary.get(field_name)
            if s:
                lines.append(
                    f"| {display_name} | {s.avg_ms:.0f}ms | {s.min_ms:.0f}ms | "
                    f"{s.max_ms:.0f}ms | {s.pct:.1f}% |"
                )

        # Total row
        total = summary.get("total_ms")
        if total:
            lines.append(
                f"| **TOTAL** | **{total.avg_ms:.0f}ms** | **{total.min_ms:.0f}ms** | "
                f"**{total.max_ms:.0f}ms** | |"
            )

        return "\n".join(lines)

    def save_to_file(self, logs_dir: Path = Path("logs")) -> str:
        """Save detailed JSON log with timestamp.

        Args:
            logs_dir: Directory to save logs (created if doesn't exist).

        Returns:
            Path to the saved file.
        """
        logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self._session_start.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"embodied_{timestamp}.json"
        filepath = logs_dir / filename

        # Build summary dict
        summary_dict = {}
        for stage_name, stats in self.get_summary().items():
            summary_dict[stage_name.replace("_ms", "")] = asdict(stats)

        data = {
            "session_start": self._session_start.isoformat(),
            "model": self._model,
            "image_settings": self._image_settings,
            "iterations": [asdict(t) for t in self.iterations],
            "summary": summary_dict,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return str(filepath)

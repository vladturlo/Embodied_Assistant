"""Mouse control tool for embodied agent.

This module provides mouse cursor control functionality using pyautogui.
Designed for use with a vision-based feedback loop where the model
observes visual input and controls the mouse accordingly.

Windows-only implementation (pyautogui works cross-platform but tested on Windows).
"""

import threading
import time
from collections import deque
from typing import Optional

import pyautogui

# Safety settings
pyautogui.FAILSAFE = True   # Move mouse to screen corner to abort
pyautogui.PAUSE = 0.01      # Minimal pause between actions (10ms)

# Movement limits
MAX_DISTANCE = 200  # Maximum pixels per move


def move_mouse(direction: str, distance: int = 50) -> str:
    """Move the mouse cursor in a direction.

    Args:
        direction: One of "up", "down", "left", "right"
        distance: Pixels to move (default 50, max 200)

    Returns:
        Result message with new cursor position.
    """
    # Enforce safety limit
    distance = max(1, min(distance, MAX_DISTANCE))

    # Calculate movement delta
    dx, dy = 0, 0
    direction = direction.lower().strip()

    directions = {
        "up": (0, -distance),
        "down": (0, distance),
        "left": (-distance, 0),
        "right": (distance, 0),
    }

    if direction not in directions:
        return f"Error: Invalid direction '{direction}'. Use up/down/left/right."

    dx, dy = directions[direction]

    try:
        pyautogui.moveRel(dx, dy)
        x, y = pyautogui.position()
        return f"Moved mouse {direction} by {distance}px. New position: ({x}, {y})"
    except pyautogui.FailSafeException:
        return "FAILSAFE triggered - mouse moved to screen corner. Aborting."
    except Exception as e:
        return f"Error moving mouse: {e}"


def get_mouse_position() -> str:
    """Get current mouse cursor position.

    Returns:
        String with current mouse coordinates.
    """
    try:
        x, y = pyautogui.position()
        return f"Mouse position: ({x}, {y})"
    except Exception as e:
        return f"Error getting mouse position: {e}"


def get_screen_size() -> str:
    """Get screen dimensions.

    Returns:
        String with screen width and height.
    """
    try:
        w, h = pyautogui.size()
        return f"Screen size: {w}x{h}"
    except Exception as e:
        return f"Error getting screen size: {e}"


class SmoothMouseController:
    """Continuous cursor movement with direction averaging.

    Instead of discrete jumps, this controller moves the cursor smoothly
    at a constant speed. Direction updates from the model are averaged
    over the last N samples to create fluid motion.
    """

    # Direction string to unit vector mapping
    DIRECTION_VECTORS = {
        "up": (0.0, -1.0),
        "down": (0.0, 1.0),
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "up-left": (-0.707, -0.707),
        "up-right": (0.707, -0.707),
        "down-left": (-0.707, 0.707),
        "down-right": (0.707, 0.707),
    }

    def __init__(self, num_slots: int = 8, speed_px_per_sec: float = 200.0):
        """Initialize the smooth mouse controller.

        Args:
            num_slots: Number of direction samples to average (default 8).
            speed_px_per_sec: Cursor movement speed in pixels per second.
        """
        self.num_slots = num_slots
        self.speed = speed_px_per_sec
        self.direction_history: deque[tuple[float, float]] = deque(maxlen=num_slots)
        self.current_velocity: tuple[float, float] = (0.0, 0.0)
        self.running = False
        self.failsafe_triggered = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        """Start the background movement thread."""
        self.running = True
        self.failsafe_triggered = False
        self._thread = threading.Thread(target=self._movement_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background movement thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=0.5)

    def update_direction(self, direction: Optional[str]):
        """Update the movement direction based on model output.

        Args:
            direction: Direction string ("up", "down", "left", "right", or diagonals).
                       Pass None to stop movement immediately.
        """
        if direction is None:
            # No clear direction detected — stop immediately
            with self._lock:
                self.direction_history.clear()
                self.current_velocity = (0.0, 0.0)
            return

        vector = self.DIRECTION_VECTORS.get(direction.lower().strip())
        if vector is None:
            # Unknown direction string — stop
            with self._lock:
                self.direction_history.clear()
                self.current_velocity = (0.0, 0.0)
            return

        with self._lock:
            self.direction_history.append(vector)
            # Average all directions in history
            avg_dx = sum(d[0] for d in self.direction_history) / len(self.direction_history)
            avg_dy = sum(d[1] for d in self.direction_history) / len(self.direction_history)
            # Normalize to unit vector
            mag = (avg_dx**2 + avg_dy**2) ** 0.5
            if mag > 0.01:
                self.current_velocity = (avg_dx / mag, avg_dy / mag)
            else:
                self.current_velocity = (0.0, 0.0)

    def _movement_loop(self):
        """Background thread: move cursor at constant speed in current direction."""
        interval = 0.016  # ~60 FPS
        while self.running:
            with self._lock:
                vx, vy = self.current_velocity

            if abs(vx) > 0.01 or abs(vy) > 0.01:
                # Move by speed * interval pixels
                dx = vx * self.speed * interval
                dy = vy * self.speed * interval
                try:
                    pyautogui.moveRel(dx, dy, _pause=False)
                except pyautogui.FailSafeException:
                    self.failsafe_triggered = True
                    self.running = False
                    break

            time.sleep(interval)


# For testing
if __name__ == "__main__":
    print("Mouse Control Test")
    print("=" * 40)
    print(get_screen_size())
    print(get_mouse_position())
    print("\nMoving mouse right by 50px...")
    print(move_mouse("right", 50))
    print("\nMoving mouse down by 50px...")
    print(move_mouse("down", 50))
    print("\nFinal position:")
    print(get_mouse_position())

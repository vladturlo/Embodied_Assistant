"""Mouse control tool for embodied agent.

This module provides mouse cursor control functionality using pyautogui.
Designed for use with a vision-based feedback loop where the model
observes visual input and controls the mouse accordingly.

Windows-only implementation (pyautogui works cross-platform but tested on Windows).
"""

import pyautogui

# Safety settings
pyautogui.FAILSAFE = True   # Move mouse to screen corner to abort
pyautogui.PAUSE = 0.1       # Small pause between actions (100ms)

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

    if direction == "up":
        dy = -distance
    elif direction == "down":
        dy = distance
    elif direction == "left":
        dx = -distance
    elif direction == "right":
        dx = distance
    else:
        return f"Error: Invalid direction '{direction}'. Use up/down/left/right."

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

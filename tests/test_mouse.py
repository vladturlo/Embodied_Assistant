"""Tests for the mouse control tool.

Run with:
    python tests/test_mouse.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.mouse import move_mouse, get_mouse_position, get_screen_size


def test_get_screen_size():
    """Test getting screen size."""
    print("\n=== Testing get_screen_size ===")
    result = get_screen_size()
    print(f"  {result}")
    assert "Screen size:" in result
    assert "x" in result
    print("get_screen_size test PASSED")


def test_get_mouse_position():
    """Test getting mouse position."""
    print("\n=== Testing get_mouse_position ===")
    result = get_mouse_position()
    print(f"  {result}")
    assert "Mouse position:" in result
    assert "(" in result and ")" in result
    print("get_mouse_position test PASSED")


def test_move_mouse_directions():
    """Test moving mouse in all directions."""
    print("\n=== Testing move_mouse directions ===")

    # Get initial position
    initial = get_mouse_position()
    print(f"  Initial: {initial}")

    # Test each direction
    for direction in ["right", "down", "left", "up"]:
        result = move_mouse(direction, 10)
        print(f"  {direction}: {result}")
        assert "Moved mouse" in result or "Error" not in result

    print("move_mouse directions test PASSED")


def test_move_mouse_distance_limit():
    """Test that distance is limited to MAX_DISTANCE."""
    print("\n=== Testing distance limit ===")

    # Try to move 1000px (should be capped to 200)
    result = move_mouse("right", 1000)
    print(f"  Result: {result}")
    # The result should show 200px, not 1000px
    assert "200px" in result
    print("distance limit test PASSED")


def test_invalid_direction():
    """Test invalid direction handling."""
    print("\n=== Testing invalid direction ===")

    result = move_mouse("diagonal", 50)
    print(f"  Result: {result}")
    assert "Error" in result
    assert "Invalid direction" in result
    print("invalid direction test PASSED")


def test_case_insensitive_direction():
    """Test that directions are case-insensitive."""
    print("\n=== Testing case insensitivity ===")

    for direction in ["UP", "Down", "LEFT", "RiGhT"]:
        result = move_mouse(direction, 5)
        print(f"  {direction}: {result}")
        assert "Moved mouse" in result
    print("case insensitivity test PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Mouse Control Tests")
    print("=" * 50)
    print("\nWARNING: This test will move your mouse cursor!")
    print("Move mouse to screen corner to abort (FAILSAFE).\n")

    test_get_screen_size()
    test_get_mouse_position()
    test_move_mouse_directions()
    test_move_mouse_distance_limit()
    test_invalid_direction()
    test_case_insensitive_direction()

    print("\n" + "=" * 50)
    print("All tests PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()

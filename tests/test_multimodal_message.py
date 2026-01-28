"""Test AutoGen MultiModalMessage creation and handling.

This test verifies that MultiModalMessage objects can be created
with text and images for use with vision models.

Usage:
    python tests/test_multimodal_message.py
"""

import sys
from pathlib import Path

from PIL import Image as PILImage


def test_text_message() -> bool:
    """Test TextMessage creation.

    Returns:
        True if TextMessage is created successfully.
    """
    print(f"\n{'='*60}")
    print("Test 1: TextMessage Creation")
    print(f"{'='*60}")

    try:
        from autogen_agentchat.messages import TextMessage

        msg = TextMessage(content="Hello world", source="user")
        print(f"TextMessage created:")
        print(f"  Content: {msg.content}")
        print(f"  Source: {msg.source}")
        print(f"  Type: {type(msg).__name__}")

        if msg.content == "Hello world" and msg.source == "user":
            print(f"\n[PASS] TextMessage created successfully")
            return True
        else:
            print(f"\n[FAIL] TextMessage values incorrect")
            return False

    except ImportError as e:
        print(f"\n[FAIL] Import error: {e}")
        print("Make sure autogen-agentchat is installed")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


def test_agimage_creation() -> bool:
    """Test AGImage creation from PIL Image.

    Returns:
        True if AGImage is created successfully.
    """
    print(f"\n{'='*60}")
    print("Test 2: AGImage Creation")
    print(f"{'='*60}")

    try:
        from autogen_core import Image as AGImage

        # Create a test PIL image
        pil_img = PILImage.new('RGB', (100, 100), color='red')
        ag_image = AGImage(pil_img)

        print(f"AGImage created from PIL Image:")
        print(f"  Type: {type(ag_image).__name__}")
        print(f"  Base64 length: {len(ag_image.to_base64())}")
        print(f"  Data URI prefix: {ag_image.data_uri[:50]}...")

        print(f"\n[PASS] AGImage created successfully")
        return True

    except ImportError as e:
        print(f"\n[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


def test_multimodal_message_text_only() -> bool:
    """Test MultiModalMessage with text only.

    Returns:
        True if message is created successfully.
    """
    print(f"\n{'='*60}")
    print("Test 3: MultiModalMessage (Text Only)")
    print(f"{'='*60}")

    try:
        from autogen_agentchat.messages import MultiModalMessage

        msg = MultiModalMessage(content=["Hello world"], source="user")

        print(f"MultiModalMessage created (text only):")
        print(f"  Content: {msg.content}")
        print(f"  Content length: {len(msg.content)}")
        print(f"  Source: {msg.source}")

        if len(msg.content) == 1 and msg.content[0] == "Hello world":
            print(f"\n[PASS] MultiModalMessage (text) created successfully")
            return True
        else:
            print(f"\n[FAIL] MultiModalMessage values incorrect")
            return False

    except ImportError as e:
        print(f"\n[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


def test_multimodal_message_with_image() -> bool:
    """Test MultiModalMessage with text and image.

    Returns:
        True if message is created successfully.
    """
    print(f"\n{'='*60}")
    print("Test 4: MultiModalMessage (Text + Image)")
    print(f"{'='*60}")

    try:
        from autogen_agentchat.messages import MultiModalMessage
        from autogen_core import Image as AGImage

        # Create test image
        pil_img = PILImage.new('RGB', (100, 100), color='blue')
        ag_image = AGImage(pil_img)

        msg = MultiModalMessage(
            content=["Describe this image:", ag_image],
            source="user"
        )

        print(f"MultiModalMessage created (text + image):")
        print(f"  Content length: {len(msg.content)}")
        print(f"  Content types: {[type(c).__name__ for c in msg.content]}")
        print(f"  Source: {msg.source}")

        if len(msg.content) == 2:
            print(f"\n[PASS] MultiModalMessage (text + image) created successfully")
            return True
        else:
            print(f"\n[FAIL] Content length incorrect")
            return False

    except ImportError as e:
        print(f"\n[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multimodal_message_multiple_images() -> bool:
    """Test MultiModalMessage with multiple images.

    Returns:
        True if message is created successfully.
    """
    print(f"\n{'='*60}")
    print("Test 5: MultiModalMessage (Multiple Images)")
    print(f"{'='*60}")

    try:
        from autogen_agentchat.messages import MultiModalMessage
        from autogen_core import Image as AGImage

        # Create multiple test images
        colors = ['red', 'green', 'blue']
        images = []
        for color in colors:
            pil_img = PILImage.new('RGB', (50, 50), color=color)
            images.append(AGImage(pil_img))

        msg = MultiModalMessage(
            content=["Compare these images:"] + images,
            source="user"
        )

        print(f"MultiModalMessage created (text + {len(images)} images):")
        print(f"  Content length: {len(msg.content)}")
        print(f"  Content types: {[type(c).__name__ for c in msg.content]}")

        if len(msg.content) == 4:  # 1 text + 3 images
            print(f"\n[PASS] MultiModalMessage (multiple images) created successfully")
            return True
        else:
            print(f"\n[FAIL] Content length incorrect")
            return False

    except ImportError as e:
        print(f"\n[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


def test_image_from_file() -> bool:
    """Test loading image from file.

    Returns:
        True if image is loaded successfully.
    """
    print(f"\n{'='*60}")
    print("Test 6: Image from File")
    print(f"{'='*60}")

    try:
        from autogen_core import Image as AGImage
        import tempfile

        # Create a temporary test image file
        temp_dir = Path(tempfile.mkdtemp())
        temp_path = temp_dir / "test_image.png"

        pil_img = PILImage.new('RGB', (200, 200), color='purple')
        pil_img.save(str(temp_path))

        print(f"Created test image: {temp_path}")

        # Load with AGImage.from_file
        ag_image = AGImage.from_file(temp_path)

        print(f"Loaded image from file:")
        print(f"  Type: {type(ag_image).__name__}")
        print(f"  Base64 length: {len(ag_image.to_base64())}")

        print(f"\n[PASS] Image loaded from file successfully")
        return True

    except ImportError as e:
        print(f"\n[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_from_base64() -> bool:
    """Test creating image from base64.

    Returns:
        True if image is created successfully.
    """
    print(f"\n{'='*60}")
    print("Test 7: Image from Base64")
    print(f"{'='*60}")

    try:
        from autogen_core import Image as AGImage
        import base64
        import io

        # Create a simple image and encode to base64
        pil_img = PILImage.new('RGB', (10, 10), color='yellow')
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        b64_string = base64.b64encode(buffer.getvalue()).decode()

        print(f"Created base64 string (length: {len(b64_string)})")

        # Create AGImage from base64
        ag_image = AGImage.from_base64(b64_string)

        print(f"Created AGImage from base64:")
        print(f"  Type: {type(ag_image).__name__}")
        print(f"  Roundtrip base64 length: {len(ag_image.to_base64())}")

        print(f"\n[PASS] Image from base64 created successfully")
        return True

    except ImportError as e:
        print(f"\n[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_message_to_model_format() -> bool:
    """Test converting message to model format.

    Returns:
        True if conversion works.
    """
    print(f"\n{'='*60}")
    print("Test 8: Message to Model Format")
    print(f"{'='*60}")

    try:
        from autogen_agentchat.messages import MultiModalMessage
        from autogen_core import Image as AGImage

        # Create message with image
        pil_img = PILImage.new('RGB', (50, 50), color='cyan')
        ag_image = AGImage(pil_img)

        msg = MultiModalMessage(
            content=["What is this?", ag_image],
            source="user"
        )

        # Test to_text method
        text_repr = msg.to_text()
        print(f"to_text(): {text_repr[:100]}...")

        # Test to_model_text method
        model_text = msg.to_model_text()
        print(f"to_model_text(): {model_text[:100]}...")

        print(f"\n[PASS] Message conversion successful")
        return True

    except ImportError as e:
        print(f"\n[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all MultiModalMessage tests."""
    print("\n" + "="*60)
    print("MULTIMODAL MESSAGE TESTS")
    print("="*60)

    results = {
        "text_message": test_text_message(),
        "agimage_creation": test_agimage_creation(),
        "multimodal_text_only": test_multimodal_message_text_only(),
        "multimodal_with_image": test_multimodal_message_with_image(),
        "multimodal_multiple_images": test_multimodal_message_multiple_images(),
        "image_from_file": test_image_from_file(),
        "image_from_base64": test_image_from_base64(),
        "message_to_model_format": test_message_to_model_format(),
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(results.values())
    total = len(results)
    print(f"Passed: {passed}/{total}")

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    if passed == total:
        print("\nAll tests passed! MultiModalMessage handling is ready.")
        return 0
    else:
        print("\nSome tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

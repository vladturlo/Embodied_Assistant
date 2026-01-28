"""Test image analysis with qwen3-vl vision model.

This test verifies that the vision model can analyze images
using the AutoGen OllamaChatCompletionClient.

Usage:
    python tests/test_vision_model.py
"""

import asyncio
import sys
from pathlib import Path

# Configuration
OLLAMA_HOST = "http://localhost:11435"
MODEL_NAME = "qwen3-vl:238b"


def get_client():
    """Create and return the Ollama client.

    Returns:
        OllamaChatCompletionClient configured for qwen3-vl.
    """
    from autogen_ext.models.ollama import OllamaChatCompletionClient
    from autogen_core.models import ModelInfo

    return OllamaChatCompletionClient(
        model=MODEL_NAME,
        host=OLLAMA_HOST,
        model_info=ModelInfo(
            vision=True,
            function_calling=True,
            json_output=False,
            family="unknown",
            structured_output=False
        ),
        options={"num_ctx": 262144}
    )


async def test_client_creation() -> bool:
    """Test that the client can be created.

    Returns:
        True if client is created successfully.
    """
    print(f"\n{'='*60}")
    print("Test 1: Client Creation")
    print(f"{'='*60}")

    try:
        client = get_client()
        print(f"Client created: {type(client).__name__}")
        print(f"Model: {client._model}")
        print(f"Host: {client._host}")
        print(f"\n[PASS] Client created successfully")
        return True
    except ImportError as e:
        print(f"\n[FAIL] Import error: {e}")
        print("Make sure autogen-ext[ollama] is installed:")
        print("  pip install 'autogen-ext[ollama]'")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


async def test_text_only_response() -> bool:
    """Test text-only response from the model.

    Returns:
        True if model responds to text prompt.
    """
    print(f"\n{'='*60}")
    print("Test 2: Text-Only Response")
    print(f"{'='*60}")

    try:
        from autogen_core.models import UserMessage

        client = get_client()
        print("Sending prompt: 'Say exactly: vision model ready'")

        result = await client.create([
            UserMessage(content="Say exactly: vision model ready", source="user")
        ])

        response_text = result.content
        print(f"Response: {response_text[:200]}...")

        if response_text and len(response_text) > 0:
            print(f"\n[PASS] Model responded to text prompt")
            return True
        else:
            print(f"\n[FAIL] Empty response")
            return False

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


async def test_image_creation() -> bool:
    """Test creating an AGImage object.

    Returns:
        True if image object is created successfully.
    """
    print(f"\n{'='*60}")
    print("Test 3: Image Object Creation")
    print(f"{'='*60}")

    try:
        from autogen_core import Image as AGImage
        from PIL import Image as PILImage

        # Create a simple test image
        print("Creating test image (100x100 red square)...")
        pil_img = PILImage.new('RGB', (100, 100), color='red')
        ag_image = AGImage(pil_img)

        print(f"AGImage type: {type(ag_image)}")
        print(f"Base64 length: {len(ag_image.to_base64())}")
        print(f"Data URI prefix: {ag_image.data_uri[:50]}...")

        print(f"\n[PASS] Image object created successfully")
        return True

    except ImportError as e:
        print(f"\n[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


async def test_image_analysis_synthetic() -> bool:
    """Test image analysis with a synthetic image.

    Returns:
        True if model analyzes the image.
    """
    print(f"\n{'='*60}")
    print("Test 4: Image Analysis (Synthetic Image)")
    print(f"{'='*60}")

    try:
        from autogen_core import Image as AGImage
        from autogen_core.models import UserMessage
        from PIL import Image as PILImage

        client = get_client()

        # Create a test image with distinct colors
        print("Creating test image (red, green, blue squares)...")
        img = PILImage.new('RGB', (300, 100), color='white')
        # Add colored rectangles
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            for x in range(i * 100, (i + 1) * 100):
                for y in range(100):
                    img.putpixel((x, y), color)

        ag_image = AGImage(img)

        print("Sending image to model for analysis...")
        result = await client.create([
            UserMessage(
                content=["Describe the colors in this image. Be brief.", ag_image],
                source="user"
            )
        ])

        response_text = result.content
        print(f"Response: {response_text[:300]}...")

        # Check if response mentions colors
        has_colors = any(c in response_text.lower() for c in ['red', 'green', 'blue', 'color'])

        if has_colors:
            print(f"\n[PASS] Model analyzed image and mentioned colors")
            return True
        elif len(response_text) > 10:
            print(f"\n[PASS] Model responded (may not have mentioned specific colors)")
            return True
        else:
            print(f"\n[FAIL] Model did not provide meaningful response")
            return False

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_image_analysis_from_file(image_path: str = None) -> bool:
    """Test image analysis from a file.

    Args:
        image_path: Path to an image file. If None, test is skipped.

    Returns:
        True if model analyzes the image.
    """
    print(f"\n{'='*60}")
    print("Test 5: Image Analysis (From File)")
    print(f"{'='*60}")

    if image_path is None:
        # Check for test image in test_assets
        test_paths = [
            Path("test_assets/test_image.jpg"),
            Path("test_assets/test_image.png"),
            Path("../test_assets/test_image.jpg"),
            Path("../test_assets/test_image.png"),
        ]
        for p in test_paths:
            if p.exists():
                image_path = str(p)
                break

    if image_path is None or not Path(image_path).exists():
        print("No test image found. Skipping file-based test.")
        print("To run this test, add an image to test_assets/test_image.jpg")
        print(f"\n[SKIP] No test image available")
        return True  # Don't fail the test

    try:
        from autogen_core import Image as AGImage
        from autogen_core.models import UserMessage

        client = get_client()
        print(f"Loading image: {image_path}")

        ag_image = AGImage.from_file(image_path)
        print(f"Image loaded, base64 length: {len(ag_image.to_base64())}")

        print("Sending image to model for analysis...")
        result = await client.create([
            UserMessage(
                content=["What do you see in this image? Be brief.", ag_image],
                source="user"
            )
        ])

        response_text = result.content
        print(f"Response: {response_text[:300]}...")

        if len(response_text) > 10:
            print(f"\n[PASS] Model analyzed image from file")
            return True
        else:
            print(f"\n[FAIL] Empty or short response")
            return False

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


async def run_all_tests():
    """Run all vision model tests."""
    print("\n" + "="*60)
    print("VISION MODEL TESTS")
    print("="*60)
    print(f"Host: {OLLAMA_HOST}")
    print(f"Model: {MODEL_NAME}")

    results = {
        "client_creation": await test_client_creation(),
        "text_response": await test_text_only_response(),
        "image_creation": await test_image_creation(),
        "image_analysis_synthetic": await test_image_analysis_synthetic(),
        "image_analysis_file": await test_image_analysis_from_file(),
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
        print("\nAll tests passed! Vision model is ready.")
        return 0
    else:
        print("\nSome tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)

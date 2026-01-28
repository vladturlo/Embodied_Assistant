"""Test Ollama API connectivity and model availability.

Run this test first to verify the Ollama server is reachable
and the qwen3-vl:238b model is available.

Usage:
    python tests/test_ollama_connection.py
"""

import asyncio
import sys

import httpx

# Configuration
OLLAMA_HOST = "http://localhost:11435"
MODEL_NAME = "qwen3-vl:238b"


async def test_ollama_health() -> bool:
    """Test if Ollama server is reachable.

    Returns:
        True if server responds with 200 status.
    """
    print(f"\n{'='*60}")
    print("Test 1: Ollama Server Health Check")
    print(f"{'='*60}")
    print(f"Connecting to: {OLLAMA_HOST}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags")
            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                print(f"Available models: {len(models)}")
                for model in models[:5]:  # Show first 5 models
                    print(f"  - {model.get('name', 'unknown')}")
                if len(models) > 5:
                    print(f"  ... and {len(models) - 5} more")
                print("\n[PASS] Ollama server is healthy")
                return True
            else:
                print(f"\n[FAIL] Unexpected status code: {response.status_code}")
                return False

    except httpx.ConnectError as e:
        print(f"\n[FAIL] Connection error: {e}")
        print("Make sure the SSH tunnel is active:")
        print("  ssh -L 11435:localhost:11434 user@remote-server")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


async def test_model_available() -> bool:
    """Test if qwen3-vl:238b model is available.

    Returns:
        True if model is found in available models.
    """
    print(f"\n{'='*60}")
    print("Test 2: Model Availability Check")
    print(f"{'='*60}")
    print(f"Looking for model: {MODEL_NAME}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags")

            if response.status_code != 200:
                print(f"\n[FAIL] Could not fetch models: {response.status_code}")
                return False

            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]

            # Check for exact match or partial match
            found = MODEL_NAME in models or any(MODEL_NAME in m for m in models)

            if found:
                matching = [m for m in models if MODEL_NAME in m]
                print(f"Found matching models: {matching}")
                print(f"\n[PASS] Model {MODEL_NAME} is available")
                return True
            else:
                print(f"Available models: {models}")
                print(f"\n[FAIL] Model {MODEL_NAME} not found")
                print("You may need to pull the model:")
                print(f"  ollama pull {MODEL_NAME}")
                return False

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


async def test_simple_generation() -> bool:
    """Test basic text generation with the model.

    Returns:
        True if model generates a response.
    """
    print(f"\n{'='*60}")
    print("Test 3: Simple Text Generation")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print("Prompt: 'Hello, respond with OK if you can hear me.'")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": "Hello, respond with OK if you can hear me.",
                    "stream": False,
                    "options": {
                        "num_ctx": 2048,  # Small context for quick test
                    }
                }
            )

            if response.status_code != 200:
                print(f"\n[FAIL] Generation failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False

            data = response.json()
            generated = data.get("response", "")
            print(f"\nGenerated response: {generated[:200]}...")

            if generated:
                print(f"\n[PASS] Model generated response successfully")
                return True
            else:
                print(f"\n[FAIL] Empty response from model")
                return False

    except httpx.TimeoutException:
        print(f"\n[FAIL] Request timed out (120s)")
        print("The model may be loading. Try again in a few minutes.")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


async def test_model_info() -> bool:
    """Get detailed information about the model.

    Returns:
        True if model info is retrieved.
    """
    print(f"\n{'='*60}")
    print("Test 4: Model Information")
    print(f"{'='*60}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/show",
                json={"name": MODEL_NAME}
            )

            if response.status_code != 200:
                print(f"\n[FAIL] Could not get model info: {response.status_code}")
                return False

            data = response.json()
            print(f"Model: {data.get('modelfile', 'N/A')[:100]}...")
            print(f"Parameters: {data.get('parameters', 'N/A')}")
            print(f"Template: {data.get('template', 'N/A')[:100]}...")

            details = data.get("details", {})
            print(f"\nDetails:")
            print(f"  Format: {details.get('format', 'N/A')}")
            print(f"  Family: {details.get('family', 'N/A')}")
            print(f"  Parameter Size: {details.get('parameter_size', 'N/A')}")
            print(f"  Quantization: {details.get('quantization_level', 'N/A')}")

            print(f"\n[PASS] Model info retrieved")
            return True

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


async def run_all_tests():
    """Run all Ollama connection tests."""
    print("\n" + "="*60)
    print("OLLAMA CONNECTION TESTS")
    print("="*60)
    print(f"Host: {OLLAMA_HOST}")
    print(f"Model: {MODEL_NAME}")

    results = {
        "health": await test_ollama_health(),
        "model_available": await test_model_available(),
        "generation": await test_simple_generation(),
        "model_info": await test_model_info(),
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
        print("\nAll tests passed! Ollama connection is ready.")
        return 0
    else:
        print("\nSome tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)

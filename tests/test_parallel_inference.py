"""Benchmark: Sequential vs Concurrent Inference Latency.

Tests whether OLLAMA_NUM_PARALLEL=2 enables faster effective
control frequency via staggered concurrent requests.

Prerequisites:
  - Set OLLAMA_NUM_PARALLEL=2 on server before running concurrent tests
  - SSH tunnel active (localhost:11435)

Usage:
    python tests/test_parallel_inference.py
"""

import asyncio
import time

from PIL import Image as PILImage

from autogen_core import Image as AGImage
from autogen_core.models import ModelInfo, SystemMessage, UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient

OLLAMA_HOST = "http://localhost:11435"
MODEL_NAME = "ministral-3:8b"
NUM_REQUESTS = 10
SYSTEM_PROMPT = "You control a mouse cursor. State what you see briefly."


def create_client():
    return OllamaChatCompletionClient(
        model=MODEL_NAME,
        host=OLLAMA_HOST,
        model_info=ModelInfo(
            vision=True,
            function_calling=True,
            json_output=False,
            family="mistral",
        ),
        options={"num_ctx": 16384, "num_predict": 128, "temperature": 0.15},
    )


def make_test_image():
    """Create a 480x360 synthetic image (simulates webcam frame)."""
    img = PILImage.new("RGB", (480, 360), color=(100, 150, 200))
    return AGImage(img)


async def single_request(client, image, request_id):
    """Send one vision request and measure timing."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT, source="system"),
        UserMessage(
            content=["What direction is the thumb pointing?", image],
            source="user",
        ),
    ]
    start = time.perf_counter()
    result = await client.create(messages)
    elapsed = (time.perf_counter() - start) * 1000
    return {
        "id": request_id,
        "elapsed_ms": elapsed,
        "response": result.content[:80],
    }


async def test_sequential(n=NUM_REQUESTS):
    """Test 1: Sequential requests (baseline)."""
    client = create_client()
    image = make_test_image()
    results = []

    # Warmup (first request loads model)
    print("  Warmup...")
    await single_request(client, image, -1)

    print(f"  Running {n} sequential requests...")
    for i in range(n):
        r = await single_request(client, image, i)
        results.append(r)
        print(f"    Request {i}: {r['elapsed_ms']:.0f}ms")

    latencies = [r["elapsed_ms"] for r in results]
    return {
        "avg_ms": sum(latencies) / len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "results": results,
    }


async def test_concurrent_pairs(n_pairs=NUM_REQUESTS // 2):
    """Test 2: Concurrent pairs (2 at a time)."""
    client = create_client()
    image = make_test_image()
    results = []

    # Warmup
    print("  Warmup...")
    await single_request(client, image, -1)

    print(f"  Running {n_pairs} concurrent pairs...")
    for i in range(n_pairs):
        pair_start = time.perf_counter()
        r1, r2 = await asyncio.gather(
            single_request(client, image, i * 2),
            single_request(client, image, i * 2 + 1),
        )
        pair_elapsed = (time.perf_counter() - pair_start) * 1000
        results.extend([r1, r2])
        print(
            f"    Pair {i}: {r1['elapsed_ms']:.0f}ms, {r2['elapsed_ms']:.0f}ms"
            f" (wall: {pair_elapsed:.0f}ms)"
        )

    latencies = [r["elapsed_ms"] for r in results]
    return {
        "avg_ms": sum(latencies) / len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "results": results,
    }


async def test_staggered_pipeline(n=NUM_REQUESTS):
    """Test 3: Staggered pipeline (always 2 in flight)."""
    client = create_client()
    image = make_test_image()

    # Warmup
    print("  Warmup...")
    await single_request(client, image, -1)

    print(f"  Running {n} staggered requests (2 in flight)...")
    completion_times = []
    pending = set()
    next_id = 0
    pipeline_start = time.perf_counter()

    # Start 2 initial tasks
    for _ in range(2):
        task = asyncio.create_task(single_request(client, image, next_id))
        pending.add(task)
        next_id += 1
        await asyncio.sleep(0.2)  # 200ms stagger between submissions

    while pending:
        done, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            r = task.result()
            t = time.perf_counter()
            completion_times.append(t)
            print(
                f"    Request {r['id']}: {r['elapsed_ms']:.0f}ms"
                f" (t={((t - pipeline_start) * 1000):.0f}ms)"
            )

            # Submit replacement if we haven't sent all requests yet
            if next_id < n:
                new_task = asyncio.create_task(
                    single_request(client, image, next_id)
                )
                pending.add(new_task)
                next_id += 1

    # Calculate intervals between consecutive completions
    intervals = []
    for i in range(1, len(completion_times)):
        intervals.append((completion_times[i] - completion_times[i - 1]) * 1000)

    return {
        "avg_interval_ms": sum(intervals) / len(intervals) if intervals else 0,
        "min_interval_ms": min(intervals) if intervals else 0,
        "max_interval_ms": max(intervals) if intervals else 0,
        "total_ms": (completion_times[-1] - pipeline_start) * 1000,
        "intervals": intervals,
    }


async def main():
    print("=" * 60)
    print("PARALLEL INFERENCE BENCHMARK")
    print("=" * 60)
    print(f"Host: {OLLAMA_HOST}")
    print(f"Model: {MODEL_NAME}")
    print(f"Requests per test: {NUM_REQUESTS}")
    print()

    # Test 1: Sequential
    print("TEST 1: Sequential (baseline)")
    seq = await test_sequential()
    print(
        f"  => Avg: {seq['avg_ms']:.0f}ms"
        f"  Min: {seq['min_ms']:.0f}ms"
        f"  Max: {seq['max_ms']:.0f}ms"
    )
    print()

    # Test 2: Concurrent pairs
    print("TEST 2: Concurrent pairs (2 simultaneous)")
    pairs = await test_concurrent_pairs()
    print(
        f"  => Avg per-request: {pairs['avg_ms']:.0f}ms"
        f"  Min: {pairs['min_ms']:.0f}ms"
        f"  Max: {pairs['max_ms']:.0f}ms"
    )
    print()

    # Test 3: Staggered pipeline
    print("TEST 3: Staggered pipeline (2 in flight)")
    stag = await test_staggered_pipeline()
    print(
        f"  => Avg interval between completions: {stag['avg_interval_ms']:.0f}ms"
    )
    print(
        f"  => Min: {stag['min_interval_ms']:.0f}ms"
        f"  Max: {stag['max_interval_ms']:.0f}ms"
    )
    print(
        f"  => Total wall time for {NUM_REQUESTS} requests:"
        f" {stag['total_ms']:.0f}ms"
    )
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sequential:   {seq['avg_ms']:.0f}ms per request")
    print(
        f"Concurrent:   {pairs['avg_ms']:.0f}ms per request"
        f" ({pairs['avg_ms']/seq['avg_ms']:.2f}x sequential)"
    )
    print(
        f"Pipeline:     {stag['avg_interval_ms']:.0f}ms between completions"
        f" ({stag['avg_interval_ms']/seq['avg_ms']:.2f}x sequential)"
    )
    print()

    slowdown = pairs["avg_ms"] / seq["avg_ms"]
    if slowdown < 2.0:
        print(f"VIABLE: Concurrent slowdown is {slowdown:.2f}x (<2x)")
        print(
            f"  => Staggered pipeline would deliver moves"
            f" every ~{stag['avg_interval_ms']:.0f}ms"
        )
        print(f"  => vs sequential every ~{seq['avg_ms']:.0f}ms")
        effective_speedup = seq["avg_ms"] / stag["avg_interval_ms"]
        print(f"  => {effective_speedup:.1f}x more frequent moves!")
    else:
        print(f"NOT VIABLE: Concurrent slowdown is {slowdown:.2f}x (>=2x)")
        print("  => Staggering would not improve effective frequency")


if __name__ == "__main__":
    asyncio.run(main())

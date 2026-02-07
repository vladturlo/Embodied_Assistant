"""Multimodal AutoGen-Chainlit Agent.

This is the main application that integrates AutoGen with Chainlit
to provide a multimodal AI agent with webcam capabilities.

Usage:
    chainlit run app.py
"""

# Windows asyncio compatibility fix - must be before other imports
import sys
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import re
from pathlib import Path
from typing import List, Optional, cast

import chainlit as cl
import yaml
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    MultiModalMessage,
    TextMessage,
    ModelClientStreamingChunkEvent,
    ToolCallRequestEvent,
    ToolCallExecutionEvent,
)
from autogen_core import CancellationToken, Image as AGImage
from autogen_core.models import ModelInfo
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient

import asyncio
import io
import tempfile
import time

from PIL import Image as PILImage

from tools.webcam import (
    capture_webcam,
    capture_frame_bytes,
    capture_frame_from_cap,
    extract_video_frames,
    get_video_capture,
)
from tools.mouse import move_mouse, get_mouse_position, get_screen_size
from tools.profiler import EmbodiedProfiler

# Configuration
SERVER_BASE_URL = "http://localhost:11435"  # Ollama server (via SSH tunnel)
MODEL_NAME = "ministral-3:8b"  # Model name (set in model_config.yaml)
CONTEXT_SIZE = 262144  # 256K context
EMBODIED_CONTEXT_SIZE = 16384  # 16K - sufficient for ~10 buffered messages

# Video processing defaults (can be overridden in model_config.yaml)
VIDEO_FRAMES_PER_SECOND = 5.0
VIDEO_MAX_FRAMES = 50

# Image capture defaults for embodied control (can be overridden in model_config.yaml)
IMAGE_MAX_WIDTH = 640
IMAGE_MAX_HEIGHT = 480
IMAGE_JPEG_QUALITY = 70

# Embodied control settings
EMBODIED_MAX_ITERATIONS = 500  # Safety limit
EMBODIED_NUM_PREDICT = 128  # Max output tokens (embodied responses are brief)

# Pipeline defaults (overridden by model_config.yaml pipeline section)
PIPELINE_MOVE_DISTANCE = 30
PIPELINE_INITIAL_STAGGER_S = 0.1  # 100ms stagger between initial slot submissions
EMBODIED_KEYWORDS = ["until", "keep going", "keep taking", "continuously", "feedback loop"]
STOP_INDICATORS = ["stop", "stopped", "stopping", "fist", "closed fist", "abort", "done", "finished"]

# System prompt for embodied control mode (minimal to avoid conflicting with user instruction)
EMBODIED_SYSTEM_PROMPT = """You control a mouse cursor based on visual input. Follow the user's instruction exactly.
Keep responses brief. State what you see, then act."""


def load_model_config() -> dict:
    """Load model configuration from YAML file.

    Also sets global video and image processing settings if defined.

    Returns:
        Configuration dictionary.
    """
    global VIDEO_FRAMES_PER_SECOND, VIDEO_MAX_FRAMES
    global IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT, IMAGE_JPEG_QUALITY

    config_path = Path(__file__).parent / "model_config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Load video settings if present
        video_config = config.get("video", {})
        if "frames_per_second" in video_config:
            VIDEO_FRAMES_PER_SECOND = float(video_config["frames_per_second"])
        if "max_frames" in video_config:
            VIDEO_MAX_FRAMES = int(video_config["max_frames"])

        # Load image settings if present
        image_config = config.get("image", {})
        if "max_width" in image_config:
            IMAGE_MAX_WIDTH = int(image_config["max_width"])
        if "max_height" in image_config:
            IMAGE_MAX_HEIGHT = int(image_config["max_height"])
        if "jpeg_quality" in image_config:
            IMAGE_JPEG_QUALITY = int(image_config["jpeg_quality"])

        return config
    return {}


def create_model_client(
    num_ctx: int | None = None,
    num_predict: int | None = None,
):
    """Create and configure the model client.

    Supports multiple providers via model_config.yaml:
    - ollama: Direct Ollama API (OllamaChatCompletionClient)
    - litellm/llamacpp/openai-compatible: OpenAI-compatible endpoint

    Args:
        num_ctx: Override context window size. Defaults to model_config or 256K.
        num_predict: Override max output tokens. None = unlimited.

    Returns:
        Configured model client (Ollama or OpenAI-compatible).
    """
    config = load_model_config()
    provider = config.get("provider", "ollama")
    model_info = ModelInfo(
        vision=True,
        function_calling=True,
        json_output=False,
        family=config.get("model_info", {}).get("family", "mistral"),
        structured_output=False,
    )

    if provider in ("litellm", "openai-compatible", "llamacpp"):
        # OpenAI-compatible endpoint (LiteLLM proxy, llama.cpp server, etc.)
        kwargs = {
            "model": config.get("model", MODEL_NAME),
            "base_url": config.get("base_url", f"{SERVER_BASE_URL}/v1"),
            "api_key": config.get("api_key", "not-needed"),
            "model_info": model_info,
            "temperature": config.get("options", {}).get("temperature", 0.15),
        }
        if num_predict is not None:
            kwargs["max_tokens"] = num_predict
        # Pass num_ctx through extra_body for backends that support it
        ctx = num_ctx or config.get("options", {}).get("num_ctx", CONTEXT_SIZE)
        extra = {"num_ctx": ctx}
        # Include llamacpp-specific options if present
        llamacpp_cfg = config.get("llamacpp", {})
        if llamacpp_cfg:
            extra.update(llamacpp_cfg)
        kwargs["extra_body"] = extra
        return OpenAIChatCompletionClient(**kwargs)
    else:
        # Ollama native API (default)
        context_size = num_ctx or config.get("options", {}).get("num_ctx", CONTEXT_SIZE)
        options: dict = {
            "temperature": config.get("options", {}).get("temperature", 0.15),
            "num_ctx": context_size,
        }
        if num_predict is not None:
            options["num_predict"] = num_predict
        return OllamaChatCompletionClient(
            model=config.get("model", MODEL_NAME),
            host=config.get("host", SERVER_BASE_URL),
            model_info=model_info,
            options=options,
        )


def create_embodied_agent(model_client, move_distance: int = 50) -> AssistantAgent:
    """Create agent for embodied control with bounded context.

    Uses BufferedChatCompletionContext to keep only the last ~2 frames
    of context, preventing inference degradation from context growth.

    Args:
        model_client: The model client to use.
        move_distance: Default mouse move distance in pixels.
            Sequential mode uses 50px, pipeline mode uses ~30px.

    Returns:
        AssistantAgent configured for embodied control.
    """
    # Closure-based tool with configurable default distance
    async def _mouse_move(direction: str, distance: Optional[int] = None) -> str:
        """Move the mouse cursor in a direction.

        Args:
            direction: One of "up", "down", "left", "right",
                "up-left", "up-right", "down-left", "down-right"
            distance: Pixels to move (max 200)

        Returns:
            Result message with new cursor position.
        """
        if distance is None:
            distance = move_distance
        return move_mouse(direction, distance)

    # Single-frame context: clear before each iteration, buffer is just a safety net
    # 1 frame = ~4 messages (user, assistant tool call, tool result, assistant response)
    model_context = BufferedChatCompletionContext(buffer_size=5)

    return AssistantAgent(
        name="embodied_assistant",
        model_client=model_client,
        model_context=model_context,  # Bounded context!
        tools=[_mouse_move, mouse_position_tool],
        system_message=EMBODIED_SYSTEM_PROMPT,
        model_client_stream=True,
        reflect_on_tool_use=False,
    )


# Wrapper function for the webcam tool
async def webcam_capture_tool(mode: str = "image", duration: Optional[float] = None) -> str:
    """Capture image or video from webcam.

    This tool captures media from the webcam and displays it in the chat.
    The captured file path is returned for the vision model to analyze.

    Args:
        mode: Either "image" for single frame or "video" for short clip.
        duration: Duration in seconds for video capture (1-10). Defaults to 3.0.

    Returns:
        Path to the captured file or error message.
    """
    if duration is None:
        duration = 3.0
    result = await capture_webcam(mode=mode, duration=duration)
    return result


async def mouse_move_tool(direction: str, distance: Optional[int] = None) -> str:
    """Move the mouse cursor in a direction.

    Use this tool to control the mouse based on visual input.
    For embodied control, combine with webcam_capture_tool to create
    a feedback loop: capture -> analyze -> move -> capture -> ...

    Args:
        direction: One of "up", "down", "left", "right"
        distance: Pixels to move (default 50, max 200)

    Returns:
        Result message with new cursor position.
    """
    if distance is None:
        distance = 50
    return move_mouse(direction, distance)


async def mouse_position_tool() -> str:
    """Get current mouse cursor position.

    Returns:
        String with current mouse coordinates.
    """
    return get_mouse_position()


@cl.set_starters
async def set_starters() -> List[cl.Starter]:
    """Set starter prompts for the chat interface.

    Returns:
        List of starter prompts.
    """
    return [
        cl.Starter(
            label="Embodied Control",
            message="Move the mouse in the direction my thumb is pointing. Keep taking snapshots and moving until I close my fist.",
        ),
        cl.Starter(
            label="Webcam Capture",
            message="Take a picture with the webcam and tell me what you see.",
        ),
        cl.Starter(
            label="Mouse Control",
            message="Move the mouse to the right by 100 pixels.",
        ),
        cl.Starter(
            label="Image Analysis",
            message="I'll upload an image for you to analyze.",
        ),
    ]


@cl.on_chat_start
async def start_chat() -> None:
    """Initialize the chat session with the multimodal agent."""
    # Create model client
    model_client = create_model_client()

    # Create the assistant agent with webcam and mouse tools
    assistant = AssistantAgent(
        name="multimodal_assistant",
        model_client=model_client,
        tools=[webcam_capture_tool, mouse_move_tool, mouse_position_tool],
        system_message="""You are a helpful multimodal AI assistant with vision and mouse control capabilities.

You can:
1. Analyze images that users upload
2. Capture images or short videos from the webcam using webcam_capture_tool
3. Control the mouse cursor using mouse_move_tool (directions: up, down, left, right)
4. Get the current mouse position using mouse_position_tool

When users ask you to "look", "see", "capture", or use the webcam, use webcam_capture_tool.
After capturing, analyze the image and describe what you see.

## Embodied Control Mode

When you receive an image with instructions to control the mouse based on visual input:

1. Analyze what you see in the image (pointing direction, hand gesture)
2. If the STOP CONDITION is met → respond with "STOP" and explain why
3. If the stop condition is NOT met → call mouse_move_tool with the detected direction

Images will be provided automatically - do NOT call webcam_capture_tool during embodied control.

Keep responses brief during embodied control. Just state what you see and call the tool.
Example: "Thumb pointing right." Then call mouse_move_tool(direction="right", distance=50)

SAFETY: Moving the mouse to any screen corner will trigger FAILSAFE and abort.

Be helpful, accurate, and descriptive in your visual analysis.""",
        model_client_stream=True,
        reflect_on_tool_use=False,  # Disabled: we handle image analysis explicitly after tool execution
    )

    # Store agent in session
    cl.user_session.set("agent", assistant)

    # Send welcome message
    await cl.Message(
        content="Hello! I'm a multimodal AI assistant with embodied control capabilities. I can:\n"
                "- Analyze images you upload\n"
                "- Capture images/videos from your webcam\n"
                "- Control the mouse cursor based on visual input\n"
                "- Answer questions about visual content\n\n"
                "**Embodied Control**: Try asking me to move the mouse based on your hand gestures!\n\n"
                "Example: \"Move the mouse in the direction my thumb is pointing. Keep going until I close my fist.\""
    ).send()


def detect_embodied_mode(text: str) -> bool:
    """Check if the message requests embodied control mode."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in EMBODIED_KEYWORDS)


def detect_stop_condition(text: str) -> bool:
    """Check if the model's response indicates stopping."""
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in STOP_INDICATORS)


async def run_agent_turn(agent: AssistantAgent, agent_message, embodied_mode: bool = False):
    """Run a single turn of the agent and return results.

    Returns:
        tuple: (response_text, captured_image_path, used_mouse, failsafe_triggered)
    """
    response_text = ""
    captured_image_path = None
    used_mouse = False
    failsafe_triggered = False

    response_msg = cl.Message(content="")

    async for event in agent.on_messages_stream(
        messages=[agent_message],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(event, ModelClientStreamingChunkEvent):
            if event.content:
                await response_msg.stream_token(event.content)
                response_text += event.content

        elif isinstance(event, ToolCallRequestEvent):
            for call in event.content:
                await cl.Message(content=f"Using tool: **{call.name}**").send()

        elif isinstance(event, ToolCallExecutionEvent):
            for result in event.content:
                if result.is_error:
                    await cl.Message(content=f"Tool error: {result.content}").send()
                else:
                    # Check for FAILSAFE
                    if "FAILSAFE" in result.content:
                        failsafe_triggered = True

                    # Check for mouse movement
                    if "Moved mouse" in result.content:
                        used_mouse = True

                    # Check if result is a captured image path
                    try:
                        result_path = Path(result.content)
                        if result_path.exists():
                            suffix = result_path.suffix.lower()
                            if suffix in ['.jpg', '.jpeg', '.png']:
                                img_element = cl.Image(
                                    path=str(result_path),
                                    name="captured_image",
                                    display="inline"
                                )
                                await cl.Message(
                                    content="📷 Captured:",
                                    elements=[img_element]
                                ).send()
                                captured_image_path = result_path
                    except Exception:
                        pass

        elif isinstance(event, Response):
            if response_msg.content:
                await response_msg.send()
            else:
                final_content = event.chat_message.content
                if final_content:
                    response_text = final_content
                    await cl.Message(content=final_content).send()

    return response_text, captured_image_path, used_mouse, failsafe_triggered


async def run_embodied_turn(
    agent: AssistantAgent,
    agent_message,
    profiler: Optional[EmbodiedProfiler] = None,
    status_msg: Optional[cl.Message] = None,
) -> tuple:
    """Run a single embodied control turn.

    Args:
        agent: The AutoGen assistant agent.
        agent_message: The multimodal message to send.
        profiler: Optional profiler for timing measurements.
        status_msg: Optional persistent message to update in-place (embodied mode).
            When provided, updates this message instead of creating new ones.

    Returns:
        tuple: (response_text, used_mouse, tool_called, failsafe_triggered)
    """
    response_text = ""
    used_mouse = False
    tool_called = False
    failsafe_triggered = False
    first_token_received = False

    response_msg = cl.Message(content="")

    async for event in agent.on_messages_stream(
        messages=[agent_message],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(event, ModelClientStreamingChunkEvent):
            if event.content:
                # Track time to first token
                if not first_token_received:
                    if profiler:
                        profiler.mark("inference_ttft")
                    first_token_received = True
                response_text += event.content
                if status_msg is None:
                    await response_msg.stream_token(event.content)

        elif isinstance(event, ToolCallRequestEvent):
            tool_called = True
            if profiler:
                profiler.mark("tool_request")
            for call in event.content:
                if status_msg is None:
                    await cl.Message(content=f"🔧 {call.name}").send()

        elif isinstance(event, ToolCallExecutionEvent):
            if profiler:
                profiler.mark("tool_executed")
            for result in event.content:
                if result.is_error:
                    if status_msg is not None:
                        status_msg.content = f"Tool error: {result.content}"
                        await status_msg.update()
                    else:
                        await cl.Message(content=f"Tool error: {result.content}").send()
                else:
                    if "FAILSAFE" in result.content:
                        failsafe_triggered = True
                    if "Moved mouse" in result.content:
                        used_mouse = True
                        if status_msg is not None:
                            status_msg.content = f"🖱️ {result.content}"
                            await status_msg.update()
                        else:
                            await cl.Message(content=f"🖱️ {result.content}").send()

        elif isinstance(event, Response):
            if status_msg is not None:
                final_content = event.chat_message.content
                if isinstance(final_content, str) and final_content:
                    response_text = final_content
                status_msg.content = response_text or status_msg.content
                await status_msg.update()
            else:
                if response_msg.content:
                    await response_msg.send()
                else:
                    final_content = event.chat_message.content
                    if final_content:
                        response_text = final_content
                        await cl.Message(content=final_content).send()

    return response_text, used_mouse, tool_called, failsafe_triggered


async def run_inference_slot(
    agent: AssistantAgent,
    message,
) -> tuple:
    """Run one inference slot silently (no UI interactions).

    Used by the pipelined embodied loop. The main loop handles all UI
    updates after each completion to avoid race conditions.

    Args:
        agent: The AutoGen assistant agent for this slot.
        message: The multimodal message to send.

    Returns:
        tuple: (response_text, used_mouse, tool_called, failsafe)
    """
    response_text = ""
    used_mouse = False
    tool_called = False
    failsafe = False

    async for event in agent.on_messages_stream(
        messages=[message],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(event, ModelClientStreamingChunkEvent):
            if event.content:
                response_text += event.content
        elif isinstance(event, ToolCallRequestEvent):
            tool_called = True
        elif isinstance(event, ToolCallExecutionEvent):
            for result in event.content:
                if not result.is_error:
                    if "FAILSAFE" in result.content:
                        failsafe = True
                    if "Moved mouse" in result.content:
                        used_mouse = True
        elif isinstance(event, Response):
            final = event.chat_message.content
            if isinstance(final, str) and final:
                response_text = final

    return response_text, used_mouse, tool_called, failsafe


async def run_embodied_loop(instruction: str) -> int:
    """Run the app-driven embodied control loop.

    The app captures images directly and sends them to the model.
    The model only needs to analyze and call mouse_move_tool.

    Creates a dedicated embodied agent with BufferedChatCompletionContext
    to limit history to ~2 frames, preventing inference degradation.

    Args:
        instruction: The user's original instruction with stop condition.

    Returns:
        Number of iterations completed.
    """
    await cl.Message(content="**Starting embodied control loop...**").send()

    # Initialize profiler with model and image settings
    profiler = EmbodiedProfiler(
        model=MODEL_NAME,
        image_settings={
            "max_width": IMAGE_MAX_WIDTH,
            "max_height": IMAGE_MAX_HEIGHT,
            "jpeg_quality": IMAGE_JPEG_QUALITY,
        },
    )

    # Create fresh embodied agent with bounded context, reduced num_ctx and output limit
    model_client = create_model_client(
        num_ctx=EMBODIED_CONTEXT_SIZE,
        num_predict=EMBODIED_NUM_PREDICT,
    )
    agent = create_embodied_agent(model_client)

    iterations_completed = 0

    # Open capture ONCE before loop (avoids 3+ second RTSP reconnection per frame)
    cap = get_video_capture()
    if not cap.isOpened():
        await cl.Message(content="**Failed to open webcam.**").send()
        return 0

    # Single temp directory for all frames (reuse path each iteration)
    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / "frame.jpg"

    try:
        # Sidebar for live frame display (replaces elements properly each call)
        await cl.ElementSidebar.set_title("Live Camera Feed")

        # Single persistent status message — updates in-place (no scroll)
        status_msg = cl.Message(content="Starting embodied loop...")
        await status_msg.send()

        # Prefetch first frame before loop starts
        next_frame_bytes = capture_frame_from_cap(
            cap,
            max_width=IMAGE_MAX_WIDTH,
            max_height=IMAGE_MAX_HEIGHT,
            jpeg_quality=IMAGE_JPEG_QUALITY,
        )

        for iteration in range(EMBODIED_MAX_ITERATIONS):
            profiler.start_iteration(iteration)

            # Check if user cancelled
            if not cl.user_session.get("embodied_mode", True):
                profiler.end_iteration()
                status_msg.content = "**Cancelled by user.**"
                await status_msg.update()
                break

            # 1. Use prefetched frame (captured during previous iteration's inference)
            profiler.mark("capture_start")
            image_bytes = next_frame_bytes
            profiler.mark("capture_end")

            if image_bytes is None:
                profiler.end_iteration()
                status_msg.content = "**Failed to capture image from webcam.**"
                await status_msg.update()
                break

            # Save captured image to temp file (overwrite previous)
            profiler.mark("file_write_start")
            temp_path.write_bytes(image_bytes)
            profiler.mark("file_write_end")

            # Display captured image in sidebar + update status (parallel)
            profiler.mark("ui_display_start")
            img_element = cl.Image(path=str(temp_path), name=f"frame_{iteration}", display="inline")
            status_msg.content = f"Iteration {iteration + 1}: Analyzing frame..."
            await asyncio.gather(
                cl.ElementSidebar.set_elements([img_element], key=f"frame_{iteration}"),
                status_msg.update(),
            )
            profiler.mark("ui_display_end")

            # 2. Clear context and build fresh message with instruction + current image
            await agent.model_context.clear()
            profiler.mark("msg_build_start")
            prompt = f"""{instruction}

Analyze this image. What direction should I move the mouse?
- If stop condition is met: respond with STOP and explain why
- If stop condition NOT met: call mouse_move_tool with the direction (up/down/left/right/up-left/up-right/down-left/down-right)"""

            # Convert bytes to PIL Image for AGImage
            pil_image = PILImage.open(io.BytesIO(image_bytes))
            message_content = [prompt, AGImage(pil_image)]
            agent_message = MultiModalMessage(content=message_content, source="user")
            profiler.mark("msg_build_end")

            # 3. Get model response + prefetch next frame in parallel
            profiler.mark("inference_start")
            loop = asyncio.get_event_loop()
            next_frame_task = loop.run_in_executor(
                None, capture_frame_from_cap, cap,
                IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT, IMAGE_JPEG_QUALITY,
            )
            response_text, used_mouse, tool_called, failsafe = await run_embodied_turn(
                agent, agent_message, profiler, status_msg=status_msg
            )
            # Await prefetched frame (already captured during inference)
            next_frame_bytes = await next_frame_task
            profiler.mark("inference_end")

            # 4. Check stop conditions
            profiler.mark("stop_check_start")
            should_stop = False
            if failsafe:
                status_msg.content = "**FAILSAFE triggered - stopping.**"
                await status_msg.update()
                should_stop = True
            elif detect_stop_condition(response_text):
                status_msg.content = f"**Embodied control stopped** (after {iteration + 1} iterations)"
                await status_msg.update()
                should_stop = True
            elif not used_mouse and not tool_called:
                # Model genuinely chose not to call any tool - decided to stop
                status_msg.content = f"**Stopped** (no mouse movement after {iteration + 1} iterations)"
                await status_msg.update()
                should_stop = True
            profiler.mark("stop_check_end")

            if should_stop:
                iterations_completed = iteration + 1
                profiler.end_iteration()
                break

            profiler.end_iteration()
            iterations_completed = iteration + 1

        else:
            # Loop completed without break (max iterations reached)
            status_msg.content = f"**Reached max iterations ({EMBODIED_MAX_ITERATIONS})**"
            await status_msg.update()
            iterations_completed = EMBODIED_MAX_ITERATIONS

    finally:
        cap.release()  # Always release the capture
        await cl.ElementSidebar.set_elements([])  # Close sidebar

    # Display timing summary in UI
    if profiler.iterations:
        summary_table = profiler.format_ui_table()
        log_path = profiler.save_to_file()
        await cl.Message(content=f"{summary_table}\n\nSaved to: `{log_path}`").send()

    return iterations_completed


async def run_pipelined_embodied_loop(instruction: str) -> int:
    """Run pipelined embodied control with multiple concurrent inference slots.

    Keeps N inference requests in flight at all times (staggered). When one
    completes, its result is processed (mouse move) and a fresh frame is
    submitted to that slot. Moves arrive ~1.7x more frequently than sequential.

    Requires OLLAMA_NUM_PARALLEL >= slots on the server.

    Args:
        instruction: The user's original instruction with stop condition.

    Returns:
        Number of moves completed.
    """
    config = load_model_config()
    pipeline_cfg = config.get("pipeline", {})
    num_slots = pipeline_cfg.get("slots", 2)
    move_distance = pipeline_cfg.get("move_distance", PIPELINE_MOVE_DISTANCE)

    await cl.Message(
        content=f"**Starting pipelined embodied control** ({num_slots} slots, {move_distance}px moves)"
    ).send()

    # Create separate model_client + agent per slot (avoids _tool_id race)
    agents = []
    for i in range(num_slots):
        client = create_model_client(
            num_ctx=EMBODIED_CONTEXT_SIZE,
            num_predict=EMBODIED_NUM_PREDICT,
        )
        agent = create_embodied_agent(client, move_distance=move_distance)
        agents.append(agent)

    # Open webcam once
    cap = get_video_capture()
    if not cap.isOpened():
        await cl.Message(content="**Failed to open webcam.**").send()
        return 0

    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / "frame.jpg"

    # Build the prompt template
    prompt = f"""{instruction}

Analyze this image. What direction should I move the mouse?
- If stop condition is met: respond with STOP and explain why
- If stop condition NOT met: call _mouse_move with the direction (up/down/left/right/up-left/up-right/down-left/down-right)"""

    total_completed = 0
    total_submitted = 0
    should_stop = False
    pending: set = set()
    task_meta: dict = {}  # task -> {"agent_idx": int, "frame_bytes": bytes}
    completion_times: list[float] = []
    pipeline_start = time.perf_counter()

    try:
        await cl.ElementSidebar.set_title("Live Camera Feed (Pipeline)")
        status_msg = cl.Message(content="Starting pipeline...")
        await status_msg.send()

        # Submit initial tasks with stagger
        for i in range(min(num_slots, EMBODIED_MAX_ITERATIONS)):
            frame_bytes = capture_frame_from_cap(
                cap, IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT, IMAGE_JPEG_QUALITY,
            )
            if frame_bytes is None:
                break

            pil_image = PILImage.open(io.BytesIO(frame_bytes))
            message = MultiModalMessage(
                content=[prompt, AGImage(pil_image)], source="user",
            )

            agent = agents[i]
            await agent.model_context.clear()
            task = asyncio.create_task(run_inference_slot(agent, message))
            pending.add(task)
            task_meta[task] = {"agent_idx": i, "frame_bytes": frame_bytes}
            total_submitted += 1

            # Stagger initial submissions so requests don't hit GPU simultaneously
            if i < num_slots - 1:
                await asyncio.sleep(PIPELINE_INITIAL_STAGGER_S)

        # Main pipeline loop
        while pending and not should_stop:
            # Check user cancel
            if not cl.user_session.get("embodied_mode", True):
                should_stop = True
                for t in pending:
                    t.cancel()
                status_msg.content = "**Cancelled by user.**"
                await status_msg.update()
                break

            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                try:
                    response_text, used_mouse, tool_called, failsafe = task.result()
                except asyncio.CancelledError:
                    continue
                except Exception as e:
                    # Inference error — log and continue with remaining slots
                    status_msg.content = f"Slot error: {e}"
                    await status_msg.update()
                    continue

                total_completed += 1
                t_now = time.perf_counter()
                completion_times.append(t_now)

                # Update UI with the frame used for this inference
                meta = task_meta.pop(task, {})
                frame_bytes = meta.get("frame_bytes")
                if frame_bytes:
                    temp_path.write_bytes(frame_bytes)
                    img_element = cl.Image(
                        path=str(temp_path),
                        name=f"frame_{total_completed}",
                        display="inline",
                    )

                    # Show interval in status
                    interval_str = ""
                    if len(completion_times) >= 2:
                        ivl = (completion_times[-1] - completion_times[-2]) * 1000
                        interval_str = f" | {ivl:.0f}ms"

                    status_msg.content = (
                        f"[{total_completed}] {response_text[:60]}{interval_str}"
                    )
                    await asyncio.gather(
                        cl.ElementSidebar.set_elements(
                            [img_element], key=f"pframe_{total_completed}",
                        ),
                        status_msg.update(),
                    )

                # Check stop conditions
                if failsafe:
                    should_stop = True
                    status_msg.content = "**FAILSAFE triggered — stopping.**"
                    await status_msg.update()
                elif detect_stop_condition(response_text):
                    should_stop = True
                    status_msg.content = (
                        f"**Stopped** (after {total_completed} moves)"
                    )
                    await status_msg.update()
                elif not used_mouse and not tool_called:
                    should_stop = True
                    status_msg.content = (
                        f"**Stopped** (no move after {total_completed} iterations)"
                    )
                    await status_msg.update()

                if should_stop:
                    for t in pending:
                        t.cancel()
                    break

                # Submit replacement task to the same agent slot
                if total_submitted < EMBODIED_MAX_ITERATIONS:
                    agent_idx = meta.get("agent_idx", 0)
                    agent = agents[agent_idx]

                    frame_bytes = capture_frame_from_cap(
                        cap, IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT, IMAGE_JPEG_QUALITY,
                    )
                    if frame_bytes is None:
                        should_stop = True
                        break

                    pil_image = PILImage.open(io.BytesIO(frame_bytes))
                    message = MultiModalMessage(
                        content=[prompt, AGImage(pil_image)], source="user",
                    )

                    await agent.model_context.clear()
                    new_task = asyncio.create_task(
                        run_inference_slot(agent, message)
                    )
                    pending.add(new_task)
                    task_meta[new_task] = {
                        "agent_idx": agent_idx,
                        "frame_bytes": frame_bytes,
                    }
                    total_submitted += 1

    finally:
        cap.release()
        await cl.ElementSidebar.set_elements([])

    # Display pipeline summary
    if len(completion_times) >= 2:
        intervals = [
            (completion_times[i] - completion_times[i - 1]) * 1000
            for i in range(1, len(completion_times))
        ]
        avg_interval = sum(intervals) / len(intervals)
        total_time = (completion_times[-1] - pipeline_start) * 1000
        await cl.Message(
            content=(
                f"### Pipeline Summary ({num_slots} slots, {move_distance}px)\n"
                f"- Moves: {total_completed}\n"
                f"- Avg interval: {avg_interval:.0f}ms\n"
                f"- Min/Max: {min(intervals):.0f}ms / {max(intervals):.0f}ms\n"
                f"- Total: {total_time:.0f}ms\n"
                f"- Effective freq: {1000 / avg_interval:.1f} moves/s"
            )
        ).send()
    elif total_completed > 0:
        await cl.Message(
            content=f"Pipeline completed {total_completed} move(s)."
        ).send()

    return total_completed


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming messages from the user."""
    agent = cast(AssistantAgent, cl.user_session.get("agent"))

    if agent is None:
        await cl.Message(content="Error: Agent not initialized. Please refresh.").send()
        return

    # Check if user wants to stop embodied mode
    if message.content and message.content.lower().strip() in ["stop", "cancel", "abort"]:
        cl.user_session.set("embodied_mode", False)
        await cl.Message(content="Stopped.").send()
        return

    # Check for image attachments
    images = [el for el in message.elements if el.mime and "image" in el.mime]
    videos = [el for el in message.elements if el.mime and "video" in el.mime]

    # Build the message for the agent
    if images or videos:
        content = []
        if message.content:
            content.append(message.content)

        for img in images:
            try:
                ag_image = AGImage.from_file(Path(img.path))
                content.append(ag_image)
            except Exception as e:
                await cl.Message(content=f"Error loading image {img.name}: {e}").send()

        for vid in videos:
            try:
                frames = extract_video_frames(vid.path, VIDEO_FRAMES_PER_SECOND, VIDEO_MAX_FRAMES)
                if frames:
                    content.append(f"[Video: {vid.name} - {len(frames)} frames extracted]")
                    for frame in frames:
                        content.append(AGImage(frame))
            except Exception as e:
                await cl.Message(content=f"Error processing video {vid.name}: {e}").send()

        agent_message = MultiModalMessage(content=content, source="user") if content else TextMessage(content=message.content or "Analyze this", source="user")
    else:
        agent_message = TextMessage(content=message.content, source="user")

    # Detect embodied control mode
    embodied_mode = detect_embodied_mode(message.content or "")

    try:
        if embodied_mode:
            # Use app-driven embodied loop with dedicated bounded-context agent
            cl.user_session.set("embodied_mode", True)
            pipeline_cfg = load_model_config().get("pipeline", {})
            if pipeline_cfg.get("enabled", False):
                await run_pipelined_embodied_loop(message.content)
            else:
                await run_embodied_loop(message.content)
            cl.user_session.set("embodied_mode", False)
            return

        # Normal mode - run agent turn
        response_text, captured_image, used_mouse, failsafe = await run_agent_turn(
            agent, agent_message, embodied_mode=False
        )

        # If media was captured, send for analysis
        if captured_image:
            analysis_content = [
                "Describe what you see in this image in detail:",
                AGImage.from_file(captured_image)
            ]
            analysis_msg = MultiModalMessage(content=analysis_content, source="user")

            analysis_response = cl.Message(content="")
            async for event in agent.on_messages_stream(
                messages=[analysis_msg],
                cancellation_token=CancellationToken(),
            ):
                if isinstance(event, ModelClientStreamingChunkEvent):
                    if event.content:
                        await analysis_response.stream_token(event.content)
                elif isinstance(event, Response):
                    if analysis_response.content:
                        await analysis_response.send()
                    elif event.chat_message.content:
                        await cl.Message(content=event.chat_message.content).send()

    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()
        import traceback
        traceback.print_exc()
        cl.user_session.set("embodied_mode", False)


@cl.on_stop
async def on_stop():
    """Handle chat stop/interrupt."""
    await cl.Message(content="Chat stopped.").send()


# For running directly (debugging)
if __name__ == "__main__":
    print("This is a Chainlit app. Run with:")
    print("  chainlit run app.py")
    print("\nOr for development with auto-reload:")
    print("  chainlit run app.py -w")

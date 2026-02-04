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
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_ext.models.ollama import OllamaChatCompletionClient

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
MODEL_NAME = "qwen3-vl:30b-a3b"  # Model name (set in model_config.yaml)
CONTEXT_SIZE = 262144  # 256K context

# Video processing defaults (can be overridden in model_config.yaml)
VIDEO_FRAMES_PER_SECOND = 5.0
VIDEO_MAX_FRAMES = 50

# Image capture defaults for embodied control (can be overridden in model_config.yaml)
IMAGE_MAX_WIDTH = 640
IMAGE_MAX_HEIGHT = 480
IMAGE_JPEG_QUALITY = 70

# Embodied control settings
EMBODIED_MAX_ITERATIONS = 50  # Safety limit
EMBODIED_KEYWORDS = ["until", "keep going", "keep taking", "continuously", "feedback loop"]
STOP_INDICATORS = ["stop", "stopped", "stopping", "fist", "closed fist", "abort", "done", "finished"]

# System prompt for embodied control mode (concise for fast inference)
EMBODIED_SYSTEM_PROMPT = """You are an embodied AI assistant controlling a mouse cursor based on visual input.

Your job:
1. Analyze the image to detect hand gesture or pointing direction
2. If STOP condition is met (e.g., fist closed) → respond with "STOP" and explain why
3. If stop condition NOT met → call mouse_move_tool with the detected direction

Keep responses brief. Just state what you see and call the tool.
Example: "Thumb pointing right." Then call mouse_move_tool(direction="right", distance=50)

SAFETY: Moving the mouse to any screen corner will trigger FAILSAFE and abort."""


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


def create_model_client() -> OllamaChatCompletionClient:
    """Create and configure the Ollama model client.

    Returns:
        Configured OllamaChatCompletionClient.
    """
    config = load_model_config()

    return OllamaChatCompletionClient(
        model=config.get("model", MODEL_NAME),
        host=config.get("host", SERVER_BASE_URL),
        options={
            "temperature": config.get("options", {}).get("temperature", 0.7),
            "num_ctx": config.get("options", {}).get("num_ctx", CONTEXT_SIZE),
        },
    )


def create_embodied_agent(model_client) -> AssistantAgent:
    """Create agent for embodied control with bounded context.

    Uses BufferedChatCompletionContext to keep only the last ~2 frames
    of context, preventing inference degradation from context growth.

    Args:
        model_client: The model client to use.

    Returns:
        AssistantAgent configured for embodied control.
    """
    # Buffer size calculation:
    # Each frame generates ~4 messages:
    #   1. UserMessage (with image)
    #   2. AssistantMessage (tool call request)
    #   3. FunctionExecutionResultMessage
    #   4. AssistantMessage (final response)
    # For 2 frames = 8 messages, use buffer_size=10 for safety margin
    model_context = BufferedChatCompletionContext(buffer_size=10)

    return AssistantAgent(
        name="embodied_assistant",
        model_client=model_client,
        model_context=model_context,  # Bounded context!
        tools=[mouse_move_tool, mouse_position_tool],
        system_message=EMBODIED_SYSTEM_PROMPT,
        model_client_stream=True,
        reflect_on_tool_use=False,
    )


# Wrapper function for the webcam tool that shows in Chainlit
@cl.step(type="tool")
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


@cl.step(type="tool")
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


@cl.step(type="tool")
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
) -> tuple:
    """Run a single embodied control turn.

    Args:
        agent: The AutoGen assistant agent.
        agent_message: The multimodal message to send.
        profiler: Optional profiler for timing measurements.

    Returns:
        tuple: (response_text, used_mouse, failsafe_triggered)
    """
    response_text = ""
    used_mouse = False
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
                await response_msg.stream_token(event.content)
                response_text += event.content

        elif isinstance(event, ToolCallRequestEvent):
            if profiler:
                profiler.mark("tool_request")
            for call in event.content:
                await cl.Message(content=f"🔧 {call.name}").send()

        elif isinstance(event, ToolCallExecutionEvent):
            if profiler:
                profiler.mark("tool_executed")
            for result in event.content:
                if result.is_error:
                    await cl.Message(content=f"Tool error: {result.content}").send()
                else:
                    if "FAILSAFE" in result.content:
                        failsafe_triggered = True
                    if "Moved mouse" in result.content:
                        used_mouse = True
                        await cl.Message(content=f"🖱️ {result.content}").send()

        elif isinstance(event, Response):
            if response_msg.content:
                await response_msg.send()
            else:
                final_content = event.chat_message.content
                if final_content:
                    response_text = final_content
                    await cl.Message(content=final_content).send()

    return response_text, used_mouse, failsafe_triggered


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

    # Create fresh embodied agent with bounded context
    # This ensures no history carryover and limits context to ~2 frames
    model_client = create_model_client()
    agent = create_embodied_agent(model_client)

    iterations_completed = 0

    # Open capture ONCE before loop (avoids 3+ second RTSP reconnection per frame)
    cap = get_video_capture()
    if not cap.isOpened():
        await cl.Message(content="**Failed to open webcam.**").send()
        return 0

    try:
        for iteration in range(EMBODIED_MAX_ITERATIONS):
            profiler.start_iteration(iteration)

            # Check if user cancelled
            if not cl.user_session.get("embodied_mode", True):
                profiler.end_iteration()
                await cl.Message(content="**Cancelled by user.**").send()
                break

            # 1. APP captures image from persistent connection (fast)
            profiler.mark("capture_start")
            image_bytes = capture_frame_from_cap(
                cap,
                max_width=IMAGE_MAX_WIDTH,
                max_height=IMAGE_MAX_HEIGHT,
                jpeg_quality=IMAGE_JPEG_QUALITY
            )
            profiler.mark("capture_end")

            if image_bytes is None:
                profiler.end_iteration()
                await cl.Message(content="**Failed to capture image from webcam.**").send()
                break

            # Save captured image to temp file
            profiler.mark("file_write_start")
            temp_dir = Path(tempfile.mkdtemp())
            temp_path = temp_dir / f"embodied_{int(time.time())}_{iteration}.jpg"
            temp_path.write_bytes(image_bytes)
            profiler.mark("file_write_end")

            # Display captured image in UI
            profiler.mark("ui_display_start")
            img_element = cl.Image(path=str(temp_path), name="frame", display="inline")
            await cl.Message(content=f"Frame {iteration + 1}:", elements=[img_element]).send()
            profiler.mark("ui_display_end")

            # 2. Build message with actual image data
            profiler.mark("msg_build_start")
            if iteration == 0:
                prompt = f"""{instruction}

Analyze this image. What direction should I move the mouse?
- If stop condition is met: respond with STOP and explain why
- If stop condition NOT met: call mouse_move_tool with the direction (up/down/left/right)"""
            else:
                prompt = """Continue. Analyze this image:
- If stop condition is met: respond with STOP and explain why
- If stop condition NOT met: call mouse_move_tool with the direction"""

            # Convert bytes to PIL Image for AGImage
            pil_image = PILImage.open(io.BytesIO(image_bytes))
            message_content = [prompt, AGImage(pil_image)]
            agent_message = MultiModalMessage(content=message_content, source="user")
            profiler.mark("msg_build_end")

            # 3. Get model response
            profiler.mark("inference_start")
            response_text, used_mouse, failsafe = await run_embodied_turn(
                agent, agent_message, profiler
            )
            profiler.mark("inference_end")

            # 4. Check stop conditions
            profiler.mark("stop_check_start")
            should_stop = False
            if failsafe:
                await cl.Message(content="**FAILSAFE triggered - stopping.**").send()
                should_stop = True
            elif detect_stop_condition(response_text):
                await cl.Message(content=f"**Embodied control stopped** (after {iteration + 1} iterations)").send()
                should_stop = True
            elif not used_mouse:
                # Model didn't call mouse_move_tool - probably decided to stop
                await cl.Message(content=f"**Stopped** (no mouse movement after {iteration + 1} iterations)").send()
                should_stop = True
            profiler.mark("stop_check_end")

            if should_stop:
                iterations_completed = iteration + 1
                profiler.end_iteration()
                break

            # 5. Small delay before next capture
            profiler.mark("delay_start")
            await asyncio.sleep(0.3)
            profiler.mark("delay_end")

            profiler.end_iteration()
            iterations_completed = iteration + 1

        else:
            # Loop completed without break (max iterations reached)
            await cl.Message(content=f"**Reached max iterations ({EMBODIED_MAX_ITERATIONS})**").send()
            iterations_completed = EMBODIED_MAX_ITERATIONS

    finally:
        cap.release()  # Always release the capture

    # Display timing summary in UI
    if profiler.iterations:
        summary_table = profiler.format_ui_table()
        log_path = profiler.save_to_file()
        await cl.Message(content=f"{summary_table}\n\nSaved to: `{log_path}`").send()

    return iterations_completed


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

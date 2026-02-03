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
from autogen_ext.models.ollama import OllamaChatCompletionClient

from tools.webcam import capture_webcam, extract_video_frames
from tools.mouse import move_mouse, get_mouse_position, get_screen_size

# Configuration
OLLAMA_HOST = "http://localhost:11435"
MODEL_NAME = "qwen3-vl:235b"
CONTEXT_SIZE = 262144  # 256K context

# Video processing defaults (can be overridden in model_config.yaml)
VIDEO_FRAMES_PER_SECOND = 5.0
VIDEO_MAX_FRAMES = 50

# Embodied control settings
EMBODIED_MAX_ITERATIONS = 50  # Safety limit
EMBODIED_KEYWORDS = ["until", "keep going", "keep taking", "continuously", "feedback loop"]
STOP_INDICATORS = ["stop", "stopped", "stopping", "fist", "closed fist", "abort", "done", "finished"]


def load_model_config() -> dict:
    """Load model configuration from YAML file.

    Also sets global video processing settings if defined.

    Returns:
        Configuration dictionary.
    """
    global VIDEO_FRAMES_PER_SECOND, VIDEO_MAX_FRAMES

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
        host=config.get("host", OLLAMA_HOST),
        model_info=ModelInfo(
            vision=True,
            function_calling=True,
            json_output=False,
            family="unknown",
            structured_output=False
        ),
        options={
            "temperature": config.get("options", {}).get("temperature", 0.7),
            "num_ctx": config.get("options", {}).get("num_ctx", CONTEXT_SIZE),
        }
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

When the user asks you to control the mouse based on visual input (e.g., "move mouse where I'm pointing"):

IMPORTANT: You must continue the feedback loop AUTONOMOUSLY without waiting for user input between iterations. Keep calling tools in sequence until the stop condition is met.

Loop steps:
1. Capture an image using webcam_capture_tool
2. Briefly describe what you see (1 sentence)
3. If stop condition is met → STOP and explain why
4. If stop condition is NOT met → call mouse_move_tool in the detected direction
5. Immediately go back to step 1 (capture another image)

DO NOT pause or wait for user confirmation between iterations. Keep the loop going until:
- The stop condition is observed (e.g., closed fist, hand removed)
- The user explicitly says "stop" in a new message
- FAILSAFE is triggered (mouse moved to screen corner)

Example stop conditions:
- "until I close my fist" → stop when you see a closed fist
- "until I say stop" → stop only when user sends a new "stop" message
- "until you don't see my hand" → stop when hand is not visible

Keep descriptions brief during the loop. Move in small increments (50-100px).

SAFETY: Moving the mouse to any screen corner will trigger FAILSAFE and abort all mouse operations.

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
    if embodied_mode:
        cl.user_session.set("embodied_mode", True)
        cl.user_session.set("embodied_instruction", message.content)

    try:
        # Run first turn
        response_text, captured_image, used_mouse, failsafe = await run_agent_turn(
            agent, agent_message, embodied_mode
        )

        # If in embodied mode, continue the loop
        if embodied_mode and captured_image and used_mouse and not failsafe:
            iteration = 1

            while iteration < EMBODIED_MAX_ITERATIONS:
                # Check if model indicated stop
                if detect_stop_condition(response_text):
                    await cl.Message(content=f"**Embodied control stopped** (detected stop condition after {iteration} iterations)").send()
                    break

                iteration += 1

                # Build continuation message with the captured image
                continuation_content = [
                    AGImage.from_file(captured_image),
                    "Continue: analyze this image and take the next action. Remember the instruction and stop condition."
                ]
                continuation_msg = MultiModalMessage(content=continuation_content, source="user")

                # Run next turn
                response_text, captured_image, used_mouse, failsafe = await run_agent_turn(
                    agent, continuation_msg, embodied_mode=True
                )

                # Check stop conditions
                if failsafe:
                    await cl.Message(content="**FAILSAFE triggered** - embodied control stopped.").send()
                    break

                if not captured_image or not used_mouse:
                    # Model didn't capture or move - might have decided to stop
                    if detect_stop_condition(response_text):
                        await cl.Message(content=f"**Embodied control stopped** (after {iteration} iterations)").send()
                    break

                # Check if user session was manually stopped
                if not cl.user_session.get("embodied_mode", False):
                    await cl.Message(content="**Embodied control cancelled by user.**").send()
                    break

            if iteration >= EMBODIED_MAX_ITERATIONS:
                await cl.Message(content=f"**Embodied control stopped** (reached max {EMBODIED_MAX_ITERATIONS} iterations)").send()

            cl.user_session.set("embodied_mode", False)

        # If not embodied mode but media was captured, send for analysis
        elif captured_image and not embodied_mode:
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

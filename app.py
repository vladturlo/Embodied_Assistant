"""Multimodal AutoGen-Chainlit Agent.

This is the main application that integrates AutoGen with Chainlit
to provide a multimodal AI agent with webcam capabilities.

Features:
- Image/video upload and analysis
- Webcam capture via tool
- Live Vision mode: continuous camera streaming

Usage:
    chainlit run app.py
"""

# Windows asyncio compatibility fix - must be before other imports
import sys
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import asyncio
import tempfile
import time
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
from tools.live_capture import LiveCaptureService, frames_to_images

# Configuration
OLLAMA_HOST = "http://localhost:11435"
MODEL_NAME = "qwen3-vl:235b"
CONTEXT_SIZE = 262144  # 256K context

# Video processing defaults (can be overridden in model_config.yaml)
VIDEO_FRAMES_PER_SECOND = 5.0
VIDEO_MAX_FRAMES = 50

# Live Vision defaults (can be overridden in model_config.yaml)
LIVE_VISION_ENABLED = False
LIVE_CAPTURE_FPS = 2.0
LIVE_BUFFER_SECONDS = 5.0
LIVE_MAX_FRAMES_PER_MESSAGE = 10
LIVE_PREVIEW_FPS = 2.0
LIVE_INACTIVITY_TIMEOUT = 300
LIVE_ADD_TIMESTAMP = True


def load_model_config() -> dict:
    """Load model configuration from YAML file.

    Also sets global video and live_vision processing settings if defined.

    Returns:
        Configuration dictionary.
    """
    global VIDEO_FRAMES_PER_SECOND, VIDEO_MAX_FRAMES
    global LIVE_VISION_ENABLED, LIVE_CAPTURE_FPS, LIVE_BUFFER_SECONDS
    global LIVE_MAX_FRAMES_PER_MESSAGE, LIVE_PREVIEW_FPS
    global LIVE_INACTIVITY_TIMEOUT, LIVE_ADD_TIMESTAMP

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

        # Load live_vision settings if present
        live_config = config.get("live_vision", {})
        if "enabled" in live_config:
            LIVE_VISION_ENABLED = bool(live_config["enabled"])
        if "capture_fps" in live_config:
            LIVE_CAPTURE_FPS = float(live_config["capture_fps"])
        if "buffer_seconds" in live_config:
            LIVE_BUFFER_SECONDS = float(live_config["buffer_seconds"])
        if "max_frames_per_message" in live_config:
            LIVE_MAX_FRAMES_PER_MESSAGE = int(live_config["max_frames_per_message"])
        if "preview_fps" in live_config:
            LIVE_PREVIEW_FPS = float(live_config["preview_fps"])
        if "inactivity_timeout" in live_config:
            LIVE_INACTIVITY_TIMEOUT = int(live_config["inactivity_timeout"])
        if "add_timestamp_overlay" in live_config:
            LIVE_ADD_TIMESTAMP = bool(live_config["add_timestamp_overlay"])

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
async def webcam_capture_tool(mode: str = "image", duration: float = 3.0) -> str:
    """Capture image or video from webcam.

    This tool captures media from the webcam and displays it in the chat.
    The captured file path is returned for the vision model to analyze.

    Args:
        mode: Either "image" for single frame or "video" for short clip.
        duration: Duration in seconds for video capture (1-10).

    Returns:
        Path to the captured file or error message.
    """
    result = await capture_webcam(mode=mode, duration=duration)
    return result


async def start_live_vision() -> tuple[bool, str]:
    """Start the live vision capture service.

    Returns:
        Tuple of (success, message).
    """
    capture_service = cl.user_session.get("capture_service")

    if capture_service and capture_service.is_running:
        return True, "Live Vision is already running"

    # Create new service if needed
    if not capture_service:
        capture_service = LiveCaptureService(
            capture_fps=LIVE_CAPTURE_FPS,
            buffer_seconds=LIVE_BUFFER_SECONDS,
            max_buffer_frames=int(LIVE_BUFFER_SECONDS * LIVE_CAPTURE_FPS * 2),
            add_timestamp_overlay=LIVE_ADD_TIMESTAMP,
        )
        capture_service.inactivity_timeout = LIVE_INACTIVITY_TIMEOUT
        cl.user_session.set("capture_service", capture_service)

    # Start capture
    if capture_service.start():
        cl.user_session.set("live_mode", True)

        # Start preview update task
        preview_task = asyncio.create_task(update_live_preview())
        cl.user_session.set("preview_task", preview_task)

        return True, "Live Vision started - I can now see your camera continuously"
    else:
        error = capture_service.error_message or "Unknown error"
        return False, f"Failed to start Live Vision: {error}"


async def stop_live_vision() -> tuple[bool, str]:
    """Stop the live vision capture service.

    Returns:
        Tuple of (success, message).
    """
    capture_service = cl.user_session.get("capture_service")

    if not capture_service or not capture_service.is_running:
        cl.user_session.set("live_mode", False)
        return True, "Live Vision is not running"

    # Cancel preview task
    preview_task = cl.user_session.get("preview_task")
    if preview_task:
        preview_task.cancel()
        try:
            await preview_task
        except asyncio.CancelledError:
            pass
        cl.user_session.set("preview_task", None)

    # Stop capture
    capture_service.stop()
    cl.user_session.set("live_mode", False)

    return True, "Live Vision stopped"


async def update_live_preview():
    """Background task to update the live preview in the UI."""
    preview_interval = 1.0 / LIVE_PREVIEW_FPS
    temp_dir = Path(tempfile.mkdtemp())
    preview_path = temp_dir / "live_preview.jpg"
    preview_msg: Optional[cl.Message] = None

    try:
        while True:
            capture_service = cl.user_session.get("capture_service")
            if not capture_service or not capture_service.is_running:
                break

            frame = capture_service.get_latest_frame()
            if frame:
                # Save frame as JPEG
                frame.image.save(str(preview_path), "JPEG", quality=80)

                # Create or update preview message
                if preview_msg is None:
                    preview_element = cl.Image(
                        path=str(preview_path),
                        name="live_preview",
                        display="inline",
                        size="medium"
                    )
                    preview_msg = cl.Message(
                        content="**Live Preview** (updating...)",
                        elements=[preview_element]
                    )
                    await preview_msg.send()
                else:
                    # Update existing preview by sending new message
                    # (Chainlit doesn't support true in-place image updates)
                    try:
                        await preview_msg.remove()
                    except Exception:
                        pass

                    preview_element = cl.Image(
                        path=str(preview_path),
                        name="live_preview",
                        display="inline",
                        size="medium"
                    )
                    preview_msg = cl.Message(
                        content="**Live Preview** (updating...)",
                        elements=[preview_element]
                    )
                    await preview_msg.send()

            await asyncio.sleep(preview_interval)

    except asyncio.CancelledError:
        # Clean up preview message on cancel
        if preview_msg:
            try:
                await preview_msg.remove()
            except Exception:
                pass
        raise


async def take_snapshot() -> tuple[bool, str, Optional[str]]:
    """Take a snapshot from the live feed and save it.

    Returns:
        Tuple of (success, message, file_path or None).
    """
    capture_service = cl.user_session.get("capture_service")

    if not capture_service or not capture_service.is_running:
        return False, "Live Vision is not running. Start it first with /live on", None

    result = capture_service.take_snapshot()
    if result:
        image, timestamp = result
        # Save to temp file
        temp_dir = Path(tempfile.mkdtemp())
        snapshot_path = temp_dir / f"snapshot_{timestamp.replace(':', '-')}.jpg"
        image.save(str(snapshot_path), "JPEG", quality=95)
        return True, f"Snapshot taken at {timestamp}", str(snapshot_path)
    else:
        return False, "No frame available", None


async def handle_live_command(args: str) -> None:
    """Handle /live command variants.

    Args:
        args: Command arguments (on, off, status, snapshot).
    """
    args = args.strip().lower()

    if args == "on" or args == "start":
        success, message = await start_live_vision()
        if success:
            await cl.Message(content=f"**Live Vision ON** - {message}").send()
        else:
            await cl.Message(content=f"**Error:** {message}").send()

    elif args == "off" or args == "stop":
        success, message = await stop_live_vision()
        await cl.Message(content=f"**Live Vision OFF** - {message}").send()

    elif args == "snapshot" or args == "snap":
        success, message, path = await take_snapshot()
        if success and path:
            img_element = cl.Image(path=path, name="snapshot", display="inline")
            await cl.Message(content=f"**Snapshot:** {message}", elements=[img_element]).send()
        else:
            await cl.Message(content=f"**Snapshot Error:** {message}").send()

    elif args == "" or args == "status":
        live_mode = cl.user_session.get("live_mode", False)
        capture_service = cl.user_session.get("capture_service")

        if live_mode and capture_service and capture_service.is_running:
            frame_count = capture_service.frame_count
            status = f"""**Live Vision Status: ON**
- Frames in buffer: {frame_count}
- Capture FPS: {LIVE_CAPTURE_FPS}
- Buffer duration: {LIVE_BUFFER_SECONDS}s
- Max frames per message: {LIVE_MAX_FRAMES_PER_MESSAGE}

Commands:
- `/live off` - Stop live vision
- `/live snapshot` - Take a snapshot"""
        else:
            status = """**Live Vision Status: OFF**

Commands:
- `/live on` - Start live vision (camera streams continuously)
- When ON, I'll see your camera with every message"""

        await cl.Message(content=status).send()

    else:
        await cl.Message(content=f"""**Unknown command:** /live {args}

Available commands:
- `/live on` - Start live vision
- `/live off` - Stop live vision
- `/live status` - Show current status
- `/live snapshot` - Take a snapshot from live feed""").send()


@cl.set_starters
async def set_starters() -> List[cl.Starter]:
    """Set starter prompts for the chat interface.

    Returns:
        List of starter prompts.
    """
    return [
        cl.Starter(
            label="Live Vision",
            message="/live on",
        ),
        cl.Starter(
            label="Webcam Capture",
            message="Take a picture with the webcam and tell me what you see.",
        ),
        cl.Starter(
            label="Image Analysis",
            message="I'll upload an image for you to analyze.",
        ),
        cl.Starter(
            label="Video Capture",
            message="Record a short video from the webcam and describe what happens.",
        ),
    ]


@cl.on_chat_start
async def start_chat() -> None:
    """Initialize the chat session with the multimodal agent."""
    # Create model client
    model_client = create_model_client()

    # Create the assistant agent with webcam tool
    assistant = AssistantAgent(
        name="multimodal_assistant",
        model_client=model_client,
        tools=[webcam_capture_tool],
        system_message="""You are a helpful multimodal AI assistant with vision capabilities.

You can:
1. Analyze images that users upload
2. Capture images or short videos from the webcam using the webcam_capture_tool
3. Describe what you see in images and videos
4. Answer questions about visual content

When users ask you to "look", "see", "capture", or use the webcam, use the webcam_capture_tool.
After capturing, analyze the image and describe what you see.

IMPORTANT: When Live Vision mode is active, you can see the user's camera continuously.
Recent frames will be included with each message. Describe what you see naturally without
needing to use any tools - you already have the visual context.

Be helpful, accurate, and descriptive in your visual analysis.""",
        model_client_stream=True,
        reflect_on_tool_use=False,  # Disabled: we handle image analysis explicitly after tool execution
    )

    # Store agent in session
    cl.user_session.set("agent", assistant)

    # Initialize live vision state
    cl.user_session.set("live_mode", False)
    cl.user_session.set("capture_service", None)
    cl.user_session.set("preview_task", None)

    # Send welcome message
    await cl.Message(
        content="Hello! I'm a multimodal AI assistant. I can:\n"
                "- Analyze images you upload\n"
                "- Capture images/videos from your webcam\n"
                "- Answer questions about visual content\n\n"
                "**New: Live Vision Mode**\n"
                "- `/live on` - Start continuous camera streaming\n"
                "- `/live off` - Stop streaming\n"
                "- `/live snapshot` - Take a snapshot\n\n"
                "Try uploading an image or type `/live on` to start!"
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming messages from the user.

    Args:
        message: The incoming Chainlit message.
    """
    # Check for /live command
    content = message.content.strip()
    if content.startswith("/live"):
        args = content[5:].strip()  # Remove "/live" prefix
        await handle_live_command(args)
        return

    agent = cast(AssistantAgent, cl.user_session.get("agent"))

    if agent is None:
        await cl.Message(content="Error: Agent not initialized. Please refresh.").send()
        return

    # Check for image/video attachments
    images = [el for el in message.elements if el.mime and "image" in el.mime]
    videos = [el for el in message.elements if el.mime and "video" in el.mime]

    # Check if live mode is active
    live_mode = cl.user_session.get("live_mode", False)
    capture_service = cl.user_session.get("capture_service")

    # Build the message for the agent
    content_parts = []

    # Add live vision frames if active
    if live_mode and capture_service and capture_service.is_running:
        recent_frames = capture_service.get_recent_frames(
            seconds=LIVE_BUFFER_SECONDS,
            max_count=LIVE_MAX_FRAMES_PER_MESSAGE
        )
        if recent_frames:
            content_parts.append(f"[Live Vision: {len(recent_frames)} recent frames from camera]")
            for frame in recent_frames:
                content_parts.append(AGImage(frame.image))

    # Add uploaded images
    for img in images:
        try:
            ag_image = AGImage.from_file(Path(img.path))
            content_parts.append(ag_image)
        except Exception as e:
            await cl.Message(content=f"Error loading image {img.name}: {e}").send()

    # Handle uploaded videos - extract frames for analysis
    for vid in videos:
        try:
            frames = extract_video_frames(vid.path, VIDEO_FRAMES_PER_SECOND, VIDEO_MAX_FRAMES)
            if frames:
                content_parts.append(f"[Video: {vid.name} - {len(frames)} frames extracted]")
                for frame in frames:
                    content_parts.append(AGImage(frame))
            else:
                content_parts.append(f"[Video: {vid.name} - could not extract frames]")
        except Exception as e:
            await cl.Message(content=f"Error processing video {vid.name}: {e}").send()

    # Add text content
    if message.content:
        content_parts.append(message.content)

    # Create the agent message
    if len(content_parts) > 1 or any(isinstance(p, AGImage) for p in content_parts):
        # Multimodal message with images/frames
        agent_message = MultiModalMessage(content=content_parts, source="user")
    else:
        # Text-only message
        agent_message = TextMessage(content=message.content or "Hello", source="user")

    # Create response message for streaming
    response_msg = cl.Message(content="")
    captured_images = []  # Track captured images for follow-up analysis

    try:
        # Stream the response
        async for event in agent.on_messages_stream(
            messages=[agent_message],
            cancellation_token=CancellationToken(),
        ):
            if isinstance(event, ModelClientStreamingChunkEvent):
                # Stream text chunks
                if event.content:
                    await response_msg.stream_token(event.content)

            elif isinstance(event, ToolCallRequestEvent):
                # Tool is being called
                for call in event.content:
                    await cl.Message(
                        content=f"Using tool: **{call.name}**"
                    ).send()

            elif isinstance(event, ToolCallExecutionEvent):
                # Tool execution completed
                for result in event.content:
                    if result.is_error:
                        await cl.Message(
                            content=f"Tool error: {result.content}"
                        ).send()
                    else:
                        # Check if result is a captured media path
                        result_path = Path(result.content)
                        if result_path.exists():
                            suffix = result_path.suffix.lower()
                            if suffix in ['.jpg', '.jpeg', '.png']:
                                # Display captured image in chat
                                img_element = cl.Image(
                                    path=str(result_path),
                                    name="captured_image",
                                    display="inline"
                                )
                                await cl.Message(
                                    content="Captured from webcam:",
                                    elements=[img_element]
                                ).send()
                                # Store for follow-up analysis
                                captured_images.append(("image", result_path))
                            elif suffix in ['.mp4', '.avi', '.mov']:
                                # Video is already displayed by webcam tool
                                # Store for follow-up analysis
                                captured_images.append(("video", result_path))

            elif isinstance(event, Response):
                # Final response
                if response_msg.content:
                    await response_msg.send()
                else:
                    # If no streaming content, send the final response
                    final_content = event.chat_message.content
                    if final_content:
                        await cl.Message(content=final_content).send()

        # If media was captured via tool (not live mode), send to model for analysis
        if captured_images and not live_mode:
            # Build multimodal message with captured media
            analysis_content = []
            has_video = False

            for media_type, media_path in captured_images:
                if media_type == "image":
                    analysis_content.append(AGImage.from_file(media_path))
                elif media_type == "video":
                    has_video = True
                    # Extract frames from video for vision analysis
                    frames = extract_video_frames(str(media_path), VIDEO_FRAMES_PER_SECOND, VIDEO_MAX_FRAMES)
                    if frames:
                        analysis_content.append(f"[Video with {len(frames)} frames]")
                        for frame in frames:
                            analysis_content.append(AGImage(frame))

            if analysis_content:
                # Add prompt based on content type
                if has_video:
                    prompt = "Describe what you see happening in this video (shown as frames):"
                else:
                    prompt = "Describe what you see in this image in detail:"
                analysis_content.insert(0, prompt)

                analysis_msg = MultiModalMessage(content=analysis_content, source="user")

                # Get analysis from model and stream response
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


@cl.on_chat_end
async def on_chat_end():
    """Clean up when chat session ends."""
    # Stop live vision if running
    await stop_live_vision()


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

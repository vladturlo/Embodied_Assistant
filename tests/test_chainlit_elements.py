"""Test Chainlit image/video display elements.

This is a Chainlit app for testing image upload and display functionality.
Run with: chainlit run tests/test_chainlit_elements.py

Usage:
    cd projects/multimodal_agent
    chainlit run tests/test_chainlit_elements.py
"""

import tempfile
from pathlib import Path

import chainlit as cl
from PIL import Image as PILImage


@cl.on_chat_start
async def start():
    """Initialize the test chat."""
    await cl.Message(
        content="Chainlit Elements Test Ready!\n\n"
                "This test app verifies:\n"
                "1. File upload handling\n"
                "2. Image display\n"
                "3. Video display (if uploaded)\n\n"
                "Try these:\n"
                "- Upload an image to see it echoed back\n"
                "- Type 'test' to see a generated test image\n"
                "- Type 'info' to see upload configuration"
    ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    """Handle incoming messages and file uploads."""

    # Check for test commands
    if msg.content.lower() == "test":
        await handle_test_image()
        return

    if msg.content.lower() == "info":
        await handle_info()
        return

    # Check for file uploads
    if msg.elements:
        await handle_uploads(msg)
    else:
        await cl.Message(
            content=f"Received text: {msg.content}\n\n"
                    "Upload an image to test file handling, "
                    "or type 'test' for a generated image."
        ).send()


async def handle_uploads(msg: cl.Message):
    """Handle uploaded files."""
    await cl.Message(content=f"Received {len(msg.elements)} file(s):").send()

    for el in msg.elements:
        # Show file info
        info = f"**File:** {el.name}\n" \
               f"**MIME:** {el.mime}\n" \
               f"**Path:** {el.path}"

        await cl.Message(content=info).send()

        # Handle images
        if el.mime and "image" in el.mime:
            await handle_image_upload(el)

        # Handle videos
        elif el.mime and "video" in el.mime:
            await handle_video_upload(el)

        else:
            await cl.Message(
                content=f"File type '{el.mime}' is not an image or video."
            ).send()


async def handle_image_upload(el):
    """Echo back an uploaded image."""
    try:
        # Display the image back
        image_element = cl.Image(
            path=el.path,
            name=f"echo_{el.name}",
            display="inline"
        )

        await cl.Message(
            content="Here's your image back:",
            elements=[image_element]
        ).send()

        # Show image details
        pil_img = PILImage.open(el.path)
        details = f"**Image Details:**\n" \
                  f"- Size: {pil_img.size}\n" \
                  f"- Mode: {pil_img.mode}\n" \
                  f"- Format: {pil_img.format}"

        await cl.Message(content=details).send()

    except Exception as e:
        await cl.Message(content=f"Error processing image: {e}").send()


async def handle_video_upload(el):
    """Echo back an uploaded video."""
    try:
        video_element = cl.Video(
            path=el.path,
            name=f"echo_{el.name}"
        )

        await cl.Message(
            content="Here's your video back:",
            elements=[video_element]
        ).send()

    except Exception as e:
        await cl.Message(content=f"Error processing video: {e}").send()


async def handle_test_image():
    """Generate and display a test image."""
    # Create a colorful test image
    width, height = 300, 200
    img = PILImage.new('RGB', (width, height))

    # Create a gradient pattern
    for x in range(width):
        for y in range(height):
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = int(255 * (1 - x / width))
            img.putpixel((x, y), (r, g, b))

    # Save to temp file
    temp_path = Path(tempfile.mkdtemp()) / "test_gradient.png"
    img.save(str(temp_path))

    # Display
    image_element = cl.Image(
        path=str(temp_path),
        name="test_gradient",
        display="inline"
    )

    await cl.Message(
        content="Generated test image (gradient pattern):",
        elements=[image_element]
    ).send()


async def handle_info():
    """Show upload configuration info."""
    info = """
**Chainlit Upload Configuration:**

The `.chainlit/config.toml` file should have:
```toml
[features.spontaneous_file_upload]
enabled = true
accept = ["image/jpeg", "image/png", "image/gif", "image/webp", "video/mp4", "video/webm"]
max_files = 10
max_size_mb = 100
```

**Supported Elements:**
- `cl.Image(path=..., name=...)` - Display images
- `cl.Video(path=..., name=...)` - Display videos
- `cl.File(path=..., name=...)` - Generic file download

**Message Elements:**
- `msg.elements` - List of uploaded files
- `el.path` - File path on disk
- `el.mime` - MIME type
- `el.name` - Original filename
"""
    await cl.Message(content=info).send()


# For direct testing without chainlit run
if __name__ == "__main__":
    print("This is a Chainlit app. Run with:")
    print("  chainlit run tests/test_chainlit_elements.py")
    print("\nOr from the project root:")
    print("  cd projects/multimodal_agent")
    print("  chainlit run tests/test_chainlit_elements.py")

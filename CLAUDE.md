# Multimodal AutoGen-Chainlit Agent

## Purpose
Multimodal AI agent with webcam interaction capabilities using AutoGen + Chainlit.
Enables real-time visual interaction through webcam capture and image/video analysis.

## Architecture
- **Frontend**: Chainlit web UI with file upload and media display
- **Backend**: AutoGen AssistantAgent with vision model
- **Model**: qwen3-vl:235b via Ollama (localhost:11435)
- **Tools**: Webcam capture (image/video)
- **Context**: 256K tokens (maximum)

## Initial Setup (One-time)
```bash
# Install python3-venv if not available (requires sudo)
sudo apt install python3.12-venv python3-pip

# Create virtual environment
cd /home/tuvl/projects/multimodal_agent
python3 -m venv .venv

# Activate and install dependencies
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the Application
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the Chainlit app
chainlit run app.py
```

## SSH Tunnel Setup (Remote Ollama Access)

The multimodal agent connects to Ollama running on CSCS compute nodes via SSH tunnel.

### Prerequisites

1. **SSH keys** in `~/.ssh/`:
   - `cscs-key` (private key)
   - `cscs-key-cert.pub` (certificate - CSCS uses certificates)

   **Copy keys from Windows Downloads folder:**
   ```bash
   cp /mnt/c/Users/tuvl/Downloads/cscs-key ~/.ssh/
   cp /mnt/c/Users/tuvl/Downloads/cscs-key-cert.pub ~/.ssh/
   chmod 600 ~/.ssh/cscs-key
   chmod 644 ~/.ssh/cscs-key-cert.pub
   ```

2. **SSH config** in `~/.ssh/config`:
   ```
   Host ela
       HostName ela.cscs.ch
       User vturlo
       IdentityFile ~/.ssh/cscs-key

   Host daint
       HostName daint.alps.cscs.ch
       User vturlo
       ProxyJump ela
       IdentityFile ~/.ssh/cscs-key
       IdentitiesOnly yes

   Host nid*
       User vturlo
       ProxyJump daint
       IdentityFile ~/.ssh/cscs-key
       IdentitiesOnly yes
   ```

### Establishing the Tunnel

**Step 1: Start ssh-agent and add keys**
```bash
eval $(ssh-agent -s)
ssh-add ~/.ssh/cscs-key
```

**Step 2: Connect with port forwarding**
```bash
# Current compute node: nid005376
# Update NODE_ID if Ollama moves to a different node
ssh -A -N -L 11435:localhost:11434 nid005376
```

Leave this terminal open. The tunnel forwards:
- Local port 11435 → Compute node port 11434 (Ollama)

**Step 3: Verify tunnel** (in another terminal)
```bash
curl http://localhost:11435/api/tags
```

### Running the App with Tunnel
```bash
cd /home/tuvl/projects/multimodal_agent
source .venv/bin/activate

# Verify connection
python tests/test_ollama_connection.py

# Run the app
chainlit run app.py --host 0.0.0.0 --port 8000
```

### Current Configuration
- **Model**: qwen3-vl:235b
- **Node**: nid005376
- **Local Port**: 11435
- **Remote Port**: 11434

## GitHub Repository
https://github.com/vladturlo/Embodied_Assistant

## Project Structure
```
multimodal_agent/
├── CLAUDE.md              # This file
├── .gitignore             # Git ignore patterns
├── pyproject.toml         # Python dependencies
├── .chainlit/config.toml  # Chainlit configuration
├── model_config.yaml      # Ollama model settings
├── tools/
│   ├── __init__.py
│   └── webcam.py          # Webcam capture tool (supports RTSP, cross-platform ffmpeg)
├── scripts/               # Utility scripts
│   ├── test_rtsp_connection.py   # Test RTSP from WSL2
│   ├── start_webcam_stream.ps1   # Windows RTSP streaming script
│   ├── windows_setup.bat         # Windows one-time setup
│   └── windows_run.bat           # Windows run script (pulls & runs)
├── tests/                 # Component tests (run before integration)
│   ├── test_ollama_connection.py
│   ├── test_vision_model.py
│   ├── test_webcam_capture.py
│   ├── test_video_frames.py
│   ├── test_multimodal_message.py
│   ├── test_chainlit_elements.py
│   └── test_agent_tools.py
├── test_assets/           # Test images/videos
└── app.py                 # Main application
```

## Key Files
- **app.py** - Main agent application with Chainlit integration
- **tools/webcam.py** - Webcam capture tool (image and video)
- **model_config.yaml** - Ollama model configuration

## Testing (Run in Order)
```bash
# 1. Test Ollama connection
python tests/test_ollama_connection.py

# 2. Test vision model
python tests/test_vision_model.py

# 3. Test webcam capture
python tests/test_webcam_capture.py

# 4. Test video frame extraction
python tests/test_video_frames.py

# 5. Test MultiModalMessage
python tests/test_multimodal_message.py

# 6. Test Chainlit elements
chainlit run tests/test_chainlit_elements.py

# 7. Test agent tools
python tests/test_agent_tools.py
```

## Model Configuration
- **Model**: qwen3-vl:235b
- **Host**: http://localhost:11435
- **Context Window**: 262144 tokens (256K)
- **Vision**: Enabled
- **Function Calling**: Enabled

## Webcam Tool
The agent can capture images and short videos from the webcam:
- **Image mode**: Single frame capture
- **Video mode**: Short clip (up to 10 seconds)
- Captured media is displayed in chat for both user and agent to see

### WSL2 Webcam Support (RTSP Streaming)
WSL2 cannot access Windows webcam directly. Use RTSP streaming instead:

**Windows Setup (one-time):**
1. Install ffmpeg: https://www.gyan.dev/ffmpeg/builds/
2. Install MediaMTX: https://github.com/bluenviron/mediamtx/releases

**Start Streaming (Windows PowerShell):**
```powershell
# Find your camera name
ffmpeg -list_devices true -f dshow -i dummy

# Start MediaMTX (PowerShell #1)
cd C:\mediamtx
.\mediamtx.exe

# Start webcam stream (PowerShell #2)
ffmpeg -f dshow -i video="Integrated Camera" -framerate 30 -video_size 640x480 -vcodec libx264 -preset ultrafast -tune zerolatency -f rtsp rtsp://localhost:8554/webcam
```

**Test from WSL2:**
```bash
python scripts/test_rtsp_connection.py
```

**Use in Application:**
```bash
# Get Windows host IP
ip route list default | awk '{print $3}'

# Set environment variable (replace IP)
export WEBCAM_RTSP_URL="rtsp://172.25.192.1:8554/webcam"

# Run tests
python tests/test_webcam_capture.py

# Or run the app
chainlit run app.py
```

## Dependencies
- autogen-agentchat >= 0.4
- autogen-ext[ollama] >= 0.4
- chainlit >= 1.0
- opencv-python >= 4.9
- pillow >= 10.0
- numpy >= 1.26
- ffmpeg (system) - for H.264 video encoding

## Windows Setup (Native)

For running directly on Windows (without WSL2):

**Prerequisites:**
1. Python 3.10+ installed
2. ffmpeg in PATH or at `C:\ffmpeg\bin\`
3. Git installed
4. Ollama running locally or SSH tunnel to remote

**One-time setup:**
```powershell
git clone https://github.com/vladturlo/Embodied_Assistant.git
cd Embodied_Assistant
scripts\windows_setup.bat
```

**Run the app:**
```powershell
scripts\windows_run.bat
```

The `windows_run.bat` script automatically pulls latest changes from GitHub before running.

## Development Workflow

**Develop in WSL2:**
```bash
cd /home/tuvl/projects/multimodal_agent
source .venv/bin/activate
# Make changes, test
chainlit run app.py --host 0.0.0.0 --port 8000

# Commit and push
git add -A && git commit -m "Description" && git push
```

**Test on Windows:**
Double-click `scripts\windows_run.bat` or run from PowerShell.

## Debugging Notes

### Image Analysis Fix (2026-01-28)
**Problem**: Model was hallucinating image descriptions instead of analyzing actual images.

**Root Cause**: With `reflect_on_tool_use=True`, AutoGen's tool system converts all results to strings via `return_value_as_string()`. The model's reflection pass only received the file path text, not actual image data.

**Fix implemented in app.py**:
1. Set `reflect_on_tool_use=False` - disables automatic reflection that caused hallucination
2. Track captured media (images AND videos) during `ToolCallExecutionEvent` as tuples (type, path)
3. Display media inline with `cl.Image(display="inline")`
4. After tool execution, send `MultiModalMessage` with actual image data via `AGImage.from_file()`
5. For videos: extract frames using `extract_video_frames()` and send as multiple `AGImage` objects

**Status**: Fixed. The model now receives actual image data and provides accurate descriptions.

### Video Playback Fix (2026-01-28)
**Problem**: Captured videos not playable in browser.

**Root Cause**: OpenCV's `mp4v` codec (MPEG-4 Part 2) is not browser-compatible. Browsers require H.264.

**Fix implemented in tools/webcam.py**:
1. After OpenCV saves video, re-encode with ffmpeg to H.264
2. Added `_get_ffmpeg_path()` for cross-platform ffmpeg detection
3. Fixed FPS to use actual capture device rate instead of hardcoded 15fps

**Status**: Fixed. Videos now play in Chainlit UI.

### Quick Debug Commands
```bash
# Test RTSP connection
export WEBCAM_RTSP_URL="rtsp://$(ip route list default | awk '{print $3}'):8554/webcam"
python scripts/test_rtsp_connection.py

# Test Ollama connection
python tests/test_ollama_connection.py

# Run app with verbose output
chainlit run app.py --host 0.0.0.0 --port 8000 -w
```

## Notes
- Qwen3-VL supports native video understanding with timestamp alignment
- Large model (235B parameters) - responses may take time
- Ensure SSH tunnel is active before running (see SSH Tunnel Setup above)

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
│   └── webcam.py          # Webcam capture tool (supports RTSP)
├── scripts/               # Utility scripts
│   ├── test_rtsp_connection.py   # Test RTSP from WSL2
│   └── start_webcam_stream.ps1   # Windows streaming script
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

## Notes
- Qwen3-VL supports native video understanding with timestamp alignment
- Large model (235B parameters) - responses may take time
- Ensure SSH tunnel is active before running (see SSH Tunnel Setup above)

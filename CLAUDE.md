# Multimodal AutoGen-Chainlit Agent

## Purpose
Multimodal AI agent with webcam interaction capabilities using AutoGen + Chainlit.
Enables real-time visual interaction through webcam capture and image/video analysis.

## Architecture
- **Frontend**: Chainlit web UI with file upload and media display
- **Backend**: AutoGen AssistantAgent with vision model
- **Model**: qwen3-vl:238b via Ollama (localhost:11435)
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

## Ollama Setup
Ollama API accessible at localhost:11435 (SSH tunnel from remote server).
Ensure tunnel is active before running:
```bash
# Example SSH tunnel command (run on local machine)
ssh -L 11435:localhost:11434 user@remote-server
```

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
│   └── webcam.py          # Webcam capture tool
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
- **Model**: qwen3-vl:238b
- **Host**: http://localhost:11435
- **Context Window**: 262144 tokens (256K)
- **Vision**: Enabled
- **Function Calling**: Enabled

## Webcam Tool
The agent can capture images and short videos from the webcam:
- **Image mode**: Single frame capture
- **Video mode**: Short clip (up to 10 seconds)
- Captured media is displayed in chat for both user and agent to see

## Dependencies
- autogen-agentchat >= 0.4
- autogen-ext[ollama] >= 0.4
- chainlit >= 1.0
- opencv-python >= 4.9
- pillow >= 10.0
- numpy >= 1.26

## Notes
- Qwen3-VL supports native video understanding with timestamp alignment
- Large model (238B parameters) - responses may take time
- Ensure SSH tunnel is active before running

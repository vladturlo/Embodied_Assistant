# Multimodal Agent - Windows Setup Guide

This guide walks you through installing all required software to run the Multimodal Agent on Windows.

## Prerequisites

- Windows 10 or 11 (64-bit)
- NVIDIA GPU (recommended for running local models efficiently)
- ~50GB free disk space (for software + AI models)

---

## Step 1: Install Python 3.12

### Option A: Official Installer (Recommended)

1. Download from [python.org/downloads/windows](https://www.python.org/downloads/windows/)
2. Run the installer
3. **IMPORTANT**: Check the "Add Python to PATH" checkbox at the bottom
4. Click "Install Now"

### Option B: Microsoft Store

- Open Microsoft Store, search "Python 3.12", and click Install

### Verify Installation

```cmd
python --version
```
Expected output: `Python 3.12.x`

---

## Step 2: Install Git

1. Download from [git-scm.com/download/win](https://git-scm.com/download/win)
2. Run the installer and accept the default options
3. Verify installation:

```cmd
git --version
```
Expected output: `git version 2.52.x`

---

## Step 3: Install FFmpeg

FFmpeg is required for video processing and encoding.

1. Download the "Essentials" build from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/)
   - Look for `ffmpeg-release-essentials.zip` or `.7z`
2. Extract the archive to `C:\ffmpeg`
   - You should have `C:\ffmpeg\bin\ffmpeg.exe` after extraction
3. Add FFmpeg to your system PATH:
   - Press `Win+R`, type `sysdm.cpl`, press Enter
   - Click the **Advanced** tab
   - Click **Environment Variables**
   - Under "System variables", find and select **Path**, then click **Edit**
   - Click **New** and add: `C:\ffmpeg\bin`
   - Click **OK** to close all dialogs
4. **Restart your Command Prompt** and verify:

```cmd
ffmpeg -version
```
Expected output: `ffmpeg version 7.1.x ...`

---

## Step 4: Install Ollama

Ollama runs AI models locally on your machine.

1. Download from [ollama.com/download/windows](https://ollama.com/download/windows)
2. Run `OllamaSetup.exe` (no administrator rights required)
3. Verify installation:

```cmd
ollama --version
```

4. Pull the vision model used by this project:

```cmd
ollama pull qwen3-vl:235b
```

> **Note**: Large models like `qwen3-vl:235b` require significant disk space (50-100GB+) and a powerful GPU. For testing, you can use smaller models like `llava` or `qwen2-vl:7b`.

---

## Step 5: Install CUDA Toolkit (Recommended for GPU)

> Skip this step if you don't have an NVIDIA GPU

CUDA enables GPU acceleration for running AI models faster.

1. Download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Select: **Windows** → **x86_64** → **Your Windows version** → **exe (local)**
2. Run the installer
   - **Express installation** is recommended for most users
3. Restart your computer
4. Verify installation:

```cmd
nvcc --version
```
Expected output: `nvcc: NVIDIA (R) Cuda compiler driver ...`

---

## Step 6: Clone and Run the Project

### Option A: Using Command Line

```cmd
git clone https://github.com/vladturlo/Embodied_Assistant.git
cd Embodied_Assistant
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
chainlit run app.py
```

### Option B: Using Provided Scripts

```cmd
scripts\windows_setup.bat   # First-time setup (creates venv, installs dependencies)
scripts\windows_run.bat     # Run the application (auto-pulls latest from GitHub)
```

The application will be available at: **http://localhost:8000**

---

## Verification Checklist

Run these commands to verify all components are installed correctly:

```cmd
python --version      # Should show Python 3.12.x
git --version         # Should show git version 2.x
ffmpeg -version       # Should show ffmpeg version 7.x
ollama --version      # Should show ollama version
nvcc --version        # Should show CUDA version (if installed)
```

---

## Troubleshooting

### "ffmpeg" is not recognized as a command

- Ensure `C:\ffmpeg\bin` is added to your system PATH (not user PATH)
- Restart your Command Prompt after modifying PATH
- Verify the file exists: `dir C:\ffmpeg\bin\ffmpeg.exe`

### Ollama models run slowly

- Install CUDA Toolkit if you have an NVIDIA GPU
- Update your NVIDIA drivers to the latest version
- Ensure Ollama is using GPU: check Task Manager → Performance → GPU

### "python" is not recognized as a command

- Reinstall Python and check the "Add Python to PATH" checkbox
- Or manually add Python to PATH:
  - Find your Python installation (usually `C:\Users\<username>\AppData\Local\Programs\Python\Python312\`)
  - Add both the main folder and the `Scripts` subfolder to PATH

### CUDA installation fails

- Ensure your NVIDIA drivers are up to date
- Download drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers)
- Some antivirus software may interfere with installation - temporarily disable if needed

---

## Useful Links

- **Python**: https://www.python.org/downloads/windows/
- **Git**: https://git-scm.com/download/win
- **FFmpeg**: https://www.gyan.dev/ffmpeg/builds/
- **Ollama**: https://ollama.com/download/windows
- **Ollama Documentation**: https://docs.ollama.com/windows
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads
- **CUDA Installation Guide**: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/

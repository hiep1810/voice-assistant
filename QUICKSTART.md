# Voice Assistant - Quick Start Guide

## Quick Start

### Option 1: Interactive Setup Wizard (Recommended)

Run the setup wizard to detect your hardware and install components step-by-step:

```bash
uv run python -m voice_assistant setup
```

The wizard will:
1. **Detect your system** - Platform, GPU, RAM
2. **Recommend components** based on hardware:
   - **NVIDIA GPU**: CUDA acceleration for LLM/TTS
   - **Apple Silicon**: Metal acceleration
   - **CPU Only**: Lightweight models, remote server option
3. **Select components** to install:
   - VAD (Voice Activity Detection)
   - ASR (Speech-to-Text)
   - LLM (Local or Remote Server)
   - TTS (Text-to-Speech)
   - Audio I/O (Microphone/Speaker)
   - TUI (Terminal Interface)
   - Vision (VLM for image analysis)
4. **Install packages** with progress display
5. **Generate config file** optimized for your hardware
6. **Show next steps** for running

---

### Option 2: Manual Setup

#### For NVIDIA GPU Users (Windows/Linux)

```bash
# Full installation with CUDA support
uv pip install -e ".[all]"

# Run voice assistant
uv run python -m voice_assistant run
```

#### For macOS (Apple Silicon) Users

```bash
# Full installation with Metal support
uv pip install -e ".[all]"

# Run voice assistant
uv run python -m voice_assistant run
```

#### For CPU Only (No GPU)

```bash
# Install minimal dependencies
uv pip install llama-cpp-python huggingface_hub typer rich

# Run in text-only mode
uv run python -m voice_assistant cli -m qwen3-0.6b
```

#### Using Remote llama.cpp Server

```bash
# 1. Start llama.cpp server (separate terminal)
llama-server.exe -m qwen3-2b-q4_k_m.gguf --port 8000

# 2. Connect voice assistant (lightweight client)
uv pip install requests typer rich
uv run python -m voice_assistant cli --server-url http://localhost:8000
```

---

## Component Installation

### Individual Components

| Component | Packages | For |
|-----------|----------|-----|
| **VAD** | `silero-vad` | Voice activity detection |
| **ASR** | `sherpa-onnx` | Speech-to-text (Vietnamese) |
| **LLM (Local)** | `llama-cpp-python`, `huggingface_hub` | Local AI responses |
| **LLM (Server)** | `requests` | Remote server connection |
| **TTS** | `vieneu` | Vietnamese voice synthesis |
| **Audio** | `pyaudio`, `sounddevice`, `numpy` | Mic/speaker I/O |
| **TUI** | `rich`, `typer` | Terminal interface |
| **Vision** | `opencv-python`, `pillow` | Image/screen analysis |

### Install Individual Component

```bash
# Example: Install only ASR for transcription testing
uv pip install sherpa-onnx

# Example: Install only TTS for synthesis testing
uv pip install vieneu
```

---

## Available Commands

| Command | Description |
|---------|-------------|
| `uv run python -m voice_assistant setup` | Interactive setup wizard |
| `uv run python -m voice_assistant cli` | Text-only CLI mode |
| `uv run python -m voice_assistant run` | Full TUI with live display |
| `uv run python -m voice_assistant list-models` | List available models |

---

## CLI Options

```bash
# Use remote server
uv run python -m voice_assistant cli --server-url http://localhost:8000

# Specify model
uv run python -m voice_assistant cli -m qwen3-0.6b

# Disable TTS (text-only responses)
uv run python -m voice_assistant cli --no-tts

# Load from config file
uv run python -m voice_assistant cli -c config.json
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"
```bash
uv pip install numpy
# Or run full setup:
uv run python -m voice_assistant setup
```

### "Could not connect to llama.cpp server"
- Ensure server is running: `llama-server.exe -m model.gguf --port 8000`
- Check firewall settings
- Try `http://127.0.0.1:8000` instead of `localhost`

### "Silero VAD not installed"
```bash
uv pip install silero-vad
```

### "llama_cpp not found"
```bash
uv pip install llama-cpp-python
```

### "CUDA out of memory"
Reduce GPU layers in config:
```json
{
  "llm": {
    "n_gpu_layers": 10
  }
}
```

---

## Model Information

### LLM Models (Auto-downloaded)

| Model | Size | Context | Vietnamese | Tool Calling |
|-------|------|---------|------------|--------------|
| qwen3-2b | ~2GB (Q4) | 32K | Yes | Yes |
| qwen3-0.6b | ~500MB (Q4) | 32K | Yes | Yes |
| lfm2-1.6b | ~1GB (Q4) | 8K | No | Yes |

### TTS Voices (VieNeu-TTS)

| Voice | Description |
|-------|-------------|
| neutrale | Neutral (default) |
| hanhphuc | Happy tone |
| leloi | Historical voice |
| nguyentruothanh | Poetic tone |
| chihanh | Gentle voice |
| khanhlinh | Clear voice |

---

## Hardware Recommendations

### NVIDIA GPU (RTX 3060 or better)
- Install: `.[all]` for full CUDA acceleration
- LLM: Use 30-35 GPU layers
- TTS: GPU acceleration enabled
- Expected RTF: < 0.2 for TTS, ~50 tokens/s for LLM

### Apple Silicon (M1/M2/M3)
- Install: `.[all]` for Metal acceleration
- LLM: Use 30-35 GPU layers
- TTS: Standard backend (vieneu)
- Expected RTF: < 0.3 for TTS, ~30 tokens/s for LLM

### CPU Only
- Install: `llama-cpp-python`, minimal deps
- LLM: Use smaller model (qwen3-0.6b) or remote server
- TTS: Consider server mode
- Expected: Slower but functional for testing

---

## Config File Example

Create `config.json`:

```json
{
  "llm": {
    "model": "qwen3-2b",
    "n_gpu_layers": 35,
    "server_url": null
  },
  "tts": {
    "model": "vietneu-tts",
    "backend": "lmdeploy",
    "use_gpu": true,
    "speaker": "neutrale"
  },
  "vad": {
    "onset": 0.5,
    "offset": 0.5
  },
  "asr": {
    "model": "zipformer",
    "language": "vi"
  }
}
```

Load with: `uv run python -m voice_assistant cli -c config.json`

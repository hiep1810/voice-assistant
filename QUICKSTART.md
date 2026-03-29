# Voice Assistant - Quick Start Guide

## Quick Start

### Option 1: Use Remote llama.cpp Server (Recommended for testing)

1. **Start llama.cpp server** (in a separate terminal):
   ```bash
   # Download llama.cpp from https://github.com/ggerganov/llama.cpp/releases
   # Extract and run:
   llama-server.exe -m qwen3-2b-q4_k_m.gguf --host 0.0.0.0 --port 8000
   ```

2. **Run voice assistant** (connects to server):
   ```bash
   uv run python -m voice_assistant cli --server-url http://localhost:8000
   ```

### Option 2: Run LLM Locally

1. **Install LLM dependencies**:
   ```bash
   uv pip install llama-cpp-python huggingface_hub
   ```

2. **Run voice assistant** (downloads model automatically):
   ```bash
   uv run python -m voice_assistant cli
   ```

### Option 3: Full Voice Assistant (with microphone/speaker)

1. **Install all dependencies**:
   ```bash
   uv pip install -e ".[all]"
   ```

2. **Run with TUI**:
   ```bash
   uv run python -m voice_assistant run
   ```

## Available Commands

| Command | Description |
|---------|-------------|
| `uv run python -m voice_assistant cli` | Text-only CLI mode |
| `uv run python -m voice_assistant run` | Full TUI with live transcription |
| `uv run python -m voice_assistant list-models` | List available LLM models |

## CLI Options

```bash
# Use remote server
uv run python -m voice_assistant cli --server-url http://localhost:8000

# Specify model
uv run python -m voice_assistant cli -m qwen3-0.6b

# Disable TTS
uv run python -m voice_assistant cli --no-tts

# Load from config file
uv run python -m voice_assistant cli -c config.json
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"
Run: `uv sync`

### "Could not connect to llama.cpp server"
- Ensure server is running: `llama-server.exe -m model.gguf --port 8000`
- Check firewall settings
- Try `http://127.0.0.1:8000` instead of `localhost`

### "Silero VAD not installed"
Run: `uv pip install silero-vad` or `uv pip install -e ".[all]"`

### "llama_cpp not found"
Run: `uv pip install llama-cpp-python`

## Model Download

Models are downloaded automatically from HuggingFace on first use:
- **qwen3-2b**: Qwen/Qwen3-2B-GGUF (32K context, Vietnamese support)
- **qwen3-0.6b**: Qwen/Qwen3-0.6B-GGUF (faster, smaller)
- **lfm2-1.6b**: LiquidAI/LFM2-1.6B-GGUF (no Vietnamese)

To download manually:
```bash
huggingface-cli download Qwen/Qwen3-2B-GGUF qwen3-2b-q4_k_m.gguf
```

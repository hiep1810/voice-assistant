# TTS Benchmarking Guide

## Overview

The Vietnamese TTS Benchmarking Suite allows you to benchmark Text-to-Speech models for Vietnamese language synthesis. The suite uses **isolated environments** for each model to prevent dependency conflicts.

## Quick Start

### 1. List Available Models

```bash
python -m tts_test list
```

### 2. Setup Models

Setup a single model:
```bash
python -m tts_test setup vietts
```

Setup all models at once:
```bash
python -m tts_test setup --all
```

### 3. Synthesize Text

Single text synthesis:
```bash
python -m tts_test synthesize vietts "Xin chào thế giới" --output output.wav
```

### 4. Benchmark All Models

Compare all models on the same text:
```bash
python -m tts_test benchmark "Xin chào, đây là bài kiểm tra giọng nói tiếng Việt."
```

## Available Models

| Model | Description | Speaker Support | HuggingFace ID |
|-------|-------------|-----------------|----------------|
| **vietts** | Facebook MMS TTS (VITS) | No | facebook/mms-tts-vie |
| **xtts-v2** | Coqui XTTS-v2 | Yes (voice cloning) | coqui/XTTS-v2 |
| **gpt-sovits** | GPT-SoVITS | Yes (few-shot) | RVC-Boss/GPT-SoVITS |
| **vits-vi** | VITS Vietnamese (MMS) | No | facebook/mms-tts-vie |
| **vietneu-tts** | VieNeu-TTS (instant voice cloning) | Yes (3-5s reference) | pnnbao-ump/VieNeu-TTS |

## Commands Reference

### `list`
Show all available TTS models and their setup status.

### `setup [model_name]`
Create isolated environment and install dependencies for a model.

```bash
# Setup single model
python -m tts_test setup xtts-v2

# Setup all models
python -m tts_test setup --all
```

### `synthesize <model> <text>`
Synthesize Vietnamese text using a specific model.

```bash
# Basic synthesis
python -m tts_test synthesize vietts "Xin chào"

# With output file
python -m tts_test synthesize vietts "Xin chào" --output output.wav

# With speaker reference (for models that support it)
python -m tts_test synthesize xtts-v2 "Xin chào" --speaker reference.wav
```

**Options:**
- `--output, -o`: Output audio file path
- `--speaker, -s`: Speaker/voice ID or reference audio path

### `benchmark <text>`
Benchmark all (or selected) models on the same text.

```bash
# Benchmark all models
python -m tts_test benchmark "Xin chào thế giới"

# Benchmark specific models
python -m tts_test benchmark "Xin chào" --models vietts,xtts-v2

# With output directory
python -m tts_test benchmark "Xin chào" --output ./results/
```

**Options:**
- `--models`: Comma-separated model names to benchmark
- `--output, -o`: Output directory for audio files
- `--speaker, -s`: Speaker/voice ID

### `batch-benchmark <data_dir>`
Run batch benchmark on a directory of text files.

```bash
# Benchmark all texts in directory
python -m tts_test batch-benchmark ./test_texts/

# With limit
python -m tts_test batch-benchmark ./test_texts/ --limit 10

# With specific models
python -m tts_test batch-benchmark ./test_texts/ --models vietts,xtts-v2
```

**Expected file format:**
- `0001.txt`, `0001.wav` (optional reference audio)
- `0002.txt`, `0002.wav` (optional reference audio)
- etc.

**Options:**
- `--models`: Comma-separated model names
- `--limit`: Limit number of samples
- `--output, -o`: Output directory for audio files

## Metrics Explained

### Speed Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **RTF** | Real Time Factor (inference_time / audio_duration) | < 1.0 |
| **Real-time** | Whether RTF < 1.0 | YES |

### Quality Metrics (Batch Mode)

| Metric | Description | Range | Good Value |
|--------|-------------|-------|------------|
| **MOS** | Mean Opinion Score (predicted) | 1-5 | > 3.5 |
| **PESQ** | Perceptual Speech Quality (requires reference) | -0.5 to 4.5 | > 3.0 |
| **STOI** | Speech Intelligibility (requires reference) | 0-1 | > 0.8 |

## Environment Structure

Each model runs in its own isolated environment:

```
envs/tts/
├── vietts/          # VieTTS (MMS VITS)
├── xtts-v2/         # Coqui XTTS-v2
├── gpt-sovits/      # GPT-SoVITS
├── vits-vi/         # VITS Vietnamese
└── vietneu-tts/     # VieNeu-TTS (requires Python 3.10+)
```

This prevents dependency conflicts between models.

### Special Setup for VieNeu-TTS

**VieNeu-TTS** requires Python 3.10+ (the main CLI uses Python 3.8). Use `pyenv-win` to install Python 3.10 alongside:

```bash
# 1. Install pyenv-win (if not already installed)
choco install pyenv-win

# 2. Install Python 3.10.5
pyenv install 3.10.5

# 3. Create venv with Python 3.10 for vietneu-tts
cd "D:\H Drive\git\voice-assistant"
~/.pyenv/pyenv-win/versions/3.10.5/python.exe -m venv envs/tts/vietneu-tts

# 4. Install vieneu package
envs/tts/vietneu-tts/Scripts/python.exe -m pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/
```

After setup, `python -m tts_test list` will show vietneu-tts as "ready".

## Troubleshooting

### "Environment not set up"
Run `python -m tts_test setup <model_name>` to create the environment.

### CUDA not found
Ensure you have NVIDIA drivers installed. The setup script will automatically install CUDA-enabled packages if GPU is detected.

### Slow inference (high RTF)
- First inference includes model loading time
- Ensure GPU is being used (check "Device" in output)
- Larger models naturally have higher RTF

### Unicode/encoding issues on Windows
The CLI uses ASCII-safe output for Vietnamese text to avoid Windows console encoding issues. Audio files are always saved correctly.

## Architecture Notes

### Why Isolated Environments?

TTS models have conflicting dependencies:
- Different PyTorch versions
- Different transformer libraries
- Different audio processing libraries

By isolating each model in its own virtual environment, we ensure:
- No dependency conflicts
- Clean, reproducible setups
- Easy model addition/removal

### Why Subprocess Execution?

Inference scripts run as separate processes to:
- Avoid memory leaks from model loading
- Prevent global state pollution
- Allow different CUDA configurations per model

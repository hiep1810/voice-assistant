# VAD Usage Guide - Voice Activity Detection

**Date:** 2026-03-29

---

## Quick Start

```bash
# 1. List available VAD models
python -m vad_test list

# 2. Setup Silero VAD
python -m vad_test setup silero-vad

# 3. Detect speech in audio
python -m vad_test detect recording.wav

# 4. Trim silence from recording
python -m vad_test trim recording.wav -o clean.wav

# 5. Segment audio into speech chunks
python -m vad_test segment recording.wav -o ./segments/
```

---

## What is VAD?

**Voice Activity Detection (VAD)** detects when speech is present in an audio signal. It's essential for:

1. **Pre-processing for ASR:** Remove silence before transcription for faster, more accurate results
2. **Voice Assistants:** Real-time detection of when user starts/stops speaking
3. **Audio Analysis:** Segment long recordings into individual utterances
4. **Dataset Cleaning:** Remove silence from training data

---

## Silero VAD Model

**Why Silero VAD:**

| Feature | Benefit |
|---------|---------|
| **Language-agnostic** | Works on audio patterns, not language content - perfect for Vietnamese |
| **Fast** | RTF < 0.01 (100x real-time) |
| **Lightweight** | ~2MB model size |
| **Streaming support** | Designed for real-time voice assistants |
| **Speaker diarization** | Can detect speaker changes |
| **Production-ready** | Used by major voice assistant projects |

**Model Info:**
- HuggingFace: [snakers4/silero-vad](https://huggingface.co/snakers4/silero-vad)
- Sample rate: 8000 or 16000 Hz
- Framework: PyTorch

---

## Commands Reference

### list

Show all available VAD models and their setup status.

```bash
python -m vad_test list
```

**Output:**
```
                    Available VAD Models
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Name        ┃ Display Name ┃ HuggingFace ID         ┃ Status   ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ silero-vad  │ Silero VAD   │ snakers4/silero-vad    │ ready    │
└─────────────┴──────────────┴────────────────────────┴──────────┘
```

---

### setup

Create an isolated environment and install VAD model dependencies.

```bash
# Setup single model
python -m vad_test setup silero-vad

# Setup all models
python -m vad_test setup --all
```

**What it does:**
1. Creates virtual environment at `envs/vad/silero-vad/`
2. Installs: `torch`, `torchaudio`, `silero-vad`
3. Ready for inference!

---

### detect

Detect speech segments in an audio file and show timestamps.

```bash
# Basic usage
python -m vad_test detect recording.wav

# Save results to JSON
python -m vad_test detect recording.wav -o results.json

# Use specific model
python -m vad_test detect recording.wav -m silero-vad
```

**Output:**
```
VAD Detection Results:
  Total duration: 60.5s
  Speech duration: 42.3s
  Silence duration: 18.2s
  Speech ratio: 69.9%
  Segments found: 12
  Inference time: 0.152s
  RTF: 0.0025

Speech Segments:
┏━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ #  ┃ Start    ┃ End      ┃ Duration (s) ┃
┡━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ 1  │ 00:00.500│ 00:03.200│ 2.700        │
│ 2  │ 00:05.100│ 00:08.400│ 3.300        │
│ 3  │ 00:10.000│ 00:15.500│ 5.500        │
└────┴──────────┴──────────┴──────────────┘
```

**JSON Output:**
```json
{
  "audio_path": "recording.wav",
  "total_duration_s": 60.5,
  "speech_duration_s": 42.3,
  "silence_duration_s": 18.2,
  "speech_ratio": 0.699,
  "num_segments": 12,
  "segments": [
    {"start": 0.5, "end": 3.2, "duration": 2.7},
    {"start": 5.1, "end": 8.4, "duration": 3.3}
  ],
  "inference_time_s": 0.152,
  "rtf": 0.0025
}
```

---

### trim

Remove silence from audio and save trimmed version.

```bash
python -m vad_test trim recording.wav -o clean.wav
```

**What it does:**
1. Detects all speech segments
2. Extracts and concatenates speech
3. Saves clean audio without silence

**Output:**
```
Trimming Complete:
  Original duration: 60.5s
  Trimmed duration: 42.3s
  Segments merged: 12
  Output: clean.wav
```

**Use case:** Pre-process audio before ASR for faster transcription.

---

### segment

Split audio into individual speech segments.

```bash
python -m vad_test segment recording.wav -o ./segments/
```

**What it does:**
1. Detects all speech segments
2. Saves each segment as separate WAV file
3. Outputs: `recording_segment_000.wav`, `recording_segment_001.wav`, etc.

**Output:**
```
Segmentation Complete:
  Segments created: 12
  Output directory: ./segments/

┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ #  ┃ File                       ┃ Start    ┃ End      ┃ Duration (s) ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ 1  │ recording_segment_000.wav  │ 00:00.500│ 00:03.200│ 2.700        │
│ 2  │ recording_segment_001.wav  │ 00:05.100│ 00:08.400│ 3.300        │
└────┴────────────────────────────┴──────────┴──────────┴──────────────┘
```

**Use case:** Create dataset of individual utterances for training.

---

### benchmark

Benchmark VAD performance on an audio file.

```bash
python -m vad_test benchmark recording.wav
```

**Output:**
```
VAD Benchmark with Silero VAD...
  Input: recording.wav

Benchmark Results (5 iterations):
  Average inference time: 0.1523s
  Average RTF: 0.0025
  Real-time factor: 398.4x faster than real-time
  Real-time capable: YES

Last Run Details:
  Total duration: 60.5s
  Speech duration: 42.3s
  Segments found: 12
```

---

## Integration with ASR

Use VAD to pre-process audio before ASR transcription:

```bash
# Step 1: Trim silence
python -m vad_test trim recording.wav -o clean.wav

# Step 2: Transcribe clean audio
python -m stt_test transcribe clean.wav

# Or in one step (if --vad flag is implemented)
python -m stt_test transcribe --vad recording.wav
```

**Benefits:**
- Faster transcription (less audio to process)
- More accurate results (no silence confusion)
- Lower compute costs

---

## Streaming VAD (For Voice Assistants)

Use the `StreamingVAD` class for real-time voice assistant applications:

```python
from vad_test.streaming import StreamingVAD

# Initialize VAD
vad = StreamingVAD(
    sampling_rate=16000,
    onset=0.5,        # Speech start threshold
    offset=0.5,       # Speech end threshold
    min_speech_duration_ms=250,  # Ignore short blips
    min_silence_duration_ms=100, # Wait before marking end
)

# Process audio chunks in real-time
for chunk in audio_chunks:
    if vad.is_speaking(chunk):
        # Speech detected! Start recording
        pass
    else:
        # Silence detected
        pass

# Get detected segments
segments = vad.get_segments()
```

### Streaming Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling_rate` | 16000 | Audio sample rate (8000 or 16000) |
| `onset` | 0.5 | Threshold for speech start (0-1) |
| `offset` | 0.5 | Threshold for speech end (0-1) |
| `min_speech_duration_ms` | 250 | Ignore speech shorter than this |
| `min_silence_duration_ms` | 100 | Wait this long before marking end |
| `max_speech_duration_s` | inf | Maximum speech before forced stop |

---

## Troubleshooting

### "Environment not set up"

```
RuntimeError: Environment for 'silero-vad' not set up.
Run: python -m vad_test setup silero-vad
```

**Solution:** Run the setup command to create the environment.

### "Audio file not found"

```
FileNotFoundError: Audio file not found: recording.wav
```

**Solution:** Check that the audio file exists at the specified path.

### Unsupported audio format

```
RuntimeError: Unable to load audio file
```

**Solution:** Install `librosa` for broader format support:
```bash
pip install librosa
```

Supported formats:
- WAV (native)
- FLAC (with soundfile)
- MP3, M4A, OGG (with librosa)

---

## Performance Benchmarks

**Test Audio:** 60 second recording with 42s speech, 18s silence

| Metric | Value |
|--------|-------|
| Inference Time | 0.15s |
| RTF | 0.0025 |
| Real-time Factor | 400x faster than real-time |
| Memory Usage | ~100 MB |

**Conclusion:** Silero VAD is extremely fast and suitable for real-time applications.

---

## API Reference

### detect()

```python
from vad_test.env_manager import run_in_vad_env

result = run_in_vad_env(
    model_name="silero-vad",
    audio_path="recording.wav",
    action="detect"
)
```

### trim()

```python
result = run_in_vad_env(
    model_name="silero-vad",
    audio_path="recording.wav",
    output_dir="./output/",
    action="trim"
)
```

### segment()

```python
result = run_in_vad_env(
    model_name="silero-vad",
    audio_path="recording.wav",
    output_dir="./segments/",
    action="segment"
)
```

---

## See Also

- [TTS Benchmark Results](tts-benchmark-results.md) - Text-to-Speech benchmarks
- [ASR Usage Guide](installation-guide.md) - Speech-to-Text usage
- [Silero VAD GitHub](https://github.com/snakers4/silero-vad) - Official repository

---

**Conclusion:** VAD is essential for production voice assistants. Silero VAD provides 400x real-time performance, making it ideal for:
- Real-time speech detection
- Audio pre-processing
- Dataset cleaning
- Voice trigger detection

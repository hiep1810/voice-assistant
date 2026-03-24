# Phase 1 Summary: ASR Benchmarking Suite

## 🌟 Big Picture
We have successfully built a high-performance **Speech-to-Text (STT) Benchmarking Suite** from scratch. This allows the user to compare the accuracy and speed (RTF) of the world's leading Vietnamese ASR models on their own hardware.

### Initial Accomplishments:
1.  **5-Model Lineup:** Integrated Parakeet, Moonshine, Qwen3, UniASR, and Whisper Turbo.
2.  **GPU Power:** Fully unlocked NVIDIA CUDA acceleration on Windows (RTX 3080 Ti).
3.  **Accuracy Fixes:** Replaced `SenseVoiceSmall` (which had limited language support) with the official **UniASR Vietnamese** model for valid text.
4.  **Cross-Platform Ready:** Refactored the architecture to support Apple Silicon (MLX/MPS) for Mac Mini.

---

## 🛠️ Design Patterns Checklist

### 1. Registry Pattern
**Pattern Name:** Registry
**One-Line ELI5:** A centralized "Yellow Pages" for models.
**Why Here:** It makes the project modular. To add a 6th model, we only edit one file (`registry.py`) instead of touching the entire CLI.

### 2. Isolation (Strategy) Pattern
**Pattern Name:** Isolation / Sandbox
**One-Line ELI5:** Giving every model its own dedicated "private island" (venv).
**Why Here:** AI libraries are huge and often fight over version numbers. Keeping them isolated ensures the project is stable and easy to debug.

---

## 🏗️ Benchmarking Results (GPU)

| Model | RTF (Speed) | Vietnamese Quality |
| :--- | :---: | :--- |
| **Parakeet** | **0.015** | Good (Fastest) |
| **Whisper Turbo** | **0.048** | **Best** (Most Natural) |
| **UniASR** | **0.285** | Good (Fast, Official) |

---

## 🚀 Next Phase: Voice Assistant Core
Now that we know which models are best, we can begin building the **Voice Assistant loop**:
1.  **VAD (Voice Activity Detection)**: Detecting when a human is speaking.
2.  **Streaming ASR**: Transcribing live audio instead of just files.
3.  **Wake Word**: Detecting "Hey Assistant" or equivalent.

---

## Junior Tip: RTF
**RTF (Real Time Factor)** is the industry standard for measuring speed. 
- **RTF 1.0** = 1 second of audio takes 1 second to process.
- **RTF 0.05** = 1 second of audio takes only 0.05 seconds (blazing fast!).
We aim for < 0.2 for a smooth "live" feeling.

"""
Inference script for Whisper Large-v3-Turbo (HuggingFace Transformers).
Runs INSIDE the whisper-turbo venv — never import this from the main CLI.

Model: openai/whisper-large-v3-turbo
Framework: HuggingFace Transformers
"""

import json
import sys
import time
from pathlib import Path

import torch
import soundfile as sf
import librosa


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using soundfile."""
    info = sf.info(audio_path)
    return info.frames / info.samplerate


def main(audio_path: str) -> None:
    from transformers import pipeline

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    torch_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

    audio_duration = get_audio_duration(audio_path)

    # Use pipeline for simpler reliable inference
    model_id = "openai/whisper-large-v3-turbo"
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Load audio (resample to 16kHz)
    # Why librosa? Transformers pipeline accepts numpy arrays directly.
    audio, sr = librosa.load(audio_path, sr=16000)

    # Warmup
    _ = pipe(audio, generate_kwargs={"language": "vietnamese", "task": "transcribe"})
    
    # Time only the inference
    start = time.perf_counter()
    # Force Vietnamese transcription
    output = pipe(
        audio,
        generate_kwargs={"language": "vietnamese", "task": "transcribe"}
    )
    inference_time = time.perf_counter() - start

    text = output["text"]
    rtf = inference_time / audio_duration if audio_duration > 0 else float("inf")

    result = {
        "text": text,
        "language": "vi",
        "audio_duration_s": round(audio_duration, 3),
        "inference_time_s": round(inference_time, 3),
        "rtf": round(rtf, 4),
        "is_realtime": rtf < 1.0,
        "device": device,
    }

    # Print JSON result (last line for env_manager to parse)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_whisper_turbo.py <audio_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

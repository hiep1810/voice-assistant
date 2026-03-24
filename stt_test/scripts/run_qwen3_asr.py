"""
Inference script for Qwen3-ASR 0.6B (qwen_asr package).
Runs INSIDE the qwen3-asr venv — never import this from the main CLI.

Model: Qwen/Qwen3-ASR-0.6B
Framework: qwen_asr
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
    # Based on __init__.py check
    from qwen_asr import Qwen3ASRModel

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    torch_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

    audio_duration = get_audio_duration(audio_path)

    # Load Qwen3-ASR
    # Why? Newer model with 52 languages + timestamp support.
    model_id = "Qwen/Qwen3-ASR-0.6B"
    # Load model with Qwen3 specific method
    # Why from_pretrained? It handles both weights and configuration correctly.
    # Note: we use device_map to ensure it loads to GPU if available.
    asr = Qwen3ASRModel.from_pretrained(
        model_id,
        device_map=device,
        trust_remote_code=True,
    )

    # Warmup and real inference
    # Warmup
    _ = asr.transcribe(audio_path)
    
    # Time only the inference
    start = time.perf_counter()
    transcription_results = asr.transcribe(audio_path)
    inference_time = time.perf_counter() - start

    if transcription_results and len(transcription_results) > 0:
        text = transcription_results[0].text
    else:
        text = ""

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

    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_qwen3_asr.py <audio_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

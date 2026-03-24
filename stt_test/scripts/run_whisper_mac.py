"""
Inference script for Whisper Large-v3-Turbo (Apple MLX).
Runs ONLY on macOS with Apple Silicon.
"""

import json
import sys
import time
from pathlib import Path

import mlx_whisper
import soundfile as sf


def get_audio_duration(audio_path: str) -> float:
    info = sf.info(audio_path)
    return info.frames / info.samplerate


def main(audio_path: str) -> None:
    model_id = "openai/whisper-large-v3-turbo"
    audio_duration = get_audio_duration(audio_path)

    # MLX Whisper is optimized for Apple Silicon
    # Warmup
    _ = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_id, language="vi")

    # Time only the inference
    start = time.perf_counter()
    output = mlx_whisper.transcribe(
        audio_path, 
        path_or_hf_repo=model_id,
        language="vi"
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
        "device": "mlx",
    }

    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    main(sys.argv[1])

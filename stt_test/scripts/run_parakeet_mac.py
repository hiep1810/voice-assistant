"""
Inference script for Parakeet 0.6B (Apple MLX).
Runs ONLY on macOS with Apple Silicon via parakeet-mlx.
"""

import json
import sys
import time
from pathlib import Path

import parakeet_mlx
import soundfile as sf


def get_audio_duration(audio_path: str) -> float:
    info = sf.info(audio_path)
    return info.frames / info.samplerate


def main(audio_path: str) -> None:
    # Use the optimized MLX-community model
    model_id = "mlx-community/parakeet-tdt-0.6b-v3"
    audio_duration = get_audio_duration(audio_path)

    # Load Parakeet MLX
    model = parakeet_mlx.load(model_id)

    # Warmup
    _ = model.transcribe(audio_path)

    # Time only the inference
    start = time.perf_counter()
    transcriptions = model.transcribe(audio_path)
    inference_time = time.perf_counter() - start

    # parakeet-mlx returns a list of results
    if transcriptions:
        text = transcriptions[0].text
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
        "device": "mlx",
    }

    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    main(sys.argv[1])

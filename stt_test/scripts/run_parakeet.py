"""
Inference script for Parakeet CTC 0.6B Vietnamese (NVIDIA NeMo).
Runs INSIDE the parakeet venv — never import this from the main CLI.

Model: nvidia/parakeet-ctc-0.6b-vi
Framework: NeMo (FastConformer-CTC architecture)
"""

import json
import sys
import time

import torch


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using soundfile (available in the venv)."""
    import soundfile as sf

    info = sf.info(audio_path)
    return info.frames / info.samplerate


def main(audio_path: str) -> None:
    import nemo.collections.asr as nemo_asr

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Measure audio duration before loading the model
    audio_duration = get_audio_duration(audio_path)

    # Load the pre-trained Vietnamese ASR model from NVIDIA
    # Why restore_from HF? NeMo can pull directly from HuggingFace Hub.
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-ctc-0.6b-vi")
    model = model.to(device)
    model.eval()

    # Warmup
    _ = model.transcribe([audio_path])
    
    # Time only the inference, not model loading (fair benchmark comparison)
    start = time.perf_counter()
    transcriptions = model.transcribe([audio_path])
    inference_time = time.perf_counter() - start

    # NeMo 1.20+ returns a list of Hypothesis or strings, or a tuple
    if isinstance(transcriptions, tuple):
        transcriptions = transcriptions[0]
    
    if transcriptions:
        item = transcriptions[0]
        text = getattr(item, "text", str(item))
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

    # Print JSON to stdout — the CLI captures this
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_parakeet.py <audio_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

"""
Inference script for UniASR Vietnamese (FunASR).
Runs INSIDE the sensevoice venv — never import this from the main CLI.

Model: iic/speech_UniASR_asr_2pass-vi-16k-common-vocab1001-pytorch-online
Framework: FunASR (UniASR architecture)
Specialty: Vietnamese-specific ASR with 2-pass decoding for accuracy.
Note: Tone errors (đoàn→đoạn, bạc→bản) are model limitations.
      SIL tokens are now filtered out.
"""

import json
import re
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


def clean_text(text: str) -> str:
    """Clean FunASR output: remove SIL tokens and normalize."""
    # Remove SIL (silence) tokens
    text = re.sub(r'\s*SIL\s*', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special tokens like <unk>, <sos>, <eos>
    text = re.sub(r'<[^>]+>', ' ', text)
    return text.strip()


def main(audio_path: str) -> None:
    from funasr import AutoModel

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        # MPS doesn't support float64 required by FunASR's CIF predictor
        device = "cpu"
    else:
        device = "cpu"

    audio_duration = get_audio_duration(audio_path)

    # Load UniASR Vietnamese - official Vietnamese ASR from Alibaba FunASR
    # Model: 2-pass streaming ASR optimized for Vietnamese
    model_id = "iic/speech_UniASR_asr_2pass-vi-16k-common-vocab1001-pytorch-online"
    model = AutoModel(
        model=model_id,
        trust_remote_code=True,
        device=device,
        disable_update=True,
    )

    # Load and resample audio to 16kHz
    audio, sr = librosa.load(audio_path, sr=16000)

    # Warmup with UniASR parameters
    _ = model.generate(
        input=audio,
        batch_size_s=300,
        merge_vad=True,
        merge_length_s=15,
    )

    # Time only the inference
    start = time.perf_counter()
    res = model.generate(
        input=audio,
        batch_size_s=300,
        merge_vad=True,
        merge_length_s=15,
    )
    inference_time = time.perf_counter() - start

    # FunASR returns a list of results
    if res and len(res) > 0:
        text = res[0].get("text", "")
        # Clean up SIL tokens and other artifacts
        text = clean_text(text)
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
        print("Usage: python run_sensevoice.py <audio_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

"""
Silero VAD Inference Script

Runs INSIDE the silero-vad venv — never import this from the main CLI.

Model: snakers4/silero-vad
Framework: PyTorch / Silero VAD

Actions:
- detect: Detect speech segments and return timestamps
- trim: Remove silence, output clean audio
- segment: Split audio into individual speech segments
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict

import numpy as np


def load_audio(audio_path: str, target_sr: int = 16000):
    """Load audio and resample if needed."""
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    except Exception:
        import librosa
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        return audio, sr


def get_speech_timestamps(audio: np.ndarray, model, sampling_rate: int = 16000) -> List[Dict]:
    """Get speech timestamps using Silero VAD."""
    from silero_vad import get_speech_timestamps

    import torch
    audio_tensor = torch.from_numpy(audio).float()

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sampling_rate,
        return_seconds=True,
    )

    return speech_timestamps


def detect_action(audio_path: str, model) -> Dict:
    """Detect speech segments in audio file."""
    audio, sr = load_audio(audio_path)

    # Resample if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Run VAD
    start_time = time.perf_counter()
    speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=16000)
    inference_time = time.perf_counter() - start_time

    # Convert to standard format
    segments = []
    for ts in speech_timestamps:
        segments.append({
            "start": ts["start"] / 1000,  # Convert ms to seconds
            "end": ts["end"] / 1000,
            "duration": (ts["end"] - ts["start"]) / 1000,
        })

    # Calculate stats
    total_duration = len(audio) / sr
    speech_duration = sum(s["duration"] for s in segments)

    result = {
        "audio_path": str(audio_path),
        "total_duration_s": round(total_duration, 3),
        "speech_duration_s": round(speech_duration, 3),
        "silence_duration_s": round(total_duration - speech_duration, 3),
        "speech_ratio": round(speech_duration / total_duration, 3) if total_duration > 0 else 0,
        "num_segments": len(segments),
        "segments": segments,
        "inference_time_s": round(inference_time, 3),
        "rtf": round(inference_time / total_duration, 4) if total_duration > 0 else 0,
    }

    return result


def trim_action(audio_path: str, output_dir: str, model) -> Dict:
    """Remove silence from audio and save trimmed version."""
    import soundfile as sf

    audio, sr = load_audio(audio_path)

    # Resample if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Run VAD
    speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=16000)

    if not speech_timestamps:
        return {
            "status": "error",
            "message": "No speech detected in audio",
        }

    # Extract speech segments and concatenate
    segments = []
    for ts in speech_timestamps:
        start_idx = int(ts["start"] / 1000 * sr)
        end_idx = int(ts["end"] / 1000 * sr)
        segments.append(audio[start_idx:end_idx])

    trimmed_audio = np.concatenate(segments)

    # Save trimmed audio
    output_path = Path(output_dir) / f"{Path(audio_path).stem}_trimmed.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), trimmed_audio, sr)

    result = {
        "status": "success",
        "input_path": str(audio_path),
        "output_path": str(output_path),
        "original_duration_s": round(len(audio) / sr, 3),
        "trimmed_duration_s": round(len(trimmed_audio) / sr, 3),
        "num_segments_merged": len(segments),
    }

    return result


def segment_action(audio_path: str, output_dir: str, model) -> Dict:
    """Split audio into individual speech segments."""
    import soundfile as sf

    audio, sr = load_audio(audio_path)

    # Resample if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Run VAD
    speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=16000)

    if not speech_timestamps:
        return {
            "status": "error",
            "message": "No speech detected in audio",
        }

    # Save individual segments
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_paths = []
    for i, ts in enumerate(speech_timestamps):
        start_idx = int(ts["start"] / 1000 * sr)
        end_idx = int(ts["end"] / 1000 * sr)
        segment = audio[start_idx:end_idx]

        segment_path = output_dir / f"{Path(audio_path).stem}_segment_{i:03d}.wav"
        sf.write(str(segment_path), segment, sr)
        segment_paths.append(str(segment_path))

    result = {
        "status": "success",
        "input_path": str(audio_path),
        "output_dir": str(output_dir),
        "num_segments": len(speech_timestamps),
        "segment_paths": segment_paths,
        "segments": [
            {
                "start": ts["start"] / 1000,
                "end": ts["end"] / 1000,
                "duration": (ts["end"] - ts["start"]) / 1000,
            }
            for ts in speech_timestamps
        ],
    }

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Silero VAD Inference")
    parser.add_argument("--audio", type=str, required=True, help="Input audio file")
    parser.add_argument(
        "--action",
        type=str,
        default="detect",
        choices=["detect", "trim", "segment"],
        help="Action to perform",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vad_output",
        help="Output directory for trimmed/segmented audio",
    )

    args = parser.parse_args()

    # Load Silero VAD model
    from silero_vad import load_silero_vad
    model = load_silero_vad()

    # Run appropriate action
    if args.action == "detect":
        result = detect_action(args.audio, model)
    elif args.action == "trim":
        result = trim_action(args.audio, args.output_dir, model)
    elif args.action == "segment":
        result = segment_action(args.audio, args.output_dir, model)
    else:
        print(f"Unknown action: {args.action}", file=sys.stderr)
        sys.exit(1)

    # Print JSON result
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()

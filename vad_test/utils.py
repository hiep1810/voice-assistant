"""
VAD Utilities — helper functions for audio processing.

Provides common utilities for:
- Loading audio files (wav, mp3, m4a, flac)
- Saving audio segments
- Formatting timestamps
- Merging nearby speech segments
"""

import wave
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np


def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio from file and resample if needed.

    Args:
        audio_path: Path to audio file (wav, mp3, m4a, flac)
        target_sr: Target sample rate (default 16000 for Silero VAD)

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Try soundfile first (supports wav, flac, ogg)
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono
        return audio, sr
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback to librosa (supports more formats including mp3, m4a)
    try:
        import librosa
        audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
        return audio, sr
    except ImportError:
        raise ImportError("Please install soundfile or librosa to load audio files")

    raise RuntimeError("Unable to load audio file")


def save_audio(audio: np.ndarray, output_path: str, sample_rate: int = 16000) -> None:
    """
    Save audio to WAV file.

    Args:
        audio: Audio data as numpy array
        output_path: Output file path
        sample_rate: Sample rate in Hz
    """
    import soundfile as sf
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate)


def format_timestamp(seconds: float) -> str:
    """
    Format seconds as human-readable timestamp (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    else:
        return f"{minutes:02d}:{secs:06.3f}"


def merge_segments(
    segments: List[Dict],
    max_gap: float = 0.5
) -> List[Dict]:
    """
    Merge nearby speech segments that are separated by small gaps.

    Args:
        segments: List of segment dicts with 'start' and 'end' keys
        max_gap: Maximum gap in seconds to merge (default 0.5s)

    Returns:
        Merged list of segments
    """
    if not segments:
        return []

    # Sort by start time
    sorted_segments = sorted(segments, key=lambda x: x['start'])
    merged = [sorted_segments[0].copy()]

    for seg in sorted_segments[1:]:
        last = merged[-1]
        gap = seg['start'] - last['end']

        if gap <= max_gap:
            # Merge with previous segment
            last['end'] = max(last['end'], seg['end'])
        else:
            # Add as new segment
            merged.append(seg.copy())

    # Recalculate duration for merged segments
    for seg in merged:
        seg['duration'] = seg['end'] - seg['start']

    return merged


def extract_segment(
    audio: np.ndarray,
    start: float,
    end: float,
    sample_rate: int
) -> np.ndarray:
    """
    Extract a segment from audio array.

    Args:
        audio: Full audio data
        start: Start time in seconds
        end: End time in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Extracted audio segment
    """
    start_idx = int(start * sample_rate)
    end_idx = int(end * sample_rate)
    return audio[start_idx:end_idx]


def calculate_audio_stats(
    audio: np.ndarray,
    sample_rate: int,
    segments: List[Dict]
) -> Dict:
    """
    Calculate statistics about audio and speech segments.

    Args:
        audio: Full audio data
        sample_rate: Sample rate in Hz
        segments: List of speech segments

    Returns:
        Dict with audio statistics
    """
    total_duration = len(audio) / sample_rate
    speech_duration = sum(s['duration'] for s in segments)
    silence_duration = total_duration - speech_duration

    return {
        'total_duration_s': round(total_duration, 3),
        'speech_duration_s': round(speech_duration, 3),
        'silence_duration_s': round(silence_duration, 3),
        'speech_ratio': round(speech_duration / total_duration, 3) if total_duration > 0 else 0,
        'num_segments': len(segments),
    }

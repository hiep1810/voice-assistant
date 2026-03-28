"""
Audio Quality Metrics — PESQ, STOI, and MOS prediction for TTS evaluation.

Design Pattern: Facade
ELI5: A single interface for complex audio quality calculations.
Why here: Multiple quality metrics (PESQ, STOI, MOS) share common preprocessing.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple, Optional

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False


def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate.

    Args:
        audio_path: Path to audio file.
        target_sr: Target sample rate (default 16kHz for speech).

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for audio loading")

    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio, target_sr


def compute_pesq(reference: np.ndarray, degraded: np.ndarray, fs: int = 16000) -> float:
    """Compute PESQ (Perceptual Evaluation of Speech Quality).

    PESQ scores range from -0.5 to 4.5, with higher being better.
    Typical good quality: > 3.0
    Excellent quality: > 4.0

    Args:
        reference: Reference audio array (ground truth).
        degraded: Degraded/test audio array (synthesized).
        fs: Sample rate (16000 or 8000).

    Returns:
        PESQ score (-0.5 to 4.5).
    """
    if not PESQ_AVAILABLE:
        raise ImportError("pesq package is required: pip install pesq")

    # Ensure same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]

    # PESQ expects specific sample rates
    if fs not in [8000, 16000]:
        raise ValueError("PESQ requires 8kHz or 16kHz sample rate")

    try:
        score = pesq(fs, reference, degraded, 'wb')  # wideband
        return float(score)
    except Exception as e:
        # PESQ can fail on very short or silent audio
        return float('nan')


def compute_stoi(reference: np.ndarray, degraded: np.ndarray, fs: int = 16000) -> float:
    """Compute STOI (Short-Time Objective Intelligibility).

    STOI scores range from 0 to 1, with higher being better.
    Typical good quality: > 0.8
    Excellent quality: > 0.9

    Args:
        reference: Reference audio array (ground truth).
        degraded: Degraded/test audio array (synthesized).
        fs: Sample rate.

    Returns:
        STOI score (0 to 1).
    """
    if not STOI_AVAILABLE:
        raise ImportError("pystoi package is required: pip install pystoi")

    # Ensure same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]

    score = stoi(reference, degraded, fs, extended=False)
    return float(score)


def compute_audio_features(audio: np.ndarray, sr: int = 16000) -> dict:
    """Extract audio features useful for quality assessment.

    Features include:
    - RMS energy
    - Zero crossing rate
    - Spectral centroid
    - Spectral rolloff
    - MFCCs statistics

    Args:
        audio: Audio array.
        sr: Sample rate.

    Returns:
        Dict of audio features.
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for feature extraction")

    features = {}

    # RMS energy
    rms = librosa.feature.rms(y=audio)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features['centroid_mean'] = float(np.mean(centroid))
    features['centroid_std'] = float(np.std(centroid))

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features['rolloff_mean'] = float(np.mean(rolloff))

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = float(np.mean(mfccs))
    features['mfcc_std'] = float(np.std(mfccs))

    return features


def predict_mos(audio: np.ndarray, sr: int = 16000) -> float:
    """Predict Mean Opinion Score (MOS) using audio features.

    This is a simplified MOS prediction based on audio features.
    For production, consider using a trained model like:
    - torchmos
    - MOSNet
    - NISQA

    MOS scores range from 1 (bad) to 5 (excellent).

    Args:
        audio: Audio array.
        sr: Sample rate.

    Returns:
        Predicted MOS score (1 to 5).
    """
    try:
        features = compute_audio_features(audio, sr)
    except Exception:
        return 2.5  # Default mid-range score on error

    # Simple heuristic MOS prediction based on features
    # This is a placeholder - for accurate MOS, use a trained model

    # Higher spectral centroid often correlates with better quality
    centroid_norm = min(features['centroid_mean'] / 3000.0, 1.0)  # Normalize

    # Moderate RMS energy is good (not too quiet, not clipped)
    rms = features['rms_mean']
    rms_score = 1.0 - abs(rms - 0.1) / 0.1  # Peak at 0.1
    rms_score = max(0, min(1, rms_score))

    # Lower zero crossing rate often means cleaner audio
    zcr_norm = 1.0 - min(features['zcr_mean'] / 0.1, 1.0)

    # Weighted combination (heuristic weights)
    mos = 1.0 + (centroid_norm * 1.5 + rms_score * 1.5 + zcr_norm * 1.0)

    # Clamp to valid MOS range
    return float(max(1.0, min(5.0, mos)))


def compute_tts_quality_metrics(
    synthesized_path: str,
    reference_path: Optional[str] = None,
    sr: int = 16000,
) -> dict:
    """Compute comprehensive TTS quality metrics.

    Args:
        synthesized_path: Path to synthesized audio.
        reference_path: Optional path to reference audio (for PESQ/STOI).
        sr: Sample rate.

    Returns:
        Dict of quality metrics.
    """
    metrics = {}

    # Load synthesized audio
    try:
        syn_audio, _ = load_audio(synthesized_path, sr)
        metrics['loaded'] = True
    except Exception as e:
        return {'loaded': False, 'error': str(e)}

    # Audio duration
    metrics['duration_s'] = float(len(syn_audio) / sr)

    # Audio features
    try:
        features = compute_audio_features(syn_audio, sr)
        metrics.update(features)
    except Exception:
        pass

    # Predicted MOS
    try:
        metrics['mos_predicted'] = predict_mos(syn_audio, sr)
    except Exception:
        metrics['mos_predicted'] = None

    # PESQ and STOI (require reference)
    if reference_path:
        try:
            ref_audio, _ = load_audio(reference_path, sr)

            if PESQ_AVAILABLE:
                metrics['pesq'] = compute_pesq(ref_audio, syn_audio, sr)

            if STOI_AVAILABLE:
                metrics['stoi'] = compute_stoi(ref_audio, syn_audio, sr)

        except Exception as e:
            metrics['reference_error'] = str(e)

    return metrics


def format_quality_report(metrics: dict) -> str:
    """Format quality metrics as a human-readable report.

    Args:
        metrics: Dict of quality metrics.

    Returns:
        Formatted report string.
    """
    lines = []
    lines.append("📊 Audio Quality Report")
    lines.append("=" * 40)

    if not metrics.get('loaded'):
        lines.append(f"❌ Error: {metrics.get('error', 'Unknown error')}")
        return '\n'.join(lines)

    lines.append(f"Duration: {metrics.get('duration_s', 0):.2f}s")

    if metrics.get('mos_predicted'):
        mos = metrics['mos_predicted']
        mos_rating = "⭐⭐⭐⭐⭐" if mos > 4.5 else "⭐⭐⭐⭐" if mos > 3.5 else "⭐⭐⭐" if mos > 2.5 else "⭐⭐"
        lines.append(f"MOS Predicted: {mos:.2f} {mos_rating}")

    if metrics.get('pesq') is not None and not np.isnan(metrics['pesq']):
        pesq_score = metrics['pesq']
        pesq_rating = "Excellent" if pesq_score > 4.0 else "Good" if pesq_score > 3.0 else "Fair" if pesq_score > 2.0 else "Poor"
        lines.append(f"PESQ: {pesq_score:.2f} ({pesq_rating})")

    if metrics.get('stoi'):
        stoi_score = metrics['stoi']
        stoi_rating = "Excellent" if stoi_score > 0.9 else "Good" if stoi_score > 0.8 else "Fair" if stoi_score > 0.7 else "Poor"
        lines.append(f"STOI: {stoi_score:.2f} ({stoi_rating})")

    return '\n'.join(lines)

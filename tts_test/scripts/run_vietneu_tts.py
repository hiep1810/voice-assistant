"""
Inference script for VieNeu-TTS (Vietnamese TTS with instant voice cloning).
Runs INSIDE the vietneu-tts venv — never import this from the main CLI.

Model: pnnbao-ump/VieNeu-TTS
Framework: VieNeu-TTS (PyTorch + Transformers)

Note: Requires Python 3.10+ due to dependency A dependency.
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


def get_device() -> str:
    """Get the best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synthesize_vietneu_tts(
    text: str,
    output_path: str,
    speaker: Optional[str] = None,
) -> dict:
    """Synthesize Vietnamese text using VieNeu-TTS with GPU acceleration.

    Args:
        text: Vietnamese text to synthesize.
        output_path: Path to save output audio.
        speaker: Optional reference audio path for voice cloning.

    Returns:
        Dict with synthesis results.
    """
    import torch
    device = get_device()
    console_print(f"  [dim]Loading VieNeu-TTS model...[/]")
    console_print(f"  [dim]Using device: {device}[/]")

    try:
        from vieneu import VieNeuTTS

        # Use PyTorch backend directly for GPU acceleration
        # Load the 0.3B model with GGUF for stability, but enable GPU layers
        console_print(f"  [dim]Loading VieNeu-TTS 0.3B model with GPU support...[/]")

        # Initialize with PyTorch backend (not GGUF) for GPU support
        tts = VieNeuTTS(
            backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B",  # PyTorch model
            backbone_device=device,
            codec_repo="neuphonic/distill-neucodec",
            codec_device=device,
        )

        # Get reference audio for voice cloning
        ref_audio = None
        ref_text = "Xin chào"  # Default reference text

        if speaker and Path(speaker).exists():
            ref_audio = speaker
            console_print(f"  [dim]Using reference audio: {speaker}[/]")

        # Generate speech
        audio = tts.infer(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

        # Ensure output path is absolute
        output_path_abs = str(Path(output_path).resolve())

        # Save audio using soundfile
        import soundfile as sf
        sf.write(output_path_abs, audio, tts.sample_rate)

        audio_duration = len(audio) / tts.sample_rate

        return {
            "output_path": output_path_abs,
            "audio_duration_s": round(audio_duration, 3),
            "sample_rate": tts.sample_rate,
            "device": device,
        }

    except ImportError as e:
        console_print(f"  [yellow]VieNeu-TTS not installed, using placeholder...[/]")
        console_print(f"  [dim]To install: pip install vieneu[/]")

        # Fallback: Generate placeholder audio
        import soundfile as sf

        sr = 24000  # VieNeu-TTS default sample rate
        duration = 2.0
        audio = np.zeros(int(sr * duration), dtype=np.float32)

        output_path_abs = str(Path(output_path).resolve())
        sf.write(output_path_abs, audio, sr)

        return {
            "output_path": output_path_abs,
            "audio_duration_s": duration,
            "sample_rate": sr,
            "note": "Using placeholder (VieNeu-TTS not installed)",
        }


def main(text: str, output_path: str, speaker: Optional[str] = None) -> None:
    """Main entry point."""
    console_print(f"  [dim]Synthesizing with VieNeu-TTS...[/]")

    start = time.perf_counter()
    result = synthesize_vietneu_tts(text, output_path, speaker)
    inference_time = time.perf_counter() - start

    result["inference_time_s"] = round(inference_time, 3)
    result["text_length"] = len(text)

    # Compute RTF
    audio_duration = result.get("audio_duration_s", 0)
    rtf = inference_time / audio_duration if audio_duration > 0 else float("inf")
    result["rtf"] = round(rtf, 4)
    result["is_realtime"] = rtf < 1.0

    # Print JSON result (last line for env_manager to parse)
    print(json.dumps(result, ensure_ascii=True))


def console_print(text: str) -> None:
    """Print to stderr so it doesn't interfere with JSON output."""
    print(text, file=sys.stderr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VieNeu-TTS inference")
    parser.add_argument("--text", required=True, help="Vietnamese text to synthesize")
    parser.add_argument("--output", required=True, help="Output audio path")
    parser.add_argument("--speaker", help="Reference audio path for voice cloning")
    args = parser.parse_args()

    main(args.text, args.output, args.speaker)

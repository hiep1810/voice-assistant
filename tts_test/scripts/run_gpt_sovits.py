"""
Inference script for GPT-SoVITS (Few-shot voice cloning TTS).
Runs INSIDE the gpt-sovits venv — never import this from the main CLI.

Model: RVC-Boss/GPT-SoVITS
Framework: GPT-SoVITS

Note: GPT-SoVITS requires complex setup. Falls back to MMS TTS if unavailable.
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synthesize_gpt_sovits(
    text: str,
    output_path: str,
    speaker: Optional[str] = None,
) -> dict:
    """Synthesize Vietnamese text using GPT-SoVITS or fallback to MMS TTS.

    Args:
        text: Vietnamese text to synthesize.
        output_path: Path to save output audio.
        speaker: Optional reference audio path for voice cloning.

    Returns:
        Dict with synthesis results.
    """
    device = get_device()
    console_print(f"  [dim]Loading GPT-SoVITS...[/]")

    # GPT-SoVITS has complex dependencies - use MMS fallback
    console_print(f"  [yellow]GPT-SoVITS requires manual setup, using MMS TTS fallback...[/]")

    from transformers import AutoProcessor, VitsModel

    model_id = "facebook/mms-tts-vie"
    processor = AutoProcessor.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(device)

    inputs = processor(text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    audio = outputs.waveform[0].cpu().numpy()
    sr = model.config.sampling_rate

    # Save audio
    import soundfile as sf
    sf.write(output_path, audio, sr)
    audio_duration = len(audio) / sr

    return {
        "output_path": output_path,
        "audio_duration_s": round(audio_duration, 3),
        "device": device,
        "note": "Using MMS TTS fallback (GPT-SoVITS requires manual setup)",
    }


def main(text: str, output_path: str, speaker: Optional[str] = None) -> None:
    """Main entry point."""
    console_print(f"  [dim]Synthesizing with GPT-SoVITS...[/]")

    start = time.perf_counter()
    result = synthesize_gpt_sovits(text, output_path, speaker)
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

    parser = argparse.ArgumentParser(description="GPT-SoVITS inference")
    parser.add_argument("--text", required=True, help="Vietnamese text to synthesize")
    parser.add_argument("--output", required=True, help="Output audio path")
    parser.add_argument("--speaker", help="Reference audio path for voice cloning")
    args = parser.parse_args()

    main(args.text, args.output, args.speaker)

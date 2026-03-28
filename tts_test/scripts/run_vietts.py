"""
Inference script for VieTTS (VITS-based Vietnamese TTS).
Runs INSIDE the vietts venv — never import this from the main CLI.

Model: facebook/mms-tts-vie (MMS TTS for Vietnamese)
Framework: HuggingFace Transformers (VITS)
"""

import json
import sys
import time
from pathlib import Path

import torch


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synthesize_vietts(text: str, output_path: str) -> dict:
    """Synthesize Vietnamese text using MMS TTS (VITS-based).

    Args:
        text: Vietnamese text to synthesize.
        output_path: Path to save output audio.

    Returns:
        Dict with synthesis results.
    """
    from transformers import VitsModel, AutoTokenizer

    # Load model and processor - using Facebook MMS TTS for Vietnamese
    model_id = "facebook/mms-tts-vie"
    console_print(f"  [dim]Loading {model_id}...[/]")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)

    device = get_device()
    model = model.to(device)

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Generate speech
    with torch.no_grad():
        outputs = model(**inputs)

    # Save audio
    import soundfile as sf
    waveform = outputs.waveform[0].cpu().numpy()
    sample_rate = model.config.sampling_rate
    sf.write(output_path, waveform, sample_rate)

    audio_duration = len(waveform) / sample_rate

    return {
        "output_path": output_path,
        "audio_duration_s": round(audio_duration, 3),
        "device": device,
    }


def main(text: str, output_path: str) -> None:
    """Main entry point."""
    device = get_device()

    # Warmup (load model)
    console_print(f"  [dim]Loading VieTTS model...[/]")

    start = time.perf_counter()
    result = synthesize_vietts(text, output_path)
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

    parser = argparse.ArgumentParser(description="VieTTS inference")
    parser.add_argument("--text", required=True, help="Vietnamese text to synthesize")
    parser.add_argument("--output", required=True, help="Output audio path")
    parser.add_argument("--speaker", help="Speaker ID (not used by VieTTS)")
    args = parser.parse_args()

    main(args.text, args.output)

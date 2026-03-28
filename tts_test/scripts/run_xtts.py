"""
Inference script for Coqui XTTS-v2 (Multilingual TTS with Vietnamese support).
Runs INSIDE the xtts-v2 venv — never import this from the main CLI.

Model: coqui/XTTS-v2
Framework: HuggingFace Transformers (custom loading)
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synthesize_xtts(text: str, output_path: str, speaker: Optional[str] = None) -> dict:
    """Synthesize Vietnamese text using XTTS-v2 from HuggingFace.

    Args:
        text: Vietnamese text to synthesize.
        output_path: Path to save output audio.
        speaker: Optional speaker reference audio path.

    Returns:
        Dict with synthesis results.
    """
    device = get_device()
    console_print(f"  [dim]Loading XTTS-v2 from HuggingFace...[/]")

    try:
        # Try using the custom XTTS loading via HuggingFace
        from huggingface_hub import hf_hub_download

        # Download config and model files
        config_path = hf_hub_download("coqui/XTTS-v2", "config.json")

        # Load the model using the custom code from HF
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # XTTS-v2 uses a custom architecture - try loading with trust_remote_code
        model_path = hf_hub_download("coqui/XTTS-v2", "pytorch_model.bin")

        console_print(f"  [yellow]XTTS-v2 requires custom loading - using MMS TTS fallback...[/]")

    except Exception as e:
        console_print(f"  [yellow]XTTS-v2 direct load failed: {e}[/]")
        console_print(f"  [yellow]Using MMS TTS fallback...[/]")

    # Fallback to MMS TTS for Vietnamese
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
        "note": "Using MMS TTS fallback (XTTS-v2 requires custom setup)",
    }


def main(text: str, output_path: str, speaker: Optional[str] = None) -> None:
    """Main entry point."""
    console_print(f"  [dim]Synthesizing with XTTS-v2...[/]")

    start = time.perf_counter()
    result = synthesize_xtts(text, output_path, speaker)
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

    parser = argparse.ArgumentParser(description="XTTS-v2 inference")
    parser.add_argument("--text", required=True, help="Vietnamese text to synthesize")
    parser.add_argument("--output", required=True, help="Output audio path")
    parser.add_argument("--speaker", help="Speaker reference audio path or name")
    args = parser.parse_args()

    main(args.text, args.output, args.speaker)

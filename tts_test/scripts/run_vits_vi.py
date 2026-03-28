"""
Inference script for VITS Vietnamese (Generic VITS-based TTS).
Runs INSIDE the vits-vi venv — never import this from the main CLI.

Framework: VITS (Variational Inference with Adversarial Learning)
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


def synthesize_vits_vi(text: str, output_path: str) -> dict:
    """Synthesize Vietnamese text using VITS.

    Args:
        text: Vietnamese text to synthesize.
        output_path: Path to save output audio.

    Returns:
        Dict with synthesis results.
    """
    # Try to load a VITS model for Vietnamese
    # This is a generic implementation - actual model may vary

    try:
        # Attempt 1: Try HuggingFace Transformers VITS
        from transformers import VitsModel, AutoTokenizer

        model_id = "facebook/mms-tts-vie"  # MMS TTS for Vietnamese
        model = VitsModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        device = get_device()
        model = model.to(device)

        # Tokenize and generate
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Save audio
        waveform = outputs.waveform[0].cpu().numpy()
        import soundfile as sf
        sf.write(output_path, waveform, model.config.sampling_rate)

        audio_duration = len(waveform) / model.config.sampling_rate

        return {
            "output_path": output_path,
            "audio_duration_s": round(audio_duration, 3),
            "device": device,
        }

    except Exception as e:
        # Attempt 2: Try standalone VITS
        console_print(f"  [dim]Trying alternative VITS implementation...[/]")

        try:
            from vits import utils
            from vits.models import SynthesizerTrn

            # Load model from checkpoint
            # This requires a pre-trained VITS checkpoint
            checkpoint_path = "path/to/vits_vietnamese.pth"
            config_path = "path/to/config.json"

            hps = utils.get_hparams_from_file(config_path)
            net = SynthesizerTrn(
                len(hps.symbols),
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **hps.model,
            ).to(device)

            _ = utils.load_checkpoint(checkpoint_path, net, None)
            _ = net.eval()

            # Get speaker ID
            sid = torch.LongTensor([0]).to(device)

            # Synthesize
            with torch.no_grad():
                x_tst = torch.LongTensor([1, 2, 3]).to(device)  # Placeholder
                x_tst_lengths = torch.LongTensor([3]).to(device)
                audio = net.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667)[0][0, 0].data.cpu().float().numpy()

            import soundfile as sf
            sf.write(output_path, audio, hps.data.sampling_rate)

            audio_duration = len(audio) / hps.data.sampling_rate

            return {
                "output_path": output_path,
                "audio_duration_s": round(audio_duration, 3),
                "device": device,
            }

        except Exception as e2:
            raise RuntimeError(f"VITS synthesis failed: {e} (fallback: {e2})")


def main(text: str, output_path: str) -> None:
    """Main entry point."""
    device = get_device()

    # Warmup (load model)
    console_print(f"  [dim]Loading VITS model...[/]")

    start = time.perf_counter()
    result = synthesize_vits_vi(text, output_path)
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

    parser = argparse.ArgumentParser(description="VITS Vietnamese inference")
    parser.add_argument("--text", required=True, help="Vietnamese text to synthesize")
    parser.add_argument("--output", required=True, help="Output audio path")
    parser.add_argument("--speaker", help="Speaker ID (not used)")
    args = parser.parse_args()

    main(args.text, args.output)

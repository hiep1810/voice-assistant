"""
Inference script for Gipformer 65M RNNT (G-Group AI Lab).
Uses sherpa-onnx for fast ONNX runtime inference.

Model: g-group-ai-lab/gipformer-65M-rnnt
Framework: Zipformer RNNT with ONNX runtime
Specialty: SOTA Vietnamese ASR optimized for edge devices (65M params).
"""

import json
import sys
import time

import soundfile as sf


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using soundfile."""
    info = sf.info(audio_path)
    return info.frames / info.samplerate


def main(audio_path: str) -> None:
    try:
        import sherpa_onnx
    except ImportError:
        print(json.dumps({"error": "sherpa-onnx not installed"}))
        sys.exit(1)

    audio_duration = get_audio_duration(audio_path)

    # Download model
    try:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download("g-group-ai-lab/gipformer-65M-rnnt")
    except Exception as e:
        print(json.dumps({"error": f"Failed to download model: {e}"}))
        sys.exit(1)

    # Create offline recognizer with transducer (RNNT)
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{model_dir}/encoder-epoch-35-avg-6.int8.onnx",
        decoder=f"{model_dir}/decoder-epoch-35-avg-6.int8.onnx",
        joiner=f"{model_dir}/joiner-epoch-35-avg-6.int8.onnx",
        tokens=f"{model_dir}/tokens.txt",
        num_threads=4,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
    )

    # Load audio
    audio, sample_rate = sf.read(audio_path, dtype="float32")

    # Resample if needed
    if sample_rate != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

    # Create stream
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, audio)

    # Warmup
    recognizer.decode_stream(stream)

    # Time inference
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, audio)

    start = time.perf_counter()
    recognizer.decode_stream(stream)
    inference_time = time.perf_counter() - start

    text = stream.result.text.strip() if stream.result.text else ""

    rtf = inference_time / audio_duration if audio_duration > 0 else float("inf")

    output = {
        "text": text,
        "language": "vi",
        "audio_duration_s": round(audio_duration, 3),
        "inference_time_s": round(inference_time, 3),
        "rtf": round(rtf, 4),
        "is_realtime": rtf < 1.0,
        "device": "cpu",
    }

    print(json.dumps(output, ensure_ascii=True))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_gipformer.py <audio_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

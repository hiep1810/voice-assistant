"""
Inference script for Moonshine Tiny Vietnamese (HuggingFace Transformers).
Runs INSIDE the moonshine venv — never import this from the main CLI.

Model: UsefulSensors/moonshine-tiny-vi
Framework: HuggingFace Transformers (sequence-to-sequence ASR)
Size: ~27M params — designed for edge/low-resource devices
"""

import json
import sys
import time

import torch


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using soundfile (available in the venv)."""
    import soundfile as sf

    info = sf.info(audio_path)
    return info.frames / info.samplerate


def main(audio_path: str) -> None:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import numpy as np

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    torch_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

    audio_duration = get_audio_duration(audio_path)

    # Load model and processor from HuggingFace
    model_id = "UsefulSensors/moonshine-tiny-vi"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    # Load and resample audio using soundfile
    import soundfile as sf
    import librosa

    # Load audio (automatically resample to 16kHz via librosa)
    # Why librosa? More robust resampling than manual torch code for a benchmark.
    waveform, sr = librosa.load(audio_path, sr=16000)
    waveform = torch.from_numpy(waveform).unsqueeze(0) # add batch/channel dim as 1xN if needed? 
                                                       # transformers prefers float32 or float16 numpy/pt
                                                       # for ASR pipelines it often expects a numpy array.


    # Process audio through the model
    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=sr, # Use the actual sr from librosa
        return_tensors="pt",
    ).to(device=device, dtype=torch_dtype)

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs)
        
    # Time only the inference
    start = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
    inference_time = time.perf_counter() - start

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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
        print("Usage: python run_moonshine.py <audio_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

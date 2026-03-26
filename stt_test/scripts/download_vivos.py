"""
Download VIVOS Vietnamese ASR test dataset.

Usage:
    pip install pandas pyarrow soundfile requests
    python stt_test/scripts/download_vivos.py --output-dir ./data/vivos --split test --limit 20
"""

import argparse
import io
import requests
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download


def get_parquet_filename(repo_id: str, split: str) -> str:
    """Get the actual parquet filename from the dataset API."""
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/data"
    response = requests.get(url)
    response.raise_for_status()

    files = response.json()
    for f in files:
        path = f.get("path", "")
        if path.startswith(f"data/{split}-") and path.endswith(".parquet"):
            return path
    raise ValueError(f"No parquet file found for split '{split}'")


def main():
    parser = argparse.ArgumentParser(description="Download VIVOS dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/vivos",
        help="Output directory for audio files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to download",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for testing)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Downloading VIVOS {args.split} split ===")

    # Get actual parquet filename
    print("Finding parquet file...")
    parquet_path = get_parquet_filename("ademax/vivos-vie-speech2text", args.split)
    print(f"Found: {parquet_path}")

    # Download parquet file
    print("Downloading parquet file...")
    parquet_file = hf_hub_download(
        repo_id="ademax/vivos-vie-speech2text",
        filename=parquet_path,
        repo_type="dataset",
    )

    # Read parquet
    print("Reading parquet...")
    df = pd.read_parquet(parquet_file)
    print(f"Found {len(df)} samples")

    # Process samples
    count = 0
    for idx, row in df.iterrows():
        if args.limit and count >= args.limit:
            break

        # Get audio data - stored as dict with 'bytes' key
        audio = row.get("audio")
        if audio is None:
            continue

        # Extract audio bytes and convert to numpy array
        if isinstance(audio, dict) and "bytes" in audio:
            audio_bytes = audio["bytes"]
            # Read WAV data using soundfile
            audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
        elif isinstance(audio, dict) and "array" in audio:
            audio_array = np.array(audio["array"])
            sampling_rate = audio.get("sampling_rate", 16000)
        else:
            continue

        # Save audio file
        audio_path = output_dir / f"{count:04d}.wav"
        sf.write(str(audio_path), audio_array, sampling_rate)

        # Save transcription
        text = row.get("transcription", "")
        text_path = output_dir / f"{count:04d}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(str(text))

        count += 1
        if count % 10 == 0:
            print(f"  Downloaded {count} samples...")

    print(f"\n=== Complete ===")
    print(f"Downloaded {count} samples to {output_dir}")
    print(f"Files: {count} .wav + {count} .txt")


if __name__ == "__main__":
    main()

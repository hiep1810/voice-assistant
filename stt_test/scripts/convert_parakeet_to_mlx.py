"""
Convert NVIDIA Parakeet Vietnamese model to MLX format.

Usage:
    source envs/parakeet/bin/activate
    python stt_test/scripts/convert_parakeet_to_mlx.py --output-dir ./parakeet-vietnamese-mlx
"""

import argparse
import json
import tarfile
from pathlib import Path

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download, upload_folder
from mlx.utils import tree_flatten, tree_unflatten
from safetensors.numpy import save_file


def download_nemo_model(model_id: str, cache_dir: str | None = None) -> Path:
    """Download NeMo model from HuggingFace."""
    print(f"Downloading {model_id}...")
    download_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.nemo", "config.json"],
    )
    return Path(download_path)


def extract_weights_and_config(model_path: Path) -> tuple[dict, dict]:
    """Extract weights and config from .nemo file."""
    nemo_files = list(model_path.glob("*.nemo"))
    if not nemo_files:
        raise FileNotFoundError(f"No .nemo file found in {model_path}")

    nemo_path = nemo_files[0]
    print(f"Extracting from {nemo_path.name}...")

    weights = {}
    config = {}

    with tarfile.open(nemo_path, "r") as tar:
        # Extract model weights
        for member in tar.getmembers():
            if member.name.endswith(".ckpt"):
                print(f"  Extracting {member.name}...")
                f = tar.extractfile(member)
                if f:
                    checkpoint = torch.load(f, map_location="cpu", weights_only=True)
                    if "state_dict" in checkpoint:
                        weights = checkpoint["state_dict"]
                        print(f"  Found {len(weights)} weight tensors")
                    else:
                        weights = {k: v for k, v in checkpoint.items() if hasattr(v, 'numpy') or hasattr(v, 'detach')}
                        print(f"  Found {len(weights)} tensors in checkpoint")

            # Extract config
            if member.name == "model_config.yaml":
                f = tar.extractfile(member)
                if f:
                    import yaml
                    config = yaml.safe_load(f)
                    print(f"  Loaded model config")

    return weights, config


def convert_weights_to_mlx(pytorch_weights: dict) -> dict:
    """Convert PyTorch weights to MLX format."""
    mlx_weights = {}

    for key, tensor in pytorch_weights.items():
        # Convert torch.Tensor to numpy then to mx.array
        if isinstance(tensor, torch.Tensor):
            numpy_array = tensor.cpu().numpy()
            mlx_weights[key] = numpy_array
        else:
            mlx_weights[key] = tensor

    return mlx_weights


def create_mlx_config(nemo_config: dict) -> dict:
    """Create MLX-compatible config from NeMo config."""
    # Map NeMo config to parakeet_mlx expected format
    mlx_config = {
        "target": nemo_config.get("target", "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE"),
        "model_defaults": nemo_config.get("model_defaults", {}),
    }

    # Add specific config fields parakeet_mlx expects
    if "encoder" in nemo_config:
        mlx_config["encoder"] = nemo_config["encoder"]
    if "decoder" in nemo_config:
        mlx_config["decoder"] = nemo_config["decoder"]
    if "preprocessor" in nemo_config:
        mlx_config["preprocessor"] = nemo_config["preprocessor"]

    return mlx_config


def main():
    parser = argparse.ArgumentParser(description="Convert Parakeet Vietnamese to MLX")
    parser.add_argument(
        "--model-id",
        type=str,
        default="nvidia/parakeet-ctc-0.6b-vi",
        help="Source NeMo model ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./parakeet-vietnamese-mlx",
        help="Output directory for converted model",
    )
    parser.add_argument(
        "--upload-to-hf",
        type=str,
        default=None,
        help="Upload to HuggingFace (provide repo ID)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloads",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download NeMo model
    print(f"\n=== Step 1: Downloading {args.model_id} ===")
    model_path = download_nemo_model(args.model_id, args.cache_dir)
    print(f"Downloaded to: {model_path}")

    # Step 2: Extract weights and config
    print(f"\n=== Step 2: Extracting weights ===")
    weights, config = extract_weights_and_config(model_path)

    # Step 3: Convert weights to MLX format
    print(f"\n=== Step 3: Converting to MLX format ===")
    mlx_weights = convert_weights_to_mlx(weights)
    print(f"Converted {len(mlx_weights)} tensors")

    # Step 4: Save weights as safetensors
    print(f"\n=== Step 4: Saving weights ===")
    save_file(mlx_weights, output_dir / "model.safetensors")
    print(f"Saved to {output_dir / 'model.safetensors'}")

    # Step 5: Save config
    print(f"\n=== Step 5: Saving config ===")
    mlx_config = create_mlx_config(config)
    with open(output_dir / "config.json", "w") as f:
        json.dump(mlx_config, f, indent=2)
    print(f"Saved to {output_dir / 'config.json'}")

    # Step 6: Upload to HF if requested
    if args.upload_to_hf:
        print(f"\n=== Step 6: Uploading to {args.upload_to_hf} ===")
        upload_folder(
            folder_path=str(output_dir),
            repo_id=args.upload_to_hf,
            repo_type="model",
        )
        print(f"Uploaded to https://huggingface.co/{args.upload_to_hf}")

    print("\n=== Conversion Complete ===")
    print(f"\nModel saved to: {output_dir.absolute()}")
    print("\nTo use the converted model:")
    print("  1. Copy the model to a HuggingFace repo")
    print("  2. Update run_parakeet_mac.py with the new model ID")
    print("  3. Test with: python -m stt_test benchmark test.wav --models parakeet")


if __name__ == "__main__":
    main()

"""
Model Registry — maps model names to their configuration.

Design Pattern: Registry Pattern
ELI5: A phone book for models — look up a name, get everything needed to run it.
Why here: 5 models with different deps/scripts. A dict lookup is cleaner than if/else.
"""

from __future__ import annotations

from dataclasses import dataclass, field


import sys

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single ASR model.

    Attributes:
        name: Short CLI-friendly name (e.g. "parakeet").
        display_name: Human-readable name for tables/output.
        packages: Pip packages for Windows/Linux (CUDA).
        script: Default inference script.
        huggingface_id: Model ID.
        mac_packages: Pip packages optimized for Apple Silicon (MLX/MPS).
        mac_script: specialized inference script for Mac.
        extra_pip_args: Default: [].
    """

    name: str
    display_name: str
    packages: list[str]
    script: str
    huggingface_id: str
    mac_packages: list[str] = None
    mac_script: str = None
    extra_pip_args: list[str] = field(default_factory=list)
    mac_extra_pip_args: list[str] = field(default_factory=list)

    def get_runtime_packages(self) -> list[str]:
        """Return the correct packages for the current OS."""
        if sys.platform == "darwin" and self.mac_packages:
            return self.mac_packages
        return self.packages

    def get_runtime_script(self) -> str:
        """Return the correct script for the current OS."""
        if sys.platform == "darwin" and self.mac_script:
            return self.mac_script
        return self.script

    def get_runtime_extra_pip_args(self) -> list[str]:
        """Return extra pip arguments (like CUDA indexes) for the current OS."""
        if sys.platform == "darwin":
            return self.mac_extra_pip_args
        # On Windows, we need the CUDA index for torch if it's in the packages
        if sys.platform == "win32" and "torch" in self.packages:
             return ["--extra-index-url", "https://download.pytorch.org/whl/cu124"]
        return self.extra_pip_args


MODELS: dict[str, ModelConfig] = {
    "parakeet": ModelConfig(
        name="parakeet",
        display_name="Parakeet 0.6B (NeMo/MLX)",
        packages=["nemo_toolkit[asr]", "torch", "torchaudio", "soundfile"],
        script="run_parakeet.py",
        huggingface_id="nvidia/parakeet-ctc-0.6b-vi",
        mac_packages=["parakeet-mlx", "mlx", "soundfile"],
        mac_script="run_parakeet_mac.py",
    ),
    "moonshine": ModelConfig(
        name="moonshine",
        display_name="Moonshine Tiny",
        packages=["transformers", "torch", "torchaudio", "soundfile", "librosa"],
        script="run_moonshine.py",
        huggingface_id="UsefulSensors/moonshine-tiny-vi",
    ),
    "qwen3-asr": ModelConfig(
        name="qwen3-asr",
        display_name="Qwen3-ASR 0.6B",
        packages=["qwen-asr", "soundfile", "torch", "torchaudio"],
        script="run_qwen3_asr.py",
        huggingface_id="Qwen/Qwen3-ASR-0.6B",
    ),
    "sensevoice": ModelConfig(
        name="sensevoice",
        display_name="UniASR Vietnamese (FunASR)",
        packages=["funasr", "torch", "torchaudio", "modelscope", "soundfile", "librosa", "omegaconf"],
        script="run_sensevoice.py",
        huggingface_id="iic/speech_UniASR_asr_2pass-vi-16k-common-vocab1001-pytorch-online",
    ),
    "whisper-turbo": ModelConfig(
        name="whisper-turbo",
        display_name="Whisper Large-v3-Turbo",
        packages=["transformers", "torch", "torchaudio", "soundfile", "librosa", "accelerate"],
        script="run_whisper_turbo.py",
        huggingface_id="openai/whisper-large-v3-turbo",
        mac_packages=["mlx-whisper", "soundfile", "librosa"],
        mac_script="run_whisper_mac.py",
    ),
    "gipformer": ModelConfig(
        name="gipformer",
        display_name="Gipformer 65M RNNT",
        packages=["k2", "kaldifeat", "onnxruntime", "torch", "soundfile", "librosa"],
        script="run_gipformer.py",
        huggingface_id="g-group-ai-lab/gipformer-65M-rnnt",
    ),
}


def get_model(name: str) -> ModelConfig:
    """Look up a model by its CLI name. Raises KeyError with a helpful message."""
    if name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise KeyError(f"Model '{name}' not found. Available: {available}")
    return MODELS[name]


def list_models() -> list[ModelConfig]:
    """Return all registered models (ordered by insertion)."""
    return list(MODELS.values())

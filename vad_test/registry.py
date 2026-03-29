"""
VAD Model Registry — maps model names to their configuration.

Design Pattern: Registry Pattern
ELI5: A phone book for VAD models — look up a name, get everything needed to run it.

Silero VAD is the primary model - language-agnostic and works great for Vietnamese.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VADModelConfig:
    """Configuration for a VAD model.

    Attributes:
        name: Short CLI-friendly name (e.g. "silero-vad").
        display_name: Human-readable name for tables/output.
        packages: Pip packages for Windows/Linux (CUDA).
        script: Default inference script.
        huggingface_id: Model ID on HuggingFace.
        mac_packages: Pip packages optimized for Apple Silicon.
        mac_script: Specialized inference script for Mac.
        extra_pip_args: Additional pip arguments.
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
        """Return extra pip arguments for the current OS."""
        if sys.platform == "darwin":
            return self.mac_extra_pip_args
        return self.extra_pip_args


VAD_MODELS: dict[str, VADModelConfig] = {
    "silero-vad": VADModelConfig(
        name="silero-vad",
        display_name="Silero VAD",
        packages=["torch", "torchaudio", "silero-vad"],
        script="run_silero_vad.py",
        huggingface_id="snakers4/silero-vad",
    ),
}


def get_vad_model(name: str) -> VADModelConfig:
    """Look up a VAD model by its CLI name. Raises KeyError with a helpful message."""
    if name not in VAD_MODELS:
        available = ", ".join(VAD_MODELS.keys())
        raise KeyError(f"VAD Model '{name}' not found. Available: {available}")
    return VAD_MODELS[name]


def list_vad_models() -> list[VADModelConfig]:
    """Return all registered VAD models (ordered by insertion)."""
    return list(VAD_MODELS.values())

"""
TTS Model Registry — maps TTS model names to their configuration.

Design Pattern: Registry Pattern
ELI5: A phone book for TTS models — look up a name, get everything needed to run it.
Why here: Multiple TTS models with different deps/scripts. A dict lookup is cleaner than if/else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sys


@dataclass(frozen=True)
class TTSModelConfig:
    """Configuration for a TTS model.

    Attributes:
        name: Short CLI-friendly name (e.g. "vietts").
        display_name: Human-readable name for tables/output.
        packages: Pip packages for Windows/Linux (CUDA).
        script: Default inference script.
        huggingface_id: Model ID or repo path.
        mac_packages: Pip packages optimized for Apple Silicon (MLX/MPS).
        mac_script: Specialized inference script for Mac.
        extra_pip_args: Default: [].
        mac_extra_pip_args: Extra pip arguments for Mac.
        requires_speaker: Whether the model requires a speaker/voice ID.
        default_text: Default Vietnamese text for testing.
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
    requires_speaker: bool = False
    default_text: str = "Xin chào, đây là bài kiểm tra giọng nói tiếng Việt."

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
        if sys.platform == "win32" and "torch" in self.packages:
            return ["--extra-index-url", "https://download.pytorch.org/whl/cu124"]
        return self.extra_pip_args


TTS_MODELS: dict[str, TTSModelConfig] = {
    "vietts": TTSModelConfig(
        name="vietts",
        display_name="VieTTS (MMS VITS)",
        packages=["torch", "torchaudio", "librosa", "pysptk", "pyworld", "jieba", "transformers", "soundfile"],
        script="run_vietts.py",
        huggingface_id="facebook/mms-tts-vie",
        requires_speaker=False,
        default_text="Xin chào, tôi là trợ lý ảo tiếng Việt.",
    ),
    "xtts-v2": TTSModelConfig(
        name="xtts-v2",
        display_name="Coqui XTTS-v2",
        packages=["TTS", "torch", "torchaudio", "librosa", "soundfile"],
        script="run_xtts.py",
        huggingface_id="coqui/XTTS-v2",
        requires_speaker=True,
        default_text="Xin chào, đây là bài kiểm tra giọng nói tiếng Việt.",
    ),
    "gpt-sovits": TTSModelConfig(
        name="gpt-sovits",
        display_name="GPT-SoVITS",
        packages=["torch", "torchaudio", "librosa", "soundfile", "transformers"],
        script="run_gpt_sovits.py",
        huggingface_id="RVC-Boss/GPT-SoVITS",
        requires_speaker=True,
        default_text="Xin chào, tôi có thể giúp gì cho bạn?",
    ),
    "vits-vi": TTSModelConfig(
        name="vits-vi",
        display_name="VITS Vietnamese (MMS)",
        packages=["torch", "torchaudio", "librosa", "soundfile", "transformers"],
        script="run_vits_vi.py",
        huggingface_id="facebook/mms-tts-vie",
        requires_speaker=False,
        default_text="Đây là hệ thống chuyển văn bản thành giọng nói.",
    ),
    "vietneu-tts": TTSModelConfig(
        name="vietneu-tts",
        display_name="VieNeu-TTS",
        packages=["vieneu"],
        script="run_vietneu_tts.py",
        huggingface_id="pnnbao-ump/VieNeu-TTS",
        requires_speaker=True,
        default_text="Xin chào, đây là bài kiểm tra giọng nói tiếng Việt.",
        extra_pip_args=["--extra-index-url", "https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/"],
        # Note: Requires Python 3.10+. Setup will be skipped if Python < 3.10
    ),
}


def get_tts_model(name: str) -> TTSModelConfig:
    """Look up a TTS model by its CLI name. Raises KeyError with a helpful message."""
    if name not in TTS_MODELS:
        available = ", ".join(TTS_MODELS.keys())
        raise KeyError(f"TTS Model '{name}' not found. Available: {available}")
    return TTS_MODELS[name]


def list_tts_models() -> list[TTSModelConfig]:
    """Return all registered TTS models (ordered by insertion)."""
    return list(TTS_MODELS.values())

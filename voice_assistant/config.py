"""Voice Assistant configuration and defaults."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    sampling_rate: int = 16000
    onset: float = 0.5
    offset: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    max_speech_duration_s: float = 30.0


@dataclass
class ASRConfig:
    """Automatic Speech Recognition configuration."""
    model: str = "zipformer"  # zipformer, whisper
    language: str = "vi"
    sample_rate: int = 16000


@dataclass
class LLMConfig:
    """LLM configuration for llama.cpp."""
    model: str = "qwen3-2b"
    model_path: Optional[str] = None  # Local GGUF path
    hf_id: Optional[str] = None  # HuggingFace ID
    context_length: int = 4096
    n_gpu_layers: int = 35  # Layers to offload to GPU
    n_threads: int = 8  # CPU threads
    flash_attention: bool = True

    # Remote server (optional) - if set, uses remote llama.cpp server instead of local
    server_url: Optional[str] = None  # e.g., "http://localhost:8000"
    server_api_key: Optional[str] = None  # API key if required


@dataclass
class TTSConfig:
    """Text-to-Speech configuration."""
    model: str = "vietneu-tts"  # vietneu-tts (VieNeu-TTS with LMDeploy)
    sample_rate: int = 24000  # VieNeu-TTS uses 24kHz
    speaker: Optional[str] = None  # Voice preset (neutrale, hanhphuc, etc.)
    backend: str = "lmdeploy"  # lmdeploy (fast) or standard
    use_gpu: bool = True


@dataclass
class AudioConfig:
    """Audio pipeline configuration."""
    input_device: Optional[int] = None  # Default input device
    output_device: Optional[int] = None  # Default output device
    chunk_size: int = 512  # Audio chunk size for streaming
    buffer_size: int = 2048  # Playback buffer size


@dataclass
class MemoryConfig:
    """Conversation memory configuration."""
    max_tokens: int = 4096  # Token budget for context
    max_turns: int = 20  # Maximum conversation turns to keep
    persistent: bool = False  # Persist conversation to disk
    persist_path: Optional[Path] = None


@dataclass
class ToolConfig:
    """Tool calling configuration."""
    enabled: bool = True
    allowed_tools: List[str] = field(default_factory=lambda: [
        "get_current_time",
        "get_date",
        "set_timer",
    ])


@dataclass
class VoiceAssistantConfig:
    """Main voice assistant configuration."""
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)

    # Platform
    device: str = "auto"  # auto, cuda, mps, cpu

    # TUI
    show_transcription: bool = True
    show_history: bool = True
    hotkeys_enabled: bool = True

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "VoiceAssistantConfig":
        """Load configuration from file or use defaults."""
        if config_path is None or not config_path.exists():
            return cls()

        import json
        with open(config_path, "r") as f:
            data = json.load(f)

        # Merge with defaults
        return cls(**data)

    def save(self, config_path: Path) -> None:
        """Save configuration to file."""
        import json
        from dataclasses import asdict

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# Default model registry
LLM_MODELS = {
    "qwen3-2b": {
        "hf_id": "Qwen/Qwen3-2B-GGUF",
        "file": "qwen3-2b-q4_k_m.gguf",
        "context": 32768,
        "tool_calling": True,
        "vietnamese": True,
    },
    "qwen3-0.6b": {
        "hf_id": "Qwen/Qwen3-0.6B-GGUF",
        "file": "qwen3-0.6b-q4_k_m.gguf",
        "context": 32768,
        "tool_calling": True,
        "vietnamese": True,
    },
    "lfm2-1.6b": {
        "hf_id": "LiquidAI/LFM2-1.6B-GGUF",
        "file": "lfm2-1.6b-q4_k_m.gguf",
        "context": 8192,
        "tool_calling": True,
        "vietnamese": False,
    },
}

VLM_MODELS = {
    "qwen3-vl-2b": {
        "hf_id": "Qwen/Qwen3-VL-2B-GGUF",
        "file": "qwen3-vl-2b-q4_k_m.gguf",
        "context": 8192,
    },
    "smolvlm": {
        "hf_id": "HuggingFaceTB/SmolVLM-500M-Instruct-GGUF",
        "file": "smolvlm-500m-q4_k_m.gguf",
        "context": 4096,
    },
}

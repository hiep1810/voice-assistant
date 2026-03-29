"""Interactive setup wizard for voice assistant components with model selection."""

import sys
import platform
import subprocess
import shutil
from typing import Optional, List, Dict, Any, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.markdown import Markdown

console = Console()


class SystemInfo:
    """System information detector."""

    @staticmethod
    def get_platform() -> str:
        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        return system

    @staticmethod
    def is_windows() -> bool:
        return sys.platform == "win32"

    @staticmethod
    def is_macos() -> bool:
        return sys.platform == "darwin"

    @staticmethod
    def is_linux() -> bool:
        return sys.platform == "linux"

    @staticmethod
    def get_python_version() -> str:
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    @staticmethod
    def has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return shutil.which("nvidia-smi") is not None

    @staticmethod
    def get_gpu_info() -> Optional[Dict[str, Any]]:
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "type": "cuda",
                    "name": torch.cuda.get_device_name(0),
                    "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    "compute_capability": f"{torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}",
                }
        except ImportError:
            pass

        if sys.platform == "darwin":
            try:
                import torch
                if torch.backends.mps.is_available():
                    return {
                        "type": "metal",
                        "name": "Apple Silicon (MPS)",
                    }
            except ImportError:
                pass

        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    return {
                        "type": "cuda",
                        "name": parts[0],
                        "memory_gb": float(parts[1].replace(" MiB", "")) / 1024,
                    }
            except Exception:
                pass

        return None

    @staticmethod
    def get_ram_gb() -> float:
        try:
            if sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulonglong = ctypes.c_ulonglong

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', c_ulonglong),
                        ('ullAvailPhys', c_ulonglong),
                        ('ullTotalPageFile', c_ulonglong),
                        ('ullAvailPageFile', c_ulonglong),
                        ('ullTotalVirtual', c_ulonglong),
                        ('ullAvailVirtual', c_ulonglong),
                        ('ullAvailExtendedVirtual', c_ulonglong),
                    ]

                status = MEMORYSTATUSEX()
                status.dwLength = ctypes.sizeof(status)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
                return status.ullTotalPhys / (1024**3)
            elif sys.platform == "darwin":
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    return int(result.stdout.strip()) / (1024**3)
            else:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            return kb / (1024**2)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def is_using_uv() -> bool:
        return "UV" in str(subprocess.__file__) or shutil.which("uv") is not None


# =============================================================================
# MODEL REGISTRIES - Available models for each component
# =============================================================================

ASR_MODELS = {
    "zipformer": {
        "name": "Zipformer (Sherpa-onnx)",
        "description": "Streaming ASR for Vietnamese, fast and accurate",
        "packages": ["sherpa-onnx>=1.10.0"],
        "recommended": True,
        "language": "Vietnamese",
        "streaming": True,
        "pros": ["True streaming (real-time)", "Optimized for Vietnamese", "Low latency (~100ms)", "Works on CPU"],
        "cons": ["Vietnamese only"],
        "best_for": "Voice assistant with live transcription",
        "disk_space": "~500MB",
        "ram_usage": "~1GB",
        "rtf": "0.1-0.2 (5-10x real-time)",
    },
    "whisper": {
        "name": "Whisper (OpenAI)",
        "description": "Multilingual ASR with high accuracy",
        "packages": ["openai-whisper", "torch", "torchaudio"],
        "recommended": False,
        "language": "Multilingual (incl. Vietnamese)",
        "streaming": False,
        "pros": ["Excellent accuracy", "99 languages supported", "Robust to noise"],
        "cons": ["Slower than Zipformer", "Not true streaming", "Requires GPU for good speed"],
        "best_for": "Batch transcription, multilingual support",
        "disk_space": "~2GB (tiny) to ~5GB (large)",
        "ram_usage": "~2-4GB",
        "rtf": "0.3-1.0 (varies by model size)",
    },
    "parakeet": {
        "name": "Parakeet (NVIDIA)",
        "description": "High accuracy Vietnamese ASR from NVIDIA NeMo",
        "packages": ["nemo-toolkit>=1.20.0", "cython", "numpy"],
        "recommended": False,
        "language": "Vietnamese",
        "streaming": False,
        "pros": ["State-of-the-art accuracy", "NVIDIA optimized", "Good for long-form"],
        "cons": ["Complex dependencies", "Heavy installation (~3GB)", "Requires CUDA"],
        "best_for": "Production accuracy, NVIDIA GPU systems",
        "disk_space": "~3GB",
        "ram_usage": "~4GB",
        "rtf": "0.2-0.5 (GPU required)",
        "note": "Requires isolated environment, complex setup",
    },
}

LLM_MODELS = {
    "qwen3-2b": {
        "name": "Qwen3 2B (GGUF)",
        "description": "Best balance of speed, quality, and Vietnamese support",
        "packages": ["llama-cpp-python>=0.2.50", "huggingface_hub>=0.20.0"],
        "recommended": True,
        "ram_required_gb": 8,
        "disk_space_gb": 4,
        "context": 32768,
        "vietnamese": True,
        "tool_calling": True,
        "pros": ["Excellent Vietnamese", "32K context", "Fast with GPU", "Tool calling support"],
        "cons": ["Requires 4GB disk", "8GB+ RAM recommended"],
        "best_for": "General voice assistant, Vietnamese conversations",
        "gpu_layers_recommended": 35,
        "tokens_per_sec_gpu": "~50",
        "tokens_per_sec_cpu": "~5",
    },
    "qwen3-0.6b": {
        "name": "Qwen3 0.6B (GGUF)",
        "description": "Lightweight, fast, good for testing and low-end systems",
        "packages": ["llama-cpp-python>=0.2.50", "huggingface_hub>=0.20.0"],
        "recommended": False,
        "ram_required_gb": 4,
        "disk_space_gb": 1,
        "context": 32768,
        "vietnamese": True,
        "tool_calling": True,
        "pros": ["Very fast", "Low RAM usage", "Small disk footprint", "Vietnamese support"],
        "cons": ["Less capable reasoning", "Shorter responses"],
        "best_for": "Low-end systems, quick testing, simple Q&A",
        "gpu_layers_recommended": 20,
        "tokens_per_sec_gpu": "~100",
        "tokens_per_sec_cpu": "~15",
    },
    "lfm2-1.6b": {
        "name": "Liquid LFM2 1.6B (GGUF)",
        "description": "Strong reasoning, optimized for English",
        "packages": ["llama-cpp-python>=0.2.50", "huggingface_hub>=0.20.0"],
        "recommended": False,
        "ram_required_gb": 6,
        "disk_space_gb": 2,
        "context": 8192,
        "vietnamese": False,
        "tool_calling": True,
        "pros": ["Better reasoning than Qwen", "Efficient architecture", "Good English"],
        "cons": ["No Vietnamese support", "Smaller context (8K)"],
        "best_for": "English conversations, logic tasks",
        "gpu_layers_recommended": 28,
        "tokens_per_sec_gpu": "~60",
        "tokens_per_sec_cpu": "~8",
    },
    "remote": {
        "name": "Remote llama.cpp Server",
        "description": "Connect to external server - no local resources needed",
        "packages": ["requests>=2.28.0"],
        "recommended": False,
        "ram_required_gb": 0,
        "disk_space_gb": 0,
        "context": "Depends on server",
        "vietnamese": "Depends on server model",
        "pros": ["Zero local resources", "Works on any device", "Server handles heavy lifting"],
        "cons": ["Requires separate server setup", "Network dependency", "Latency from HTTP"],
        "best_for": "Low-RAM systems, shared LLM across devices",
        "note": "Run server: llama-server.exe -m model.gguf --port 8000",
    },
}

TTS_MODELS = {
    "vietneu-tts": {
        "name": "VieNeu-TTS (LMDeploy)",
        "description": "Fast Vietnamese TTS with LMDeploy acceleration and 6 voice presets",
        "packages": ["vieneu>=0.1.0"],
        "recommended": True,
        "language": "Vietnamese",
        "voices": 6,
        "voice_names": ["neutrale", "hanhphuc", "leloi", "nguyentruothanh", "chihanh", "khanhlinh"],
        "backend": "lmdeploy",
        "pros": ["5x real-time (RTF=0.20)", "6 natural Vietnamese voices", "Voice cloning support", "LMDeploy acceleration"],
        "cons": ["Requires GPU for best speed", "Vietnamese only"],
        "best_for": "Natural Vietnamese speech, voice variety",
        "disk_space": "~1GB",
        "ram_usage": "~2GB",
        "rtf_gpu": "0.20 (5x real-time)",
        "rtf_cpu": "1.5 (slower than real-time)",
    },
    "vietts": {
        "name": "VietTTS (Facebook MMS)",
        "description": "Lightweight Vietnamese TTS from Facebook MMS - fastest option",
        "packages": ["transformers>=4.30.0", "torch", "torchaudio", "soundfile"],
        "recommended": False,
        "language": "Vietnamese",
        "voices": 1,
        "backend": "pytorch",
        "pros": ["49x real-time (RTF=0.02)", "Very lightweight", "Works on CPU", "Simple setup"],
        "cons": ["Single voice", "Less natural than VieNeu"],
        "best_for": "Fast responses, low-resource systems",
        "disk_space": "~500MB",
        "ram_usage": "~1GB",
        "rtf_gpu": "0.02 (49x real-time)",
        "rtf_cpu": "0.1 (10x real-time)",
    },
    "xtts-v2": {
        "name": "XTTS-v2 (Coqui)",
        "description": "High-quality multilingual TTS with voice cloning",
        "packages": ["TTS>=0.20.0", "torch", "pandas"],
        "recommended": False,
        "language": "Multilingual (Vietnamese via fallback)",
        "voices": "Cloning + presets",
        "backend": "pytorch",
        "pros": ["Voice cloning (3-5s sample)", "High quality", "17 languages"],
        "cons": ["Slower (RTF=0.5)", "Vietnamese uses MMS fallback", "Heavy dependencies"],
        "best_for": "Voice cloning, multilingual support",
        "disk_space": "~2GB",
        "ram_usage": "~3GB",
        "rtf_gpu": "0.5 (2x real-time)",
        "rtf_cpu": "2.0 (slower than real-time)",
        "note": "Vietnamese uses MMS fallback, not native XTTS",
    },
}

AUDIO_PACKAGES = {
    "name": "Audio I/O",
    "description": "Microphone input and speaker output (required for voice interaction)",
    "packages": ["pyaudio>=0.2.13", "sounddevice>=0.4.6", "numpy>=1.20.0"],
    "required_for": [
        "Microphone input for ASR",
        "Speaker output for TTS playback",
        "Real-time audio streaming",
    ],
    "note": "Not needed for text-only CLI mode",
}

VAD_PACKAGES = {
    "name": "VAD (Voice Activity Detection)",
    "description": "Detects when you start/stop speaking",
    "packages": ["silero-vad>=0.4.0"],
    "required_for": [
        "Auto-start transcription when you speak",
        "Filter silence and background noise",
        "Real-time voice detection",
    ],
    "recommended": True,
}

ASR_PACKAGES = {
    "name": "ASR (Speech-to-Text)",
    "description": "Transcribes speech to text",
    "recommended": True,
}

TUI_PACKAGES = {
    "name": "TUI (Terminal UI)",
    "description": "Rich terminal interface with live transcription display",
    "packages": ["rich>=13.0", "typer>=0.9"],
    "required_for": [
        "Live transcription display",
        "Interactive interface with status panels",
        "Hotkey controls (M, V, S, Q)",
    ],
    "recommended": True,
}

VISION_PACKAGES = {
    "name": "Vision (VLM)",
    "description": "Image and screen analysis with vision language models",
    "packages": ["opencv-python>=4.9.0", "pillow>=10.0.0"],
    "required_for": [
        "Screen capture and analysis",
        "Camera input for VLM",
        "Image file analysis",
    ],
    "recommended": False,
}


# =============================================================================
# SETUP WIZARD
# =============================================================================

class SetupWizard:
    """Interactive setup wizard with model selection."""

    def __init__(self):
        self.sys_info = SystemInfo()
        self.platform = self.sys_info.get_platform()
        self.gpu_info = self.sys_info.get_gpu_info()
        self.ram_gb = self.sys_info.get_ram_gb()
        self.selected_models = {}

    def detect_environment(self) -> Dict[str, Any]:
        info = {
            "platform": self.platform,
            "python_version": self.sys_info.get_python_version(),
            "gpu": self.gpu_info,
            "ram_gb": self.ram_gb,
            "using_uv": self.sys_info.is_using_uv(),
        }

        if info["gpu"]:
            gpu_type = info["gpu"]["type"]
            if gpu_type == "cuda":
                info["recommended_backend"] = "cuda"
                info["llm_gpu_layers"] = 35
            elif gpu_type == "metal":
                info["recommended_backend"] = "metal"
                info["llm_gpu_layers"] = 35
            else:
                info["recommended_backend"] = "cpu"
                info["llm_gpu_layers"] = 0
        else:
            info["recommended_backend"] = "cpu"
            info["llm_gpu_layers"] = 0

        if info["ram_gb"] >= 16:
            info["recommended_llm"] = "qwen3-2b"
        elif info["ram_gb"] >= 8:
            info["recommended_llm"] = "qwen3-2b"
        else:
            info["recommended_llm"] = "qwen3-0.6b"

        return info

    def print_system_info(self, info: Dict[str, Any]) -> None:
        console.print(Panel.fit(
            "[bold]Your System[/bold]",
            border_style="cyan",
        ))

        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        platform_display = f"{info['platform'].capitalize()} ({platform.machine()})"
        table.add_row("Platform", platform_display)
        table.add_row("Python", info["python_version"])
        table.add_row("RAM", f"{info['ram_gb']:.1f} GB")

        if info["gpu"]:
            gpu_name = info["gpu"]["name"]
            table.add_row("GPU", f"[green]{gpu_name}[/green]")
            if info["gpu"]["type"] == "cuda":
                vram = info["gpu"].get("memory_gb", 0)
                table.add_row("CUDA", f"[green]Yes ({vram:.1f} GB VRAM)[/green]")
            elif info["gpu"]["type"] == "metal":
                table.add_row("Metal", "[green]Yes (Apple Silicon)[/green]")
        else:
            table.add_row("GPU", "[yellow]Not detected (CPU mode)[/yellow]")

        table.add_row("Package Manager", "[green]uv[/green]" if info["using_uv"] else "[yellow]pip[/yellow]")

        console.print(table)
        console.print("")

    def _show_hardware_recommendations(self, info: Dict[str, Any]) -> None:
        """Show hardware-based recommendations."""
        console.print(Panel.fit(
            "[bold]Recommendations for Your System[/bold]",
            border_style="green",
            title="Hardware Analysis",
        ))

        ram_gb = info.get("ram_gb", 0)
        gpu = info.get("gpu")
        gpu_type = gpu.get("type") if gpu else None
        vram = gpu.get("memory_gb", 0) if gpu else 0

        recommendations = []

        # LLM recommendation
        if gpu:
            if gpu_type == "cuda" and vram >= 8:
                recommendations.append("[green]LLM:[/green] Qwen3 2B with full GPU offload (35 layers)")
            elif gpu_type == "cuda" and vram >= 4:
                recommendations.append("[green]LLM:[/green] Qwen3 2B with partial GPU offload (20-25 layers)")
            elif gpu_type == "metal":
                recommendations.append("[green]LLM:[/green] Qwen3 2B with Metal acceleration")
            else:
                recommendations.append("[yellow]LLM:[/yellow] Qwen3 0.6B (limited VRAM)")
        else:
            if ram_gb >= 16:
                recommendations.append("[yellow]LLM:[/yellow] Qwen3 2B (CPU mode, slower)")
            elif ram_gb >= 8:
                recommendations.append("[yellow]LLM:[/yellow] Qwen3 0.6B recommended")
            else:
                recommendations.append("[red]LLM:[/red] Use Remote Server (insufficient RAM)")

        # TTS recommendation
        if gpu:
            recommendations.append("[green]TTS:[/green] VieNeu-TTS with LMDeploy (5x real-time)")
        else:
            recommendations.append("[yellow]TTS:[/yellow] VietTTS (MMS) for CPU speed")

        # ASR recommendation
        recommendations.append("[green]ASR:[/green] Zipformer (streaming, Vietnamese optimized)")

        # Audio/TUI
        recommendations.append("[green]Audio I/O:[/green] Required for voice interaction")
        recommendations.append("[green]TUI:[/green] Recommended for live display")

        for rec in recommendations:
            console.print(f"  {rec}")

        console.print("")

        # Show quick selection option
        if Confirm.ask("Use recommended settings?", default=True):
            self.selected_models["asr"] = "zipformer"
            if gpu and gpu_type == "cuda" and vram >= 4:
                self.selected_models["llm"] = "qwen3-2b"
            elif gpu and gpu_type == "metal":
                self.selected_models["llm"] = "qwen3-2b"
            elif ram_gb >= 8:
                self.selected_models["llm"] = "qwen3-0.6b"
            else:
                self.selected_models["llm"] = "remote"

            if gpu:
                self.selected_models["tts"] = "vietneu-tts"
            else:
                self.selected_models["tts"] = "vietts"

            console.print("[green]Using recommended settings.[/green]")
            console.print("")
            return  # Skip individual selection

        console.print("[dim]Proceeding to manual selection...[/dim]\n")

    def select_asr_model(self) -> str:
        """Select ASR model."""
        console.print(Panel.fit(
            "[bold]Step 1: Select ASR (Speech-to-Text) Model[/bold]\n\n"
            "This transcribes Vietnamese speech to text.",
            border_style="blue",
        ))
        console.print("")

        table = Table(title="Available ASR Models")
        table.add_column("#", style="cyan", width=2)
        table.add_column("Model", style="green bold")
        table.add_column("Best For", style="white")
        table.add_column("Pros", style="yellow")
        table.add_column("Requirements", style="dim")

        models = list(ASR_MODELS.keys())
        for i, key in enumerate(models, 1):
            model = ASR_MODELS[key]
            best_for = model.get("best_for", "")
            pros = ", ".join(model.get("pros", [])[:2])
            disk = model.get("disk_space", "~1GB")
            ram = model.get("ram_usage", "~1GB")

            # Recommendation badge
            rec_badge = " [green][REC][/green]" if model.get("recommended") else ""

            table.add_row(
                str(i),
                model["name"] + rec_badge,
                best_for,
                pros[:50] + "..." if len(pros) > 50 else pros,
                f"Disk: {disk}, RAM: {ram}",
            )

        console.print(table)
        console.print("")

        # Show details for recommended model
        rec_model = next((k for k, v in ASR_MODELS.items() if v.get("recommended")), None)
        if rec_model:
            console.print(Panel.fit(
                f"[bold]Recommended: {ASR_MODELS[rec_model]['name']}[/bold]\n\n"
                f"RTF: {ASR_MODELS[rec_model].get('rtf', 'N/A')}\n"
                f"Pros: {', '.join(ASR_MODELS[rec_model].get('pros', []))}\n"
                f"Cons: {', '.join(ASR_MODELS[rec_model].get('cons', []))}",
                border_style="green",
                title="[green]Recommendation[/green]",
            ))
            console.print("")

        default = "1"
        choice = Prompt.ask("Select ASR model", default=default)

        selected_key = models[int(choice) - 1]
        self.selected_models["asr"] = selected_key

        console.print(f"[green]Selected:[/green] {ASR_MODELS[selected_key]['name']}")
        console.print("")

        return selected_key

    def select_llm_model(self) -> str:
        """Select LLM model."""
        console.print(Panel.fit(
            "[bold]Step 2: Select LLM (Language Model)[/bold]\n\n"
            "This generates AI responses. Choose local or remote server.",
            border_style="blue",
        ))
        console.print("")

        table = Table(title="Available LLM Models")
        table.add_column("#", style="cyan", width=2)
        table.add_column("Model", style="green bold")
        table.add_column("Best For", style="white")
        table.add_column("Vietnamese", style="yellow")
        table.add_column("Speed (GPU/CPU)", style="dim")
        table.add_column("Requirements", style="dim")

        models = list(LLM_MODELS.keys())
        for i, key in enumerate(models, 1):
            model = LLM_MODELS[key]
            best_for = model.get("best_for", "")
            vietnamese = "[green]Yes[/green]" if model.get("vietnamese") else "[red]No[/red]"
            speed_gpu = model.get("tokens_per_sec_gpu", "~")
            speed_cpu = model.get("tokens_per_sec_cpu", "~")
            ram = model.get("ram_required_gb", 0)
            disk = model.get("disk_space_gb", 0)

            # Recommendation badge
            rec_badge = " [green][REC][/green]" if model.get("recommended") else ""

            table.add_row(
                str(i),
                model["name"] + rec_badge,
                best_for,
                vietnamese,
                f"{speed_gpu} / {speed_cpu} tok/s",
                f"RAM: {ram}GB, Disk: {disk}GB",
            )

        console.print(table)
        console.print("")

        # Show details for recommended model based on system
        info = self.detect_environment()
        rec_model = info.get("recommended_llm", "qwen3-2b")
        if rec_model in LLM_MODELS:
            m = LLM_MODELS[rec_model]
            console.print(Panel.fit(
                f"[bold]Recommended for your system: {m['name']}[/bold]\n\n"
                f"Pros: {', '.join(m.get('pros', []))}\n"
                f"Cons: {', '.join(m.get('cons', []))}\n"
                f"GPU Layers: {m.get('gpu_layers_recommended', 'N/A')}",
                border_style="green",
                title="[green]System Recommendation[/green]",
            ))
            console.print("")

        # Recommend based on RAM
        default_idx = models.index(rec_model) + 1 if rec_model in models else 1

        choice = Prompt.ask("Select LLM model", default=str(default_idx))

        selected_key = models[int(choice) - 1]
        self.selected_models["llm"] = selected_key

        console.print(f"[green]Selected:[/green] {LLM_MODELS[selected_key]['name']}")
        console.print("")

        return selected_key

    def select_tts_model(self) -> str:
        """Select TTS model."""
        console.print(Panel.fit(
            "[bold]Step 3: Select TTS (Text-to-Speech) Model[/bold]\n\n"
            "This synthesizes Vietnamese voice output.",
            border_style="blue",
        ))
        console.print("")

        table = Table(title="Available TTS Models")
        table.add_column("#", style="cyan", width=2)
        table.add_column("Model", style="green bold")
        table.add_column("Best For", style="white")
        table.add_column("Speed (RTF)", style="yellow")
        table.add_column("Voices", style="dim")
        table.add_column("Requirements", style="dim")

        models = list(TTS_MODELS.keys())
        for i, key in enumerate(models, 1):
            model = TTS_MODELS[key]
            best_for = model.get("best_for", "")
            rtf_gpu = model.get("rtf_gpu", "N/A")
            rtf_cpu = model.get("rtf_cpu", "N/A")
            voices = model.get("voices", "1")
            voice_names = model.get("voice_names", [])
            if voice_names:
                voices = f"{voices}: {', '.join(voice_names[:3])}"
                if len(voice_names) > 3:
                    voices += "..."
            disk = model.get("disk_space", "~1GB")
            ram = model.get("ram_usage", "~1GB")

            # Recommendation badge
            rec_badge = " [green][REC][/green]" if model.get("recommended") else ""

            table.add_row(
                str(i),
                model["name"] + rec_badge,
                best_for,
                f"GPU: {rtf_gpu}, CPU: {rtf_cpu}",
                voices,
                f"Disk: {disk}, RAM: {ram}",
            )

        console.print(table)
        console.print("")

        # Show details for recommended model
        rec_model = next((k for k, v in TTS_MODELS.items() if v.get("recommended")), None)
        if rec_model:
            m = TTS_MODELS[rec_model]
            console.print(Panel.fit(
                f"[bold]Recommended: {m['name']}[/bold]\n\n"
                f"Voice Presets: {', '.join(m.get('voice_names', []))}\n"
                f"Pros: {', '.join(m.get('pros', []))}\n"
                f"Cons: {', '.join(m.get('cons', []))}",
                border_style="green",
                title="[green]Recommendation[/green]",
            ))
            console.print("")

        default = "1"
        choice = Prompt.ask("Select TTS model", default=default)

        selected_key = models[int(choice) - 1]
        self.selected_models["tts"] = selected_key

        console.print(f"[green]Selected:[/green] {TTS_MODELS[selected_key]['name']}")
        console.print("")

        return selected_key

    def select_components(self) -> List[str]:
        """Select which component categories to install."""
        console.print(Panel.fit(
            "[bold]Step 4: Select Component Categories[/bold]\n\n"
            "Choose which parts of the voice assistant to install.",
            border_style="green",
        ))
        console.print("")

        table = Table(title="Component Categories")
        table.add_column("#", style="cyan", width=2)
        table.add_column("Category", style="green bold")
        table.add_column("Purpose", style="white")
        table.add_column("Needed For", style="yellow")

        components = [
            ("vad", "VAD", "Voice activity detection", "Auto speech detection"),
            ("asr", "ASR", "Speech-to-text", "Transcription"),
            ("llm", "LLM", "Language model", "AI responses"),
            ("tts", "TTS", "Text-to-speech", "Voice output"),
            ("audio", "Audio I/O", "Mic/speaker drivers", "Physical audio devices"),
            ("tui", "TUI", "Terminal interface", "Live display"),
            ("vision", "Vision", "Image/screen analysis", "VLM features"),
        ]

        for i, (key, name, desc, req) in enumerate(components, 1):
            required = "[green]Yes[/green]" if key in ["audio", "tui"] else "[yellow]No[/yellow]"
            table.add_row(str(i), name, desc, required)

        console.print(table)
        console.print("")

        # Explain Audio I/O
        console.print(Panel.fit(
            "[bold]What is Audio I/O?[/bold]\n\n"
            "Audio I/O (Input/Output) handles physical audio devices:\n"
            "- [cyan]Microphone input[/cyan] - Captures your voice for ASR\n"
            "- [cyan]Speaker output[/cyan] - Plays TTS responses\n"
            "- [cyan]Packages:[/cyan] pyaudio, sounddevice, numpy\n\n"
            "[dim]Required for: Full voice assistant with mic/speaker\n"
            "Not needed for: Text-only CLI mode[/dim]",
            border_style="dim",
            title="Audio I/O Explanation",
        ))
        console.print("")

        console.print("[dim]Default: Full installation (components 1-6, skip vision)[/dim]")
        default = "1,2,3,4,5,6"
        selection = Prompt.ask("Select components (comma-separated)", default=default)

        selected = []
        for num in selection.split(","):
            num = num.strip()
            if num.isdigit() and 1 <= int(num) <= len(components):
                selected.append(components[int(num) - 1][0])

        console.print(f"[green]Selected:[/green] {', '.join(selected)}")
        console.print("")

        return selected

    def get_packages(self, components: List[str]) -> List[str]:
        """Get all packages to install."""
        packages = set()

        # Add Audio I/O packages if audio component selected
        if "audio" in components:
            packages.update(AUDIO_PACKAGES["packages"])

        # Add TUI packages if TUI component selected
        if "tui" in components:
            packages.update(TUI_PACKAGES["packages"])

        # Add VAD packages if VAD component selected
        if "vad" in components:
            packages.update(VAD_PACKAGES["packages"])

        # Add ASR model packages
        if "asr" in components and self.selected_models.get("asr"):
            asr_key = self.selected_models["asr"]
            packages.update(ASR_MODELS[asr_key]["packages"])

        # Add LLM model packages
        if "llm" in components and self.selected_models.get("llm"):
            llm_key = self.selected_models["llm"]
            if llm_key != "remote":
                packages.update(LLM_MODELS[llm_key]["packages"])
            else:
                packages.update(LLM_MODELS[llm_key]["packages"])

        # Add TTS model packages
        if "tts" in components and self.selected_models.get("tts"):
            tts_key = self.selected_models["tts"]
            packages.update(TTS_MODELS[tts_key]["packages"])

        # Add Vision packages if selected
        if "vision" in components:
            packages.update(VISION_PACKAGES["packages"])

        return list(packages)

    def install_packages(self, packages: List[str], using_uv: bool) -> bool:
        """Install packages."""
        if not packages:
            console.print("[yellow]No packages to install[/yellow]")
            return True

        console.print(Panel.fit(
            f"[bold]Installing {len(packages)} packages...[/bold]\n\n"
            + "\n".join(packages[:10])
            + ("\n..." if len(packages) > 10 else ""),
            border_style="yellow",
        ))

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Installing...", total=None)

                if using_uv:
                    cmd = ["uv", "pip", "install"]
                else:
                    cmd = [sys.executable, "-m", "pip", "install"]

                cmd.extend(packages)

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

                progress.update(task, completed=True)

                if result.returncode == 0:
                    console.print("[green]All packages installed![/green]")
                    return True
                else:
                    console.print(f"[red]Installation failed:[/red]")
                    console.print(result.stderr[:1000])
                    return False

        except subprocess.TimeoutExpired:
            console.print("[red]Installation timed out[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return False

    def generate_config(self, info: Dict[str, Any], components: List[str]) -> str:
        """Generate config file."""
        import json

        config = {
            "_comment": "Generated by voice-assistant setup wizard",
            "llm": {
                "model": self.selected_models.get("llm", "qwen3-2b"),
            },
        }

        if info["gpu"] and self.selected_models.get("llm") != "remote":
            config["llm"]["n_gpu_layers"] = info.get("llm_gpu_layers", 35)

        if self.selected_models.get("tts"):
            config["tts"] = {
                "model": self.selected_models["tts"],
                "use_gpu": info["gpu"] is not None,
            }
            if self.selected_models["tts"] == "vietneu-tts":
                config["tts"]["backend"] = "lmdeploy"
                config["tts"]["speaker"] = "neutrale"

        if self.selected_models.get("asr"):
            config["asr"] = {
                "model": self.selected_models["asr"],
                "language": "vi",
            }

        if "audio" in components:
            config["audio"] = {
                "chunk_size": 512,
                "buffer_size": 2048,
            }

        config_path = "voice_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return config_path

    def show_next_steps(self, config_path: str) -> None:
        """Show next steps."""
        console.print(Panel.fit(
            "[bold]Setup Complete! Next Steps:[/bold]",
            border_style="green",
        ))

        llm_model = self.selected_models.get("llm", "qwen3-2b")

        if llm_model == "remote":
            console.print("""
[bold]1. Start llama.cpp Server[/bold]
   [cyan]llama-server.exe -m qwen3-2b-q4_k_m.gguf --port 8000[/cyan]

[bold]2. Run Voice Assistant[/bold]
   [cyan]uv run python -m voice_assistant cli --server-url http://localhost:8000[/cyan]
""")
        else:
            console.print(f"""
[bold]1. Run Voice Assistant[/bold]

   CLI mode (text-only):
   [cyan]uv run python -m voice_assistant cli -c {config_path}[/cyan]

   TUI mode (full interface):
   [cyan]uv run python -m voice_assistant run -c {config_path}[/cyan]

[bold]Model:[/bold] {llm_model} (downloaded on first use)
""")

        console.print("""
[bold]Available Commands:[/bold]
  [cyan]voice-assistant cli[/cyan]     - Text-only chat
  [cyan]voice-assistant run[/cyan]     - Full TUI interface
  [cyan]voice-assistant list-models[/cyan] - Show models
""")

    def run(self) -> int:
        """Run the setup wizard."""
        console.print(Panel.fit(
            "[bold cyan]Voice Assistant Setup Wizard[/bold cyan]\n\n"
            "Select and install components with specific models:\n"
            "- ASR: Zipformer, Whisper, Parakeet\n"
            "- LLM: Qwen3, LFM2, or Remote Server\n"
            "- TTS: VieNeu-TTS, VietTTS, XTTS-v2\n"
            "- Audio I/O: Microphone and speaker support\n"
            "- TUI: Rich terminal interface",
            border_style="cyan",
        ))
        console.print("")

        # Detect system
        console.print("[bold]Detecting system...[/bold]")
        info = self.detect_environment()
        self.print_system_info(info)

        # Show hardware-based recommendations
        self._show_hardware_recommendations(info)

        # Step 1: Select ASR model (skip if already selected via recommendations)
        if "asr" not in self.selected_models:
            self.select_asr_model()

        # Step 2: Select LLM model (skip if already selected via recommendations)
        if "llm" not in self.selected_models:
            self.select_llm_model()

        # Step 3: Select TTS model (skip if already selected via recommendations)
        if "tts" not in self.selected_models:
            self.select_tts_model()

        # Step 4: Select components
        components = self.select_components()

        # Confirm
        console.print("")
        console.print(Panel.fit(
            "[bold]Summary[/bold]\n\n"
            f"ASR: {ASR_MODELS[self.selected_models.get('asr', 'zipformer')]['name']}\n"
            f"LLM: {LLM_MODELS[self.selected_models.get('llm', 'qwen3-2b')]['name']}\n"
            f"TTS: {TTS_MODELS[self.selected_models.get('tts', 'vietneu-tts')]['name']}\n"
            f"Components: {', '.join(components)}",
            border_style="green",
        ))

        if not Confirm.ask("\nContinue with installation?", default=True):
            console.print("[yellow]Setup cancelled.[/yellow]")
            return 1

        # Get packages
        packages = self.get_packages(components)

        # Install
        success = self.install_packages(packages, info["using_uv"])

        # Generate config
        config_path = self.generate_config(info, components)
        console.print(f"[green]Config saved:[/green] {config_path}")

        # Show next steps
        self.show_next_steps(config_path)

        return 0 if success else 1


def main():
    """Main entry point."""
    wizard = SetupWizard()
    sys.exit(wizard.run())


if __name__ == "__main__":
    main()

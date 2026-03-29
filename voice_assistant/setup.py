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
    },
    "whisper": {
        "name": "Whisper (OpenAI)",
        "description": "Multilingual ASR, good accuracy, slower",
        "packages": ["openai-whisper", "torch"],
        "recommended": False,
        "language": "Multilingual",
        "streaming": False,
    },
    "parakeet": {
        "name": "Parakeet (NVIDIA)",
        "description": "High accuracy Vietnamese ASR from NVIDIA",
        "packages": ["nemo-toolkit", "cython"],
        "recommended": False,
        "language": "Vietnamese",
        "streaming": False,
        "note": "Requires complex setup, isolated environment",
    },
}

LLM_MODELS = {
    "qwen3-2b": {
        "name": "Qwen3 2B (GGUF)",
        "description": "Best balance of speed and quality, Vietnamese support",
        "packages": ["llama-cpp-python>=0.2.50", "huggingface_hub>=0.20.0"],
        "recommended": True,
        "ram_required_gb": 8,
        "context": 32768,
        "vietnamese": True,
    },
    "qwen3-0.6b": {
        "name": "Qwen3 0.6B (GGUF)",
        "description": "Fast, lightweight, good for testing",
        "packages": ["llama-cpp-python>=0.2.50", "huggingface_hub>=0.20.0"],
        "recommended": False,
        "ram_required_gb": 4,
        "context": 32768,
        "vietnamese": True,
    },
    "lfm2-1.6b": {
        "name": "Liquid LFM2 1.6B (GGUF)",
        "description": "Good reasoning, no Vietnamese support",
        "packages": ["llama-cpp-python>=0.2.50", "huggingface_hub>=0.20.0"],
        "recommended": False,
        "ram_required_gb": 6,
        "context": 8192,
        "vietnamese": False,
    },
    "remote": {
        "name": "Remote llama.cpp Server",
        "description": "Connect to external server (no local LLM)",
        "packages": ["requests>=2.28.0"],
        "recommended": False,
        "ram_required_gb": 0,
        "note": "Requires llama.cpp server running separately",
    },
}

TTS_MODELS = {
    "vietneu-tts": {
        "name": "VieNeu-TTS (LMDeploy)",
        "description": "Fast Vietnamese TTS with 6 voice presets, 5x real-time",
        "packages": ["vieneu>=0.1.0"],
        "recommended": True,
        "language": "Vietnamese",
        "voices": 6,
        "realtime_factor": 0.2,
    },
    "vietts": {
        "name": "VietTTS (Facebook MMS)",
        "description": "Lightweight Vietnamese TTS from Facebook MMS, 49x real-time",
        "packages": ["transformers>=4.30.0", "torch", "torchaudio"],
        "recommended": False,
        "language": "Vietnamese",
        "voices": 1,
        "realtime_factor": 0.02,
    },
    "xtts-v2": {
        "name": "XTTS-v2 (Coqui)",
        "description": "High quality with voice cloning, slower",
        "packages": ["TTS>=0.20.0"],
        "recommended": False,
        "language": "Multilingual",
        "voices": "Cloning",
        "realtime_factor": 0.5,
        "note": "Vietnamese via fallback to MMS",
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
        table.add_column("Description", style="white")
        table.add_column("Streaming", style="yellow")
        table.add_column("Packages", style="dim")

        models = list(ASR_MODELS.keys())
        for i, key in enumerate(models, 1):
            model = ASR_MODELS[key]
            streaming = "[green]Yes[/green]" if model.get("streaming") else "[yellow]No[/yellow]"
            packages = ", ".join(model["packages"][:2])
            if len(model["packages"]) > 2:
                packages += "..."
            table.add_row(
                str(i),
                model["name"],
                model["description"],
                streaming,
                packages[:40] + "..." if len(packages) > 40 else packages,
            )

        console.print(table)

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
        table.add_column("Description", style="white")
        table.add_column("Vietnamese", style="yellow")
        table.add_column("RAM Required", style="dim")

        models = list(LLM_MODELS.keys())
        for i, key in enumerate(models, 1):
            model = LLM_MODELS[key]
            vietnamese = "[green]Yes[/green]" if model.get("vietnamese") else "[red]No[/red]"
            ram = f"{model.get('ram_required_gb', 0)} GB" if model.get("ram_required_gb") else "N/A"
            table.add_row(
                str(i),
                model["name"],
                model["description"],
                vietnamese,
                ram,
            )

        console.print(table)

        # Recommend based on RAM
        recommended = self.detect_environment()["recommended_llm"]
        default_idx = models.index(recommended) + 1 if recommended in models else 1

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
        table.add_column("Description", style="white")
        table.add_column("Real-time", style="yellow")
        table.add_column("Voices", style="dim")

        models = list(TTS_MODELS.keys())
        for i, key in enumerate(models, 1):
            model = TTS_MODELS[key]
            rt = f"{model.get('realtime_factor', 1)}x"
            voices = str(model.get("voices", "1"))
            table.add_row(
                str(i),
                model["name"],
                model["description"],
                f"[green]{rt}[/green]",
                voices,
            )

        console.print(table)

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

        # Step 1: Select ASR model
        self.select_asr_model()

        # Step 2: Select LLM model
        self.select_llm_model()

        # Step 3: Select TTS model
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

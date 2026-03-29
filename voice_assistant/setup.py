"""Interactive setup wizard for voice assistant components."""

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


class ComponentSetup:
    """Component installation manager."""

    VAD = {
        "name": "VAD (Voice Activity Detection)",
        "description": "Detects speech segments and filters silence",
        "packages": ["silero-vad>=0.4.0"],
        "required_for": ["Full voice assistant", "Real-time microphone input"],
        "optional": False,
        "recommended": True,
    }

    ASR = {
        "name": "ASR (Speech-to-Text)",
        "description": "Transcribes speech to text (Vietnamese)",
        "packages": ["sherpa-onnx>=1.10.0"],
        "required_for": ["Voice transcription", "Streaming ASR"],
        "optional": False,
        "recommended": True,
    }

    LLM_LOCAL = {
        "name": "LLM (Local)",
        "description": "Run language model locally (llama.cpp)",
        "packages": ["llama-cpp-python>=0.2.50", "huggingface_hub>=0.20.0"],
        "required_for": ["Local AI responses", "No external server needed"],
        "optional": True,
        "recommended": True,
        "alternatives": {
            "server": {
                "name": "LLM (Remote Server)",
                "description": "Connect to external llama.cpp server",
                "packages": ["requests"],
                "note": "Requires llama.cpp server running separately",
            }
        }
    }

    LLM_SERVER = {
        "name": "LLM (Remote Server)",
        "description": "Connect to external llama.cpp server",
        "packages": ["requests>=2.28.0"],
        "required_for": ["Remote LLM inference"],
        "optional": True,
        "recommended": False,
    }

    TTS = {
        "name": "TTS (Text-to-Speech)",
        "description": "Synthesize speech with Vietnamese voices",
        "packages": ["vieneu>=0.1.0"],
        "required_for": ["Voice output", "Audio responses"],
        "optional": True,
        "recommended": True,
    }

    AUDIO = {
        "name": "Audio I/O",
        "description": "Microphone input and speaker output",
        "packages": ["pyaudio>=0.2.13", "sounddevice>=0.4.6", "numpy>=1.20.0"],
        "required_for": ["Microphone input", "Speaker output"],
        "optional": False,
        "recommended": True,
    }

    TUI = {
        "name": "TUI (Terminal UI)",
        "description": "Rich terminal interface with live display",
        "packages": ["rich>=13.0", "typer>=0.9"],
        "required_for": ["Live transcription display", "Interactive interface"],
        "optional": True,
        "recommended": True,
    }

    VISION = {
        "name": "Vision (VLM)",
        "description": "Image and screen analysis",
        "packages": ["opencv-python>=4.9.0", "pillow>=10.0.0"],
        "required_for": ["Screen capture", "Camera analysis"],
        "optional": True,
        "recommended": False,
    }

    ALL = {
        "name": "All Components",
        "description": "Install everything for full functionality",
        "packages": [".[all]"],
        "required_for": ["Complete voice assistant experience"],
        "optional": False,
        "recommended": True,
    }


class SetupWizard:
    """Interactive setup wizard."""

    def __init__(self):
        self.sys_info = SystemInfo()
        self.platform = self.sys_info.get_platform()
        self.gpu_info = self.sys_info.get_gpu_info()
        self.ram_gb = self.sys_info.get_ram_gb()

    def detect_environment(self) -> Dict[str, Any]:
        info = {
            "platform": self.platform,
            "python_version": self.sys_info.get_python_version(),
            "gpu": self.gpu_info,
            "ram_gb": self.ram_gb,
            "using_uv": self.sys_info.is_using_uv(),
        }

        # Determine recommendations
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

        # LLM recommendation based on RAM
        if info["ram_gb"] >= 16:
            info["recommended_llm"] = "qwen3-2b"
        elif info["ram_gb"] >= 8:
            info["recommended_llm"] = "qwen3-0.6b"
        else:
            info["recommended_llm"] = "remote-server"

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
                table.add_row("Recommendation", f"[green]Use CUDA for LLM/TTS[/green]")
            elif info["gpu"]["type"] == "metal":
                table.add_row("Metal", "[green]Yes (Apple Silicon)[/green]")
                table.add_row("Recommendation", f"[green]Use Metal for LLM/TTS[/green]")
        else:
            table.add_row("GPU", "[yellow]Not detected (CPU mode)[/yellow]")
            if info["ram_gb"] >= 16:
                table.add_row("Recommendation", "[green]CPU inference possible with smaller models[/green]")
            else:
                table.add_row("Recommendation", "[yellow]Consider remote llama.cpp server[/yellow]")

        table.add_row("Package Manager", "[green]uv[/green]" if info["using_uv"] else "[yellow]pip[/yellow]")

        console.print(table)
        console.print("")

    def show_component_menu(self) -> List[str]:
        """Show component selection menu."""
        console.print(Panel.fit(
            "[bold]Select Components to Install[/bold]\n\n"
            "Choose which parts of the voice assistant to set up.",
            border_style="green",
        ))
        console.print("")

        # Preset configurations
        presets = {
            "full": {
                "name": "Full Installation (Recommended)",
                "components": ["vad", "asr", "llm_local", "tts", "audio", "tui"],
                "description": "Everything for complete voice assistant",
            },
            "minimal": {
                "name": "Minimal (CLI text-only)",
                "components": ["llm_local", "tui"],
                "description": "Basic chat without audio",
            },
            "server": {
                "name": "Server Mode",
                "components": ["vad", "asr", "llm_server", "tts", "audio", "tui"],
                "description": "Connect to remote llama.cpp server",
            },
            "custom": {
                "name": "Custom Selection",
                "components": [],
                "description": "Pick individual components",
            },
        }

        # Show presets table
        table = Table(title="Installation Presets")
        table.add_column("#", style="cyan", width=2)
        table.add_column("Name", style="green bold")
        table.add_column("Components", style="white")
        table.add_column("Best For", style="yellow")

        table.add_row(
            "1", "Full",
            "VAD + ASR + LLM + TTS + Audio",
            "Complete voice assistant experience",
        )
        table.add_row(
            "2", "Minimal",
            "LLM + TUI only",
            "Quick testing, text-only chat",
        )
        table.add_row(
            "3", "Server Mode",
            "All + Remote LLM",
            "Systems without local LLM support",
        )
        table.add_row(
            "4", "Custom",
            "Choose individually",
            "Advanced users",
        )

        console.print(table)
        console.print("")

        choice = Prompt.ask(
            "Select preset",
            choices=["1", "2", "3", "4"],
            default="1"
        )

        if choice == "4":
            # Custom selection
            return self._custom_component_selection()
        elif choice == "1":
            return presets["full"]["components"]
        elif choice == "2":
            return presets["minimal"]["components"]
        elif choice == "3":
            return presets["server"]["components"]

        return presets["full"]["components"]

    def _custom_component_selection(self) -> List[str]:
        """Let user pick individual components."""
        console.print("")
        console.print("[bold]Select components (comma-separated numbers):[/bold]")
        console.print("")

        components = []
        idx = 1

        table = Table()
        table.add_column("#", style="cyan")
        table.add_column("Component", style="green")
        table.add_column("Description", style="white")
        table.add_column("Required", style="yellow")

        component_map = {
            "vad": ComponentSetup.VAD,
            "asr": ComponentSetup.ASR,
            "llm_local": ComponentSetup.LLM_LOCAL,
            "llm_server": ComponentSetup.LLM_SERVER,
            "tts": ComponentSetup.TTS,
            "audio": ComponentSetup.AUDIO,
            "tui": ComponentSetup.TUI,
            "vision": ComponentSetup.VISION,
        }

        for key, comp in component_map.items():
            required = "[green]Yes[/green]" if not comp.get("optional", False) else "[yellow]No[/yellow]"
            table.add_row(str(idx), comp["name"], comp["description"], required)
            components.append(key)
            idx += 1

        console.print(table)
        console.print("")

        # Auto-select recommended
        default_selection = "1,2,3,5,6,7"  # vad, asr, llm_local, tts, audio, tui
        selection = Prompt.ask(
            "Enter component numbers",
            default=default_selection
        )

        selected = []
        for num in selection.split(","):
            num = num.strip()
            if num.isdigit() and 1 <= int(num) <= len(components):
                selected.append(components[int(num) - 1])

        return selected if selected else ["vad", "asr", "llm_local", "tui"]

    def get_packages_to_install(self, selected_components: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """Get list of packages to install based on selected components."""
        packages = []
        component_status = {}

        component_map = {
            "vad": ComponentSetup.VAD,
            "asr": ComponentSetup.ASR,
            "llm_local": ComponentSetup.LLM_LOCAL,
            "llm_server": ComponentSetup.LLM_SERVER,
            "tts": ComponentSetup.TTS,
            "audio": ComponentSetup.AUDIO,
            "tui": ComponentSetup.TUI,
            "vision": ComponentSetup.VISION,
        }

        for comp_key in selected_components:
            if comp_key in component_map:
                comp = component_map[comp_key]
                packages.extend(comp["packages"])
                component_status[comp["name"]] = "pending"

        # Handle special case: .[all]
        if ".[all]" in packages:
            return [".[all]"], {"All Components": "pending"}

        return packages, component_status

    def install_packages(self, packages: List[str], using_uv: bool) -> Dict[str, bool]:
        """Install packages and return success status per package."""
        results = {}

        if not packages:
            console.print("[yellow]No packages to install[/yellow]")
            return results

        console.print("")
        console.print(Panel.fit(
            f"[bold]Installing packages:[/bold]\n{', '.join(packages)}",
            border_style="yellow",
        ))
        console.print("")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task("Installing...", total=len(packages))

                if using_uv:
                    cmd = ["uv", "pip", "install"]
                else:
                    cmd = [sys.executable, "-m", "pip", "install"]

                # Install all packages at once
                cmd.extend(packages)

                process = progress.add_task(f"Running: {' '.join(cmd[:3])}...", total=None)

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                progress.update(task, completed=len(packages))

                if result.returncode == 0:
                    console.print("[green]All packages installed successfully![/green]")
                    for pkg in packages:
                        results[pkg] = True

                    # Show any warnings
                    if "warning" in result.stderr.lower():
                        console.print("[yellow]Warnings:[/yellow]")
                        for line in result.stderr.split("\n"):
                            if "warning" in line.lower():
                                console.print(f"  {line}")
                else:
                    console.print(f"[red]Installation failed:[/red]")
                    console.print(result.stderr)
                    for pkg in packages:
                        results[pkg] = False

        except subprocess.TimeoutExpired:
            console.print("[red]Installation timed out (5 min limit)[/red]")
            for pkg in packages:
                results[pkg] = False
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            for pkg in packages:
                results[pkg] = False

        return results

    def show_component_status(self, status: Dict[str, bool]) -> None:
        """Show installation status per component."""
        console.print("")
        console.print(Panel.fit(
            "[bold]Installation Status[/bold]",
            border_style="green",
        ))

        table = Table(show_header=False, box=None)
        table.add_column("Status", style="cyan", width=3)
        table.add_column("Component", style="white")

        for pkg, success in status.items():
            icon = "[green]✓[/green]" if success else "[red]✗[/red]"
            table.add_row(icon, pkg)

        console.print(table)

    def show_next_steps(self, selected_components: List[str], install_success: Dict[str, bool]) -> None:
        """Show next steps after installation."""
        console.print("")
        console.print(Panel.fit(
            "[bold]Next Steps[/bold]",
            border_style="blue",
        ))

        steps = []

        # Check what was installed
        has_llm_local = "llm_local" in selected_components and install_success.get("llama-cpp-python>=0.2.50", False)
        has_llm_server = "llm_server" in selected_components
        has_full = "vad" in selected_components and "asr" in selected_components

        if has_llm_local:
            steps.append("""
[bold green]1. Run Voice Assistant (Local LLM)[/bold green]

   [cyan]uv run python -m voice_assistant cli[/cyan]

   The LLM model will be downloaded automatically on first use.
   Recommended model: qwen3-2b (requires ~4GB disk)
""")
        elif has_llm_server:
            steps.append("""
[bold green]1. Start llama.cpp Server[/bold green]

   Download llama.cpp from: https://github.com/ggerganov/llama.cpp/releases

   [cyan]llama-server.exe -m qwen3-2b-q4_k_m.gguf --port 8000[/cyan]

[bold green]2. Connect Voice Assistant[/bold green]

   [cyan]uv run python -m voice_assistant cli --server-url http://localhost:8000[/cyan]
""")

        if has_full:
            steps.append("""
[bold green]Full Voice Assistant (with audio)[/bold green]

   [cyan]uv run python -m voice_assistant run[/cyan]

   This starts the complete pipeline:
   - VAD: Voice activity detection
   - ASR: Speech-to-text transcription
   - LLM: AI responses
   - TTS: Voice output
""")

        steps.append("""
[bold]Available Commands[/bold]

  [cyan]uv run python -m voice_assistant cli[/cyan]      - Text-only chat
  [cyan]uv run python -m voice_assistant run[/cyan]      - Full TUI interface
  [cyan]uv run python -m voice_assistant list-models[/cyan] - Show available models
""")

        for i, step in enumerate(steps, 1):
            console.print(step)
            if i < len(steps):
                console.print("---")
                console.print("")

    def generate_config(self, info: Dict[str, Any], components: List[str]) -> None:
        """Generate a recommended config file."""
        console.print("")
        console.print("[bold]Generating recommended config...[/bold]")

        import json

        config = {
            "llm": {
                "model": info["recommended_llm"] if info["recommended_llm"] != "remote-server" else "qwen3-2b",
            },
            "tts": {
                "model": "vietneu-tts",
                "use_gpu": info["gpu"] is not None,
            },
        }

        if info["gpu"]:
            config["llm"]["n_gpu_layers"] = info.get("llm_gpu_layers", 35)

        if "llm_server" in components and info["gpu"] is None:
            config["llm"]["server_url"] = "http://localhost:8000"

        config_path = "voice_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        console.print(f"[green]Config saved to:[/green] {config_path}")
        console.print("")
        console.print("[dim]Load config with: -c voice_config.json[/dim]")

    def run(self) -> int:
        """Run the setup wizard."""
        console.print(Panel.fit(
            "[bold cyan]Voice Assistant Setup Wizard[/bold cyan]\n\n"
            "Step-by-step setup for VAD, ASR, LLM, and TTS components\n"
            "Optimized for your hardware configuration.",
            border_style="cyan",
        ))
        console.print("")

        # Detect system
        console.print("[bold]Detecting system...[/bold]")
        info = self.detect_environment()
        self.print_system_info(info)

        # Select components
        selected = self.show_component_menu()
        console.print(f"\nSelected: [green]{', '.join(selected)}[/green]\n")

        if not Confirm.ask("Continue with installation?", default=True):
            console.print("[yellow]Setup cancelled.[/yellow]")
            return 1

        # Get packages
        packages, component_status = self.get_packages_to_install(selected)

        # Install
        install_results = self.install_packages(packages, info["using_uv"])
        self.show_component_status(install_results)

        # Generate config
        self.generate_config(info, selected)

        # Show next steps
        self.show_next_steps(selected, install_results)

        # Summary
        all_success = all(install_results.values()) if install_results else False
        if all_success:
            console.print("\n[bold green]Setup completed successfully![/bold green]")
        else:
            console.print("\n[bold yellow]Setup completed with some errors.[/bold yellow]")
            console.print("[dim]You can re-run setup or install missing packages manually.[/dim]")

        return 0 if all_success else 1


def main():
    """Main entry point."""
    wizard = SetupWizard()
    sys.exit(wizard.run())


if __name__ == "__main__":
    main()

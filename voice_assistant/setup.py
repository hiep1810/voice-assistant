"""Interactive setup wizard for voice assistant."""

import sys
import platform
import subprocess
import shutil
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class SystemInfo:
    """System information detector."""

    @staticmethod
    def get_platform() -> str:
        """Get platform name."""
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
        """Check if NVIDIA CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            # Try to detect CUDA via nvidia-smi
            return shutil.which("nvidia-smi") is not None

    @staticmethod
    def get_gpu_info() -> Optional[Dict[str, Any]]:
        """Get GPU information."""
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

        # Try macOS Metal
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

        # Try nvidia-smi on Windows/Linux
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
        """Get system RAM in GB."""
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
                # Linux
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
        """Check if running under uv."""
        return "UV" in str(subprocess.__file__) or shutil.which("uv") is not None


class SetupWizard:
    """Interactive setup wizard."""

    def __init__(self):
        self.sys_info = SystemInfo()
        self.platform = self.sys_info.get_platform()
        self.gpu_info = self.sys_info.get_gpu_info()
        self.ram_gb = self.sys_info.get_ram_gb()

    def detect_environment(self) -> Dict[str, Any]:
        """Detect system environment and recommend setup."""
        info = {
            "platform": self.platform,
            "python_version": self.sys_info.get_python_version(),
            "gpu": self.gpu_info,
            "ram_gb": self.ram_gb,
            "using_uv": self.sys_info.is_using_uv(),
        }

        # Determine recommended setup
        if info["gpu"]:
            gpu_type = info["gpu"]["type"]
            if gpu_type == "cuda":
                info["recommended_backend"] = "cuda"
                info["recommended_llm"] = "qwen3-2b"
                info["gpu_memory"] = info["gpu"].get("memory_gb", 0)
            elif gpu_type == "metal":
                info["recommended_backend"] = "metal"
                info["recommended_llm"] = "qwen3-2b"
        else:
            # CPU only
            if info["ram_gb"] >= 16:
                info["recommended_llm"] = "qwen3-2b"
            else:
                info["recommended_llm"] = "qwen3-0.6b"
            info["recommended_backend"] = "cpu"

        return info

    def print_system_info(self, info: Dict[str, Any]) -> None:
        """Print detected system information."""
        console.print(Panel.fit(
            "[bold]System Detection Results[/bold]",
            border_style="blue",
        ))

        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Platform", f"{info['platform'].capitalize()} ({platform.machine()})")
        table.add_row("Python", info["python_version"])
        table.add_row("RAM", f"{info['ram_gb']:.1f} GB")

        if info["gpu"]:
            table.add_row("GPU", f"[green]{info['gpu']['name']}[/green]")
            if info["gpu"]["type"] == "cuda":
                table.add_row("CUDA", f"[green]Yes ({info['gpu'].get('memory_gb', 0):.1f} GB VRAM)[/green]")
            elif info["gpu"]["type"] == "metal":
                table.add_row("Metal", "[green]Yes (Apple Silicon)[/green]")
        else:
            table.add_row("GPU", "[yellow]Not detected (CPU only mode)[/yellow]")

        table.add_row("Package Manager", "[green]uv[/green]" if info["using_uv"] else "[yellow]pip[/yellow]")

        console.print(table)
        console.print("")

    def get_installation_options(self, info: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get installation options based on system."""
        options = []

        # Option 1: Recommended
        if info["gpu"] and info["gpu"]["type"] == "cuda":
            options.append({
                "id": "nvidia-full",
                "name": "NVIDIA GPU (Recommended)",
                "description": f"Full voice assistant with CUDA acceleration ({info['gpu']['name']})",
                "packages": ".[all]",
                "llm": info["recommended_llm"],
                "priority": 1,
            })
        elif info["gpu"] and info["gpu"]["type"] == "metal":
            options.append({
                "id": "macos-full",
                "name": "Apple Silicon (Recommended)",
                "description": "Full voice assistant with Metal acceleration",
                "packages": ".[all]",
                "llm": info["recommended_llm"],
                "priority": 1,
            })

        # Option 2: CPU only
        options.append({
            "id": "cpu-only",
            "name": "CPU Only",
            "description": f"Lightweight setup without GPU acceleration (uses {info['recommended_llm']})",
            "packages": "llama-cpp-python huggingface_hub",
            "llm": info["recommended_llm"],
            "priority": 2,
        })

        # Option 3: Server mode
        options.append({
            "id": "server-mode",
            "name": "Remote Server Mode",
            "description": "Connect to external llama.cpp server (no local LLM)",
            "packages": "",
            "llm": "remote",
            "priority": 3,
        })

        # Option 4: Minimal
        options.append({
            "id": "minimal",
            "name": "Minimal (CLI only)",
            "description": "Text-only interface, no audio processing",
            "packages": "typer rich",
            "llm": "none",
            "priority": 4,
        })

        return sorted(options, key=lambda x: x["priority"])

    def show_options(self, options: List[Dict[str, str]]) -> Dict[str, str]:
        """Show installation options and get user choice."""
        console.print(Panel.fit(
            "[bold]Installation Options[/bold]",
            border_style="green",
        ))
        console.print("")

        table = Table(title="Select Setup Type")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Name", style="green bold")
        table.add_column("Description", style="white")
        table.add_column("Packages", style="yellow")

        for i, opt in enumerate(options, 1):
            table.add_row(
                str(i),
                opt["name"],
                opt["description"],
                opt["packages"] or "N/A",
            )

        console.print(table)
        console.print("")

        # Get user choice
        while True:
            choice = Prompt.ask(
                "Select option",
                choices=[str(i) for i in range(1, len(options) + 1)],
                default="1",
            )
            return options[int(choice) - 1]

    def install_packages(self, option: Dict[str, str], using_uv: bool) -> bool:
        """Install selected packages."""
        packages = option["packages"]

        if not packages:
            console.print("[green]No packages to install (server mode)[/green]")
            return True

        console.print("")
        console.print(Panel.fit(
            f"[bold]Installing:[/] {packages}",
            border_style="yellow",
        ))
        console.print("")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Installing packages...", total=None)

                if using_uv:
                    cmd = ["uv", "pip", "install"]
                else:
                    cmd = [sys.executable, "-m", "pip", "install"]

                cmd.extend(packages.split())

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                progress.update(task, completed=True)

                if result.returncode == 0:
                    console.print("[green]Packages installed successfully![/green]")
                    return True
                else:
                    console.print(f"[red]Installation failed:[/red] {result.stderr}")
                    return False

        except subprocess.TimeoutExpired:
            console.print("[red]Installation timed out[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return False

    def show_next_steps(self, option: Dict[str, str]) -> None:
        """Show next steps after installation."""
        console.print("")
        console.print(Panel.fit(
            "[bold]Next Steps[/bold]",
            border_style="blue",
        ))

        if option["id"] == "server-mode":
            console.print("""
[bold]To use with a remote llama.cpp server:[/bold]

1. Start llama.cpp server:
   [cyan]llama-server.exe -m qwen3-2b-q4_k_m.gguf --port 8000[/cyan]

2. Run voice assistant:
   [cyan]uv run python -m voice_assistant cli --server-url http://localhost:8000[/cyan]
""")
        elif option["id"] == "minimal":
            console.print("""
[bold]Run the CLI (text-only mode):[/bold]

   [cyan]uv run python -m voice_assistant cli[/cyan]

Note: Audio features (microphone, TTS) are not available in minimal mode.
""")
        else:
            llm = option.get("llm", "qwen3-2b")
            console.print(f"""
[bold]Run voice assistant:[/bold]

1. CLI mode (text-only):
   [cyan]uv run python -m voice_assistant cli[/cyan]

2. TUI mode (full interface with live display):
   [cyan]uv run python -m voice_assistant run[/cyan]

[bold]Model:[/] {llm} (will be downloaded on first use)
""")

    def run(self) -> int:
        """Run the setup wizard."""
        console.print(Panel.fit(
            "[bold cyan]Voice Assistant Setup Wizard[/bold cyan]\n\n"
            "This will help you install and configure the voice assistant\n"
            "with optimal settings for your system.",
            border_style="cyan",
        ))
        console.print("")

        # Detect system
        console.print("[bold]Detecting system configuration...[/bold]")
        info = self.detect_environment()
        self.print_system_info(info)

        # Show options
        options = self.get_installation_options(info)
        selected = self.show_options(options)

        console.print("")
        console.print(f"You selected: [green]{selected['name']}[/green]")
        console.print("")

        # Confirm installation
        if not Confirm.ask("Continue with installation?", default=True):
            console.print("[yellow]Setup cancelled.[/yellow]")
            return 1

        # Install packages
        success = self.install_packages(selected, info["using_uv"])

        if not success:
            console.print("[red]Installation failed. Please check the error messages above.[/red]")
            return 1

        # Show next steps
        self.show_next_steps(selected)

        return 0


def main():
    """Main entry point for setup wizard."""
    wizard = SetupWizard()
    sys.exit(wizard.run())


if __name__ == "__main__":
    main()

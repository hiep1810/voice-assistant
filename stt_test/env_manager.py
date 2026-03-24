"""
Environment Manager — creates and manages isolated venvs per model.

Design Pattern: Strategy Pattern (via subprocess isolation)
ELI5: Each model runs in its own sandbox so their dependencies can't fight.
Why subprocess: Heavy ML libs (NeMo, FunASR, etc.) have conflicting deps.
              Running each in its own venv via subprocess avoids import crashes.
Real Analogy: Separate kitchens for different cuisines — they share the dining
              room (CLI) but don't contaminate each other's ingredients.

Why stdlib `venv` over `virtualenv`?
  `venv` is built into Python 3.3+, so zero extra dependencies for the CLI.
  `virtualenv` is ~20% faster at creating envs but adds a pip dependency we
  don't need for this project.
"""

from __future__ import annotations

import json
import subprocess
import sys
import venv
from pathlib import Path

from rich.console import Console

from stt_test.registry import ModelConfig, get_model

console = Console()

# ---------------------------------------------------------------------------
# All venvs live under <project_root>/envs/<model_name>/
# Why project root? Keeps everything self-contained and gitignore-able.
# ---------------------------------------------------------------------------
ENVS_DIR = Path(__file__).resolve().parent.parent / "envs"


def _get_env_dir(model_name: str) -> Path:
    """Return the venv directory path for a model."""
    return ENVS_DIR / model_name


def _get_python(env_dir: Path) -> Path:
    """Return the Python executable inside a venv (cross-platform)."""
    # Windows uses Scripts/python.exe, Unix uses bin/python
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _get_pip(env_dir: Path) -> Path:
    """Return the pip executable inside a venv (cross-platform)."""
    if sys.platform == "win32":
        return env_dir / "Scripts" / "pip.exe"
    return env_dir / "bin" / "pip"


def is_env_ready(model_name: str) -> bool:
    """Check if the venv exists and has a working Python."""
    env_dir = _get_env_dir(model_name)
    python = _get_python(env_dir)
    return python.exists()


def setup_env(model: ModelConfig) -> Path:
    """Create a venv for the model and install its packages.

    Returns the path to the venv directory.
    Skips creation if the venv already exists.
    """
    env_dir = _get_env_dir(model.name)
    python = _get_python(env_dir)

    if python.exists():
        console.print(f"  [dim]venv already exists:[/] {env_dir}")
    else:
        console.print(f"  [cyan]Creating venv:[/] {env_dir}")
        # with_pip=True ensures pip is available inside the venv
        venv.create(str(env_dir), with_pip=True)
        console.print("  [green]✓[/] venv created")

    # Always upgrade pip first to avoid stale-pip issues
    console.print("  [cyan]Upgrading pip...[/]")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
        capture_output=True,
    )

    # Install model-specific packages
    runtime_packages = model.get_runtime_packages()
    console.print(f"  [cyan]Installing packages:[/] {', '.join(runtime_packages)}")
    
    # Strategy: Try 'uv' first for blazing fast installs, fallback to pip
    # Why? 'uv' is optimized for large ML installs and handles our CUDA/MPS deps better.
    extra_args = model.get_runtime_extra_pip_args()
    try:
        # Check if uv is installed
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        # Use uv pip install --python venv/bin/python
        cmd = ["uv", "pip", "install", "--python", str(python)] + runtime_packages + extra_args
        console.print("  [dim]Using 'uv' for fast installation...[/]")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to standard pip
        cmd = [str(_get_pip(env_dir)), "install"] + runtime_packages + extra_args
        console.print("  [dim]Using standard 'pip'...[/]")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"  [red]✗ pip install failed:[/]\n{result.stderr}")
        raise RuntimeError(f"Failed to install packages for {model.name}")

    console.print(f"  [green]✓[/] packages installed for [bold]{model.display_name}[/]")
    return env_dir


def run_in_env(
    model_name: str,
    audio_path: str,
) -> dict:
    """Run the model's inference script inside its isolated venv.

    Args:
        model_name: Registry key (e.g. "parakeet").
        audio_path: Absolute path to the audio file.

    Returns:
        Parsed JSON dict from the script's stdout.
    """
    model = get_model(model_name)
    env_dir = _get_env_dir(model_name)
    python = _get_python(env_dir)

    if not python.exists():
        raise RuntimeError(
            f"Environment for '{model_name}' not set up. "
            f"Run: python -m stt_test setup {model_name}"
        )

    # Use the platform-specific script if defined (e.g. run_whisper_mac.py for MLX)
    runtime_script = model.get_runtime_script()
    script_path = Path(__file__).resolve().parent / "scripts" / runtime_script

    # Run the script in the model's venv Python
    result = subprocess.run(
        [str(python), str(script_path), audio_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        console.print(f"  [red]✗ Inference failed for {model.display_name}:[/]")
        console.print(f"  [dim]{result.stderr}[/]")
        raise RuntimeError(f"Inference failed: {result.stderr}")

    # Parse the JSON output from the script (last line to ignore logs)
    try:
        lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        if not lines:
            raise ValueError("Empty stdout")
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        console.print(f"  [red]✗ Invalid JSON from script:[/] {e}")
        console.print(f"  [dim]stdout was: {result.stdout[:500]}[/]")
        raise

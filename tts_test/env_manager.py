"""
TTS Environment Manager — creates and manages isolated venvs per TTS model.

Design Pattern: Strategy Pattern (via subprocess isolation)
ELI5: Each TTS model runs in its own sandbox so their dependencies can't fight.
Why subprocess: Heavy ML libs (TTS, VITS, GPT-SoVITS) have conflicting deps.
              Running each in its own venv via subprocess avoids import crashes.

Mirrors stt_test/env_manager.py for consistency with ASR benchmarking.
"""

from __future__ import annotations

import json
import subprocess
import sys
import venv
from pathlib import Path
from typing import Optional

from rich.console import Console

from tts_test.registry import TTSModelConfig, get_tts_model, TTS_MODELS

console = Console()

# ---------------------------------------------------------------------------
# All venvs live under <project_root>/envs/tts/<model_name>/
# Why separate from ASR? TTS models have different dependencies than STT.
# ---------------------------------------------------------------------------
ENVS_DIR = Path(__file__).resolve().parent.parent / "envs" / "tts"


def _get_env_dir(model_name: str) -> Path:
    """Return the venv directory path for a TTS model."""
    return ENVS_DIR / model_name


def _get_python(env_dir: Path) -> Path:
    """Return the Python executable inside a venv (cross-platform)."""
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _get_pip(env_dir: Path) -> Path:
    """Return the pip executable inside a venv (cross-platform)."""
    if sys.platform == "win32":
        return env_dir / "Scripts" / "pip.exe"
    return env_dir / "bin" / "pip"


def is_tts_env_ready(model_name: str) -> bool:
    """Check if the venv exists and has a working Python."""
    env_dir = _get_env_dir(model_name)
    python = _get_python(env_dir)

    # Special case: vietneu-tts uses Python 3.10 from pyenv
    if model_name == "vietneu-tts":
        if python.exists():
            return True
        # Also check for pyenv-managed venv
        pyenv_python = Path.home() / ".pyenv" / "pyenv-win" / "versions" / "3.10.5" / "python.exe"
        if pyenv_python.exists():
            return True
        return False

    return python.exists()


def setup_tts_env(model: TTSModelConfig) -> Path:
    """Create a venv for the TTS model and install its packages.

    Returns the path to the venv directory.
    Skips creation if the venv already exists.
    """
    # Check Python version requirement for specific models
    if model.name == "vietneu-tts":
        # Check if already installed via pyenv
        env_dir = _get_env_dir(model.name)
        python = _get_python(env_dir)
        if python.exists():
            console.print(f"  [dim]venv already exists:[/] {env_dir}")
            console.print(f"  [green]OK[/] VieNeu-TTS is ready (Python 3.10+)")
            return env_dir

        console.print(f"  [yellow]Warning:[/] {model.display_name} requires Python 3.10+")
        console.print(f"  [yellow]Current Python: {sys.version}[/]")
        console.print(f"  [dim]Skipping automatic setup. Install manually with Python 3.10+:[/]")
        console.print(f"  [dim]  pyenv install 3.10.5[/]")
        console.print(f"  [dim]  python -m venv envs/tts/{model.name}[/]")
        console.print(f"  [dim]  pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/[/]")
        raise RuntimeError(f"{model.display_name} requires Python 3.10+ (current: {sys.version_info.major}.{sys.version_info.minor})")

    env_dir = _get_env_dir(model.name)
    python = _get_python(env_dir)

    if python.exists():
        console.print(f"  [dim]venv already exists:[/] {env_dir}")
    else:
        console.print(f"  [cyan]Creating venv:[/] {env_dir}")
        venv.create(str(env_dir), with_pip=True)
        console.print("  [green]OK[/] venv created")

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
    extra_args = model.get_runtime_extra_pip_args()

    # Check if 'uv' is installed
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        has_uv = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        has_uv = False

    if has_uv:
        cmd = ["uv", "pip", "install", "--python", str(python)] + runtime_packages + extra_args
        console.print("  [dim]Using 'uv' for fast installation...[/]")
    else:
        # Fallback to standard pip
        cmd = [str(_get_pip(env_dir)), "install"] + runtime_packages + extra_args
        console.print("  [dim]Using standard 'pip'...[/]")

    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

    if result.returncode != 0:
        # If uv failed, try fallback to pip
        if has_uv and "No solution found" in result.stderr:
            console.print("  [dim]uv failed, falling back to pip...[/]")
            cmd = [str(_get_pip(env_dir)), "install"] + runtime_packages + extra_args
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

        if result.returncode != 0:
            console.print(f"  [red]ERROR pip install failed:[/]\n{result.stderr[:500]}")
            raise RuntimeError(f"Failed to install packages for {model.name}")

    console.print(f"  [green]OK[/] packages installed for [bold]{model.display_name}[/]")
    return env_dir


def run_in_tts_env(
    model_name: str,
    text: str,
    output_path: Optional[str] = None,
    speaker: Optional[str] = None,
) -> dict:
    """Run the TTS model's synthesis inside its isolated venv.

    Args:
        model_name: Registry key (e.g. "vietts").
        text: Vietnamese text to synthesize.
        output_path: Optional path to save output audio.
        speaker: Optional speaker/voice ID for models that support it.

    Returns:
        Parsed JSON dict from the script's stdout.
    """
    model = get_tts_model(model_name)
    env_dir = _get_env_dir(model_name)
    python = _get_python(env_dir)

    # Special case: vietneu-tts uses Python 3.10 from pyenv
    if model_name == "vietneu-tts":
        if not python.exists():
            # Try pyenv-managed venv
            pyenv_python = Path.home() / ".pyenv" / "pyenv-win" / "versions" / "3.10.5" / "python.exe"
            if pyenv_python.exists():
                python = pyenv_python
            else:
                raise RuntimeError(
                    f"Environment for '{model_name}' not set up. "
                    f"Install Python 3.10+ via pyenv: pyenv install 3.10.5"
                )
    elif not python.exists():
        raise RuntimeError(
            f"Environment for '{model_name}' not set up. "
            f"Run: python -m tts_test setup {model_name}"
        )

    runtime_script = model.get_runtime_script()
    script_path = Path(__file__).resolve().parent / "scripts" / runtime_script

    # Build command with arguments
    cmd = [str(python), str(script_path), "--text", text]
    if output_path:
        cmd.extend(["--output", output_path])
    if speaker:
        cmd.extend(["--speaker", speaker])

    # Run the script in the model's venv Python
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"  [red]ERROR Synthesis failed for {model.display_name}:[/]")
        console.print(f"  [dim]{result.stderr}[/]")
        raise RuntimeError(f"Synthesis failed: {result.stderr}")

    # Parse the JSON output from the script (last line to ignore logs)
    try:
        lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        if not lines:
            raise ValueError("Empty stdout")
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        console.print(f"  [red]ERROR Invalid JSON from script:[/] {e}")
        console.print(f"  [dim]stdout was: {result.stdout[:500]}[/]")
        raise


def setup_all_tts_envs() -> None:
    """Set up environments for all TTS models."""
    for model in TTS_MODELS.values():
        console.print(f"\n[bold]Setting up {model.display_name}...[/]")
        try:
            setup_tts_env(model)
        except Exception as e:
            console.print(f"[red]Failed to set up {model.display_name}:[/] {e}")

"""
VAD Environment Manager — creates and manages isolated venvs per VAD model.

Design Pattern: Strategy Pattern (via subprocess isolation)
ELI5: Each VAD model runs in its own sandbox so their dependencies can't fight.

Mirrors stt_test/env_manager.py and tts_test/env_manager.py for consistency.
"""

from __future__ import annotations

import json
import subprocess
import sys
import venv
from pathlib import Path
from typing import Optional

from rich.console import Console

from vad_test.registry import VADModelConfig, get_vad_model, VAD_MODELS

console = Console()

# ---------------------------------------------------------------------------
# All venvs live under <project_root>/envs/vad/<model_name>/
# ---------------------------------------------------------------------------
ENVS_DIR = Path(__file__).resolve().parent.parent / "envs" / "vad"


def _get_env_dir(model_name: str) -> Path:
    """Return the venv directory path for a VAD model."""
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


def is_vad_env_ready(model_name: str) -> bool:
    """Check if the venv exists and has a working Python."""
    env_dir = _get_env_dir(model_name)
    python = _get_python(env_dir)
    return python.exists()


def setup_vad_env(model: VADModelConfig) -> Path:
    """Create a venv for the VAD model and install its packages.

    Returns the path to the venv directory.
    Skips creation if the venv already exists.
    """
    env_dir = _get_env_dir(model.name)
    python = _get_python(env_dir)

    if python.exists():
        console.print(f"  [dim]venv already exists:[/] {env_dir}")
    else:
        console.print(f"  [cyan]Creating venv:[/] {env_dir}")
        venv.create(str(env_dir), with_pip=True)
        console.print("  [green]OK[/] venv created")

    # Always upgrade pip first
    console.print("  [cyan]Upgrading pip...[/]")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
        capture_output=True,
    )

    # Install model-specific packages
    runtime_packages = model.get_runtime_packages()
    console.print(f"  [cyan]Installing packages:[/] {', '.join(runtime_packages)}")

    # Strategy: Try 'uv' first for fast installs, fallback to pip
    extra_args = model.get_runtime_extra_pip_args()

    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        has_uv = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        has_uv = False

    if has_uv:
        cmd = ["uv", "pip", "install", "--python", str(python)] + runtime_packages + extra_args
        console.print("  [dim]Using 'uv' for fast installation...[/]")
    else:
        cmd = [str(_get_pip(env_dir)), "install"] + runtime_packages + extra_args
        console.print("  [dim]Using standard 'pip'...[/]")

    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

    if result.returncode != 0:
        if has_uv and "No solution found" in result.stderr:
            console.print("  [dim]uv failed, falling back to pip...[/]")
            cmd = [str(_get_pip(env_dir)), "install"] + runtime_packages + extra_args
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

        if result.returncode != 0:
            console.print(f"  [red]ERROR pip install failed:[/]\n{result.stderr[:500]}")
            raise RuntimeError(f"Failed to install packages for {model.name}")

    console.print(f"  [green]OK[/] packages installed for [bold]{model.display_name}[/]")
    return env_dir


def run_in_vad_env(
    model_name: str,
    audio_path: str,
    output_dir: Optional[str] = None,
    action: str = "detect",
) -> dict:
    """Run the VAD model inside its isolated venv.

    Args:
        model_name: Registry key (e.g. "silero-vad").
        audio_path: Path to input audio file.
        output_dir: Optional directory for output files.
        action: Action to perform (detect, trim, segment).

    Returns:
        Parsed JSON dict from the script's stdout.
    """
    model = get_vad_model(model_name)
    env_dir = _get_env_dir(model_name)
    python = _get_python(env_dir)

    if not python.exists():
        raise RuntimeError(
            f"Environment for '{model_name}' not set up. "
            f"Run: python -m vad_test setup {model_name}"
        )

    runtime_script = model.get_runtime_script()
    script_path = Path(__file__).resolve().parent / "scripts" / runtime_script

    # Build command with arguments
    cmd = [str(python), str(script_path), "--audio", audio_path, "--action", action]
    if output_dir:
        cmd.extend(["--output-dir", output_dir])

    # Run the script in the model's venv Python
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"  [red]ERROR VAD processing failed for {model.display_name}:[/]")
        console.print(f"  [dim]{result.stderr}[/]")
        raise RuntimeError(f"VAD processing failed: {result.stderr}")

    # Parse the JSON output from the script
    try:
        lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        if not lines:
            raise ValueError("Empty stdout")
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        console.print(f"  [red]ERROR Invalid JSON from script:[/] {e}")
        console.print(f"  [dim]stdout was: {result.stdout[:500]}[/]")
        raise


def setup_all_vad_envs() -> None:
    """Set up environments for all VAD models."""
    for model in VAD_MODELS.values():
        console.print(f"\n[bold]Setting up {model.display_name}...[/]")
        try:
            setup_vad_env(model)
        except Exception as e:
            console.print(f"[red]Failed to set up {model.display_name}:[/] {e}")

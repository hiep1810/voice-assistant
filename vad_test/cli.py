"""
VAD CLI — Typer app with commands: list, setup, detect, trim, segment, benchmark.

Design Pattern: Command Pattern (via Typer)
ELI5: Each CLI subcommand is a function. Typer wires them up so you can call
      `python -m vad_test <command> <args>` from the terminal.

Mirrors stt_test/cli.py and tts_test/cli.py for consistency.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from vad_test.env_manager import (
    is_vad_env_ready,
    run_in_vad_env,
    setup_vad_env,
    setup_all_vad_envs,
)
from vad_test.registry import VAD_MODELS, get_vad_model, list_vad_models
from vad_test.utils import format_timestamp, calculate_audio_stats, load_audio

app = typer.Typer(
    name="vad_test",
    help="CLI tool for Voice Activity Detection (VAD) on audio files.",
    add_completion=False,
)
console = Console(force_terminal=True, force_interactive=False)


# ---------------------------------------------------------------------------
# LIST — show all available VAD models
# ---------------------------------------------------------------------------
@app.command("list")
def list_cmd() -> None:
    """List all available VAD models and their setup status."""
    table = Table(title="Available VAD Models", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Display Name", style="bold")
    table.add_column("HuggingFace ID", style="dim")
    table.add_column("Status", justify="center")

    for model in list_vad_models():
        status = "[green]ready[/]" if is_vad_env_ready(model.name) else "[yellow]not set up[/]"
        hf_id = model.huggingface_id or "N/A"
        table.add_row(
            model.name,
            model.display_name,
            hf_id,
            status,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# SETUP — create venv and install packages for a VAD model
# ---------------------------------------------------------------------------
@app.command("setup")
def setup_cmd(
    model_name: Optional[str] = typer.Argument(  # noqa: UP007
        None,
        help="Model to set up (e.g. 'silero-vad'). Omit with --all to set up all.",
    ),
    all_models: bool = typer.Option(
        False, "--all", help="Set up environments for ALL models."
    ),
) -> None:
    """Create an isolated venv and install packages for a VAD model."""
    if all_models:
        targets = list(VAD_MODELS.keys())
    elif model_name is None:
        console.print("[yellow]No model specified. Use --all to set up everything.[/]")
        raise typer.Exit(1)
    else:
        targets = [model_name]

    for name in targets:
        model = get_vad_model(name)
        console.print(f"\n[bold]Setting up {model.display_name}...[/]")
        try:
            setup_vad_env(model)
        except Exception as e:
            console.print(f"[red]Failed to set up {model.display_name}:[/] {e}")
            if not all_models:
                raise typer.Exit(1)


# ---------------------------------------------------------------------------
# DETECT — run VAD on audio file and show speech timestamps
# ---------------------------------------------------------------------------
@app.command("detect")
def detect_cmd(
    audio_path: str = typer.Argument(help="Path to input audio file."),
    model_name: str = typer.Option("silero-vad", "--model", "-m", help="VAD model to use."),
    output: Optional[str] = typer.Option(  # noqa: UP007
        None, "--output", "-o", help="Output JSON file path for results."
    ),
) -> None:
    """Detect speech segments in an audio file and show timestamps."""
    model = get_vad_model(model_name)

    if not Path(audio_path).exists():
        console.print(f"[red]Audio file not found:[/] {audio_path}")
        raise typer.Exit(1)

    console.print(f"[bold]Running VAD detection with {model.display_name}...[/]")
    console.print(f"  Input: {audio_path}")

    result = run_in_vad_env(model_name, audio_path, action="detect")

    # Display results
    console.print(f"\n[bold green]VAD Detection Results:[/]")
    console.print(f"  Total duration: {result.get('total_duration_s', '?')}s")
    console.print(f"  Speech duration: {result.get('speech_duration_s', '?')}s")
    console.print(f"  Silence duration: {result.get('silence_duration_s', '?')}s")
    console.print(f"  Speech ratio: {result.get('speech_ratio', 0)*100:.1f}%")
    console.print(f"  Segments found: {result.get('num_segments', 0)}")
    console.print(f"  Inference time: {result.get('inference_time_s', '?')}s")
    console.print(f"  RTF: {result.get('rtf', '?')}")

    # Show segments
    segments = result.get("segments", [])
    if segments:
        console.print(f"\n[bold]Speech Segments:[/]")
        table = Table(show_header=True)
        table.add_column("#", style="dim")
        table.add_column("Start", justify="right")
        table.add_column("End", justify="right")
        table.add_column("Duration (s)", justify="right")

        for i, seg in enumerate(segments):
            table.add_row(
                str(i + 1),
                format_timestamp(seg["start"]),
                format_timestamp(seg["end"]),
                f"{seg['duration']:.3f}",
            )

        console.print(table)

    # Save JSON if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        console.print(f"\n[green]Results saved to:[/] {output_path}")


# ---------------------------------------------------------------------------
# TRIM — remove silence from audio
# ---------------------------------------------------------------------------
@app.command("trim")
def trim_cmd(
    audio_path: str = typer.Argument(help="Path to input audio file."),
    output: str = typer.Option(..., "--output", "-o", help="Output audio file path."),
    model_name: str = typer.Option("silero-vad", "--model", "-m", help="VAD model to use."),
) -> None:
    """Remove silence from audio and save trimmed version."""
    model = get_vad_model(model_name)

    if not Path(audio_path).exists():
        console.print(f"[red]Audio file not found:[/] {audio_path}")
        raise typer.Exit(1)

    output_dir = Path(output).parent
    console.print(f"[bold]Trimming silence with {model.display_name}...[/]")
    console.print(f"  Input: {audio_path}")
    console.print(f"  Output: {output}")

    result = run_in_vad_env(model_name, audio_path, str(output_dir), action="trim")

    if result.get("status") == "success":
        console.print(f"\n[bold green]Trimming Complete:[/]")
        console.print(f"  Original duration: {result.get('original_duration_s', '?')}s")
        console.print(f"  Trimmed duration: {result.get('trimmed_duration_s', '?')}s")
        console.print(f"  Segments merged: {result.get('num_segments_merged', 0)}")
        console.print(f"  Output: {result.get('output_path', '?')}")
    else:
        console.print(f"\n[red]Error:[/] {result.get('message', 'Unknown error')}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# SEGMENT — split audio into speech segments
# ---------------------------------------------------------------------------
@app.command("segment")
def segment_cmd(
    audio_path: str = typer.Argument(help="Path to input audio file."),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Output directory for segments."),
    model_name: str = typer.Option("silero-vad", "--model", "-m", help="VAD model to use."),
) -> None:
    """Split audio into individual speech segments."""
    model = get_vad_model(model_name)

    if not Path(audio_path).exists():
        console.print(f"[red]Audio file not found:[/] {audio_path}")
        raise typer.Exit(1)

    console.print(f"[bold]Segmenting audio with {model.display_name}...[/]")
    console.print(f"  Input: {audio_path}")
    console.print(f"  Output: {output_dir}/")

    result = run_in_vad_env(model_name, audio_path, output_dir, action="segment")

    if result.get("status") == "success":
        console.print(f"\n[bold green]Segmentation Complete:[/]")
        console.print(f"  Segments created: {result.get('num_segments', 0)}")
        console.print(f"  Output directory: {result.get('output_dir', '?')}")

        # Show segment details
        segments = result.get("segments", [])
        if segments:
            table = Table(show_header=True)
            table.add_column("#", style="dim")
            table.add_column("File", style="cyan")
            table.add_column("Start", justify="right")
            table.add_column("End", justify="right")
            table.add_column("Duration (s)", justify="right")

            for i, seg in enumerate(segments):
                table.add_row(
                    str(i + 1),
                    Path(result["segment_paths"][i]).name if "segment_paths" in result else f"segment_{i:03d}.wav",
                    format_timestamp(seg["start"]),
                    format_timestamp(seg["end"]),
                    f"{seg['duration']:.3f}",
                )

            console.print(table)
    else:
        console.print(f"\n[red]Error:[/] {result.get('message', 'Unknown error')}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# BENCHMARK — run VAD on audio file and show performance metrics
# ---------------------------------------------------------------------------
@app.command("benchmark")
def benchmark_cmd(
    audio_path: str = typer.Argument(help="Path to input audio file."),
    model_name: str = typer.Option("silero-vad", "--model", "-m", help="VAD model to use."),
) -> None:
    """Benchmark VAD performance on an audio file."""
    model = get_vad_model(model_name)

    if not Path(audio_path).exists():
        console.print(f"[red]Audio file not found:[/] {audio_path}")
        raise typer.Exit(1)

    console.print(f"[bold]VAD Benchmark with {model.display_name}...[/]")
    console.print(f"  Input: {audio_path}")

    # Run multiple iterations for stable timing
    results = []
    for i in range(5):
        result = run_in_vad_env(model_name, audio_path, action="detect")
        results.append(result)

    # Calculate average metrics
    avg_inference_time = sum(r["inference_time_s"] for r in results) / len(results)
    avg_rtf = sum(r["rtf"] for r in results) / len(results)

    console.print(f"\n[bold green]Benchmark Results (5 iterations):[/]")
    console.print(f"  Average inference time: {avg_inference_time:.4f}s")
    console.print(f"  Average RTF: {avg_rtf:.4f}")
    console.print(f"  Real-time factor: {1/avg_rtf:.1f}x faster than real-time" if avg_rtf < 1 else f"  Slowdown: {1/avg_rtf:.1f}x")
    console.print(f"  Real-time capable: {'YES' if avg_rtf < 1 else 'NO'}")

    # Show last run details
    last_result = results[-1]
    console.print(f"\n[bold]Last Run Details:[/]")
    console.print(f"  Total duration: {last_result.get('total_duration_s', '?')}s")
    console.print(f"  Speech duration: {last_result.get('speech_duration_s', '?')}s")
    console.print(f"  Segments found: {last_result.get('num_segments', 0)}")


if __name__ == "__main__":
    app()

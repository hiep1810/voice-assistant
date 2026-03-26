"""
Main CLI — Typer app with commands: list, setup, transcribe, benchmark.

Design Pattern: Command Pattern (via Typer)
ELI5: Each CLI subcommand is a function. Typer wires them up so you can call
      `python -m stt_test <command> <args>` from the terminal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from stt_test.benchmark import print_benchmark_table, run_benchmark
from stt_test.env_manager import is_env_ready, run_in_env, setup_env
from stt_test.registry import MODELS, get_model, list_models

app = typer.Typer(
    name="stt-test",
    help="CLI tool for benchmarking Vietnamese STT/ASR models.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# LIST — show all available models
# ---------------------------------------------------------------------------
@app.command("list")
def list_cmd() -> None:
    """List all available ASR models and their setup status."""
    table = Table(title="Available STT Models", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Display Name", style="bold")
    table.add_column("HuggingFace ID", style="dim")
    table.add_column("Status", justify="center")

    for model in list_models():
        status = "[green]✓ ready[/]" if is_env_ready(model.name) else "[yellow]⬡ not set up[/]"
        table.add_row(model.name, model.display_name, model.huggingface_id, status)

    console.print(table)


# ---------------------------------------------------------------------------
# SETUP — create venv and install packages for a model
# ---------------------------------------------------------------------------
@app.command("setup")
def setup_cmd(
    model_name: Optional[str] = typer.Argument(  # noqa: UP007
        None,
        help="Model to set up (e.g. 'parakeet'). Omit to set up all.",
    ),
    all_models: bool = typer.Option(
        False, "--all", help="Set up environments for ALL models."
    ),
) -> None:
    """Create an isolated venv and install packages for a model."""
    if all_models or model_name is None and all_models is False:
        # If --all or no argument, set up everything
        if model_name is None and not all_models:
            console.print("[yellow]No model specified. Use --all to set up everything.[/]")
            raise typer.Exit(1)

    targets = list(MODELS.keys()) if all_models else [model_name]

    for name in targets:
        model = get_model(name)
        console.print(f"\n[bold]Setting up {model.display_name}...[/]")
        try:
            setup_env(model)
        except Exception as e:
            console.print(f"[red]Failed to set up {model.display_name}:[/] {e}")
            if not all_models:
                raise typer.Exit(1)


# ---------------------------------------------------------------------------
# TRANSCRIBE — run a single model on an audio file
# ---------------------------------------------------------------------------
@app.command("transcribe")
def transcribe_cmd(
    model_name: str = typer.Argument(help="Model to use (e.g. 'parakeet')."),
    audio: str = typer.Argument(help="Path to the audio file (.wav, .mp3, .flac, etc)."),
) -> None:
    """Transcribe an audio file using a specific model."""
    audio_path = Path(audio).resolve()
    if not audio_path.exists():
        console.print(f"[red]Audio file not found:[/] {audio_path}")
        raise typer.Exit(1)

    model = get_model(model_name)
    console.print(f"[bold]Transcribing with {model.display_name}...[/]")

    result = run_in_env(model_name, str(audio_path))

    # Display result
    console.print(f"\n[bold green]📝 Transcription:[/]")
    console.print(f"  {result.get('text', '(no text)')}\n")

    console.print(f"  [dim]Audio duration:[/]  {result.get('audio_duration_s', '?')}s")
    console.print(f"  [dim]Inference time:[/]  {result.get('inference_time_s', '?')}s")
    rtf = result.get("rtf", 0)
    rtf_color = "green" if rtf < 1.0 else "red"
    console.print(f"  [dim]RTF:[/]             [{rtf_color}]{rtf:.3f}[/{rtf_color}]")
    console.print(f"  [dim]Real-time:[/]       {'✅ Yes' if result.get('is_realtime') else '❌ No'}")
    console.print(f"  [dim]Device:[/]          {result.get('device', '?')}")


# ---------------------------------------------------------------------------
# BENCHMARK — run all (or selected) models and compare RTF
# ---------------------------------------------------------------------------
@app.command("benchmark")
def benchmark_cmd(
    audio: str = typer.Argument(help="Path to the audio file (.wav, .mp3, .flac, etc)."),
    models: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--models",
        help="Comma-separated model names to benchmark. Default: all.",
    ),
) -> None:
    """Benchmark all (or selected) models on the same audio and compare RTF.

    RTF (Real Time Factor) = inference_time / audio_duration.
    RTF < 1.0 means the model is faster than real-time.
    """
    audio_path = Path(audio).resolve()
    if not audio_path.exists():
        console.print(f"[red]Audio file not found:[/] {audio_path}")
        raise typer.Exit(1)

    model_names = models.split(",") if models else None

    console.print(f"[bold]🎤 Benchmarking on:[/] {audio_path.name}")
    console.print()

    results = run_benchmark(str(audio_path), model_names)
    print_benchmark_table(results)


# ---------------------------------------------------------------------------
# BATCH-BENCHMARK — run models on a directory of audio files
# ---------------------------------------------------------------------------
@app.command("batch-benchmark")
def batch_benchmark_cmd(
    data_dir: str = typer.Argument(help="Directory with .wav and .txt files"),
    models: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--models",
        help="Comma-separated model names to benchmark. Default: all.",
    ),
    limit: Optional[int] = typer.Option(  # noqa: UP007
        None,
        "--limit",
        help="Limit number of samples to process",
    ),
) -> None:
    """Run batch benchmark on a directory of audio files with ground truth.

    Computes WER (Word Error Rate) and CER (Character Error Rate) across all samples.
    Expects files named: 0000.wav, 0000.txt, 0001.wav, 0001.txt, etc.
    """
    from stt_test.batch_benchmark import run_batch_benchmark, print_batch_results

    model_names = models.split(",") if models else None

    results = run_batch_benchmark(data_dir, model_names, limit)
    print_batch_results(results)


if __name__ == "__main__":
    app()

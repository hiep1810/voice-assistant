"""
TTS CLI — Typer app with commands: list, setup, synthesize, benchmark.

Design Pattern: Command Pattern (via Typer)
ELI5: Each CLI subcommand is a function. Typer wires them up so you can call
      `python -m tts_test <command> <args>` from the terminal.

Mirrors stt_test/cli.py for consistency with ASR benchmarking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tts_test.batch_benchmark import run_batch_tts_benchmark, print_batch_tts_results
from tts_test.benchmark import print_tts_benchmark_table, run_tts_benchmark
from tts_test.benchmark_vietneu import benchmark_vietneu, print_benchmark_table
from tts_test.benchmark_vietneu_all import run_all_benchmarks
from tts_test.env_manager import is_tts_env_ready, run_in_tts_env, setup_tts_env, setup_all_tts_envs
from tts_test.registry import TTS_MODELS, get_tts_model, list_tts_models

app = typer.Typer(
    name="tts-test",
    help="CLI tool for benchmarking Vietnamese TTS (Text-to-Speech) models.",
    add_completion=False,
)
# Configure Rich for UTF-8 output on Windows
console = Console(force_terminal=True, force_interactive=False)


# ---------------------------------------------------------------------------
# LIST — show all available TTS models
# ---------------------------------------------------------------------------
@app.command("list")
def list_cmd() -> None:
    """List all available TTS models and their setup status."""
    table = Table(title="Available TTS Models", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Display Name", style="bold")
    table.add_column("HuggingFace ID", style="dim")
    table.add_column("Speaker?", justify="center")
    table.add_column("Status", justify="center")

    for model in list_tts_models():
        status = "[green]ready[/]" if is_tts_env_ready(model.name) else "[yellow]not set up[/]"
        speaker = "Yes" if model.requires_speaker else "No"
        hf_id = model.huggingface_id or "N/A"
        table.add_row(
            model.name,
            model.display_name,
            hf_id,
            speaker,
            status,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# SETUP — create venv and install packages for a TTS model
# ---------------------------------------------------------------------------
@app.command("setup")
def setup_cmd(
    model_name: Optional[str] = typer.Argument(  # noqa: UP007
        None,
        help="Model to set up (e.g. 'vietts'). Omit with --all to set up all.",
    ),
    all_models: bool = typer.Option(
        False, "--all", help="Set up environments for ALL models."
    ),
) -> None:
    """Create an isolated venv and install packages for a TTS model."""
    if all_models:
        targets = list(TTS_MODELS.keys())
    elif model_name is None:
        console.print("[yellow]No model specified. Use --all to set up everything.[/]")
        raise typer.Exit(1)
    else:
        targets = [model_name]

    for name in targets:
        model = get_tts_model(name)
        console.print(f"\n[bold]Setting up {model.display_name}...[/]")
        try:
            setup_tts_env(model)
        except Exception as e:
            console.print(f"[red]Failed to set up {model.display_name}:[/] {e}")
            if not all_models:
                raise typer.Exit(1)


# ---------------------------------------------------------------------------
# SYNTHESIZE — run a single TTS model on text
# ---------------------------------------------------------------------------
@app.command("synthesize")
def synthesize_cmd(
    model_name: str = typer.Argument(help="Model to use (e.g. 'vietts')."),
    text: str = typer.Argument(help="Vietnamese text to synthesize."),
    output: Optional[str] = typer.Option(  # noqa: UP007
        None, "--output", "-o", help="Output audio file path (auto-generated if not provided)."
    ),
    speaker: Optional[str] = typer.Option(  # noqa: UP007
        None, "--speaker", "-s", help="Speaker/voice ID (if model supports it)."
    ),
) -> None:
    """Synthesize Vietnamese text using a specific TTS model."""
    model = get_tts_model(model_name)

    # Auto-generate output path if not provided
    if output is None:
        import tempfile
        output = tempfile.mktemp(suffix=f"_{model_name}_output.wav")

    console.print(f"[bold]Synthesizing with {model.display_name}...[/]")

    result = run_in_tts_env(model_name, text, output, speaker)

    # Display result
    console.print(f"\n[bold green]Synthesis Result:[/]")
    # Display Vietnamese text as ASCII-safe (avoid Windows console encoding issues)
    text_ascii = text.encode('ascii', errors='replace').decode('ascii')
    if len(text) > 50:
        text_ascii = text_ascii[:50] + "..."
    console.print(f"  [dim]Text:[/] {text_ascii} ({len(text)} chars)")
    console.print(f"  [dim]Output:[/] {result.get('output_path', 'N/A')}")
    console.print(f"  [dim]Audio duration:[/] {result.get('audio_duration_s', '?')}s")
    console.print(f"  [dim]Inference time:[/] {result.get('inference_time_s', '?')}s")
    rtf = result.get("rtf", 0)
    rtf_color = "green" if rtf < 1.0 else "red"
    console.print(f"  [dim]RTF:[/] [{rtf_color}]{rtf:.3f}[/{rtf_color}]")
    console.print(f"  [dim]Real-time:[/] {'YES' if result.get('is_realtime') else 'NO'}")
    console.print(f"  [dim]Device:[/] {result.get('device', '?')}")

    # Audio quality metrics if available
    if result.get('mos_predicted'):
        mos = result['mos_predicted']
        mos_color = "green" if mos > 3.5 else "yellow" if mos > 2.5 else "red"
        console.print(f"  [dim]MOS Predicted:[/] [{mos_color}]{mos:.2f}[/{mos_color}]")


# ---------------------------------------------------------------------------
# BENCHMARK — run all (or selected) models and compare RTF
# ---------------------------------------------------------------------------
@app.command("benchmark")
def benchmark_cmd(
    text: str = typer.Argument(help="Vietnamese text to synthesize."),
    models: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--models",
        help="Comma-separated model names to benchmark. Default: all.",
    ),
    output: Optional[str] = typer.Option(  # noqa: UP007
        None, "--output", "-o", help="Output directory for audio files."
    ),
    speaker: Optional[str] = typer.Option(  # noqa: UP007
        None, "--speaker", "-s", help="Speaker/voice ID (if model supports it)."
    ),
) -> None:
    """Benchmark all (or selected) TTS models on the same text and compare RTF.

    RTF (Real Time Factor) = inference_time / audio_duration.
    RTF < 1.0 means the model is faster than real-time.
    """
    model_names = models.split(",") if models else None

    console.print(f"[bold]TTS Benchmark:[/]")
    # Use ASCII-safe text display for Vietnamese
    text_ascii = text[:40].encode('ascii', errors='replace').decode('ascii')
    if len(text) > 40:
        text_ascii += "..."
    console.print(f"  Text: {text_ascii} ({len(text)} chars)")
    console.print(f"  Models: {model_names or 'all'}")
    console.print()

    results = run_tts_benchmark(text, model_names, output, speaker)
    print_tts_benchmark_table(results)


# ---------------------------------------------------------------------------
# BATCH-BENCHMARK — run models on a directory of text files
# ---------------------------------------------------------------------------
@app.command("batch-benchmark")
def batch_benchmark_cmd(
    data_dir: str = typer.Argument(help="Directory with .txt and optional .wav files"),
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
    output: Optional[str] = typer.Option(  # noqa: UP007
        None, "--output", "-o", help="Output directory for audio files."
    ),
) -> None:
    """Run batch benchmark on a directory of text files.

    Expects files named: 0000.txt, 0000.wav (optional reference audio), etc.
    Computes aggregate metrics: Avg RTF, MOS, PESQ, STOI.
    """
    model_names = models.split(",") if models else None

    console.print(f"[bold]TTS Batch Benchmark:[/]")
    console.print(f"  Data: {data_dir}")
    console.print(f"  Models: {model_names or 'all'}")
    console.print()

    results = run_batch_tts_benchmark(data_dir, model_names, limit, output)
    print_batch_tts_results(results)


# ---------------------------------------------------------------------------
# BENCHMARK-VIETNEU — multi-backend benchmark for VieNeu-TTS
# ---------------------------------------------------------------------------
@app.command("benchmark-vietneu")
def benchmark_vietneu_cmd(
    device: str = typer.Option("cuda", "--device", "-d",
                               help="Device to use (cuda, cpu, mps)"),
    backend: str = typer.Option("standard", "--backend", "-b",
                                help="Backend to use (standard, torch-compile, lmdeploy)"),
    iters: int = typer.Option(5, "--iters", "-i", help="Number of iterations"),
    text: str = typer.Option("Xin chao, day la bai kiem tra giong noi tieng Viet cua cac mo hinh TTS.",
                             "--text", "-t", help="Text to synthesize"),
    save_out: Optional[str] = typer.Option(None, "--save-out", "-o",
                                           help="Save output audio to file"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
) -> None:
    """Benchmark VieNeu-TTS with different backends.

    Compares performance across:
    - Standard (PyTorch/CUDA)
    - torch.compile (PyTorch 2.0+ optimization)
    - LMDeploy (if available)
    """
    from dataclasses import asdict
    import json

    # Check CUDA availability
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]CUDA not available, falling back to CPU[/]")
        device = "cpu"

    console.print(f"[bold]VieNeu-TTS Multi-Backend Benchmark[/]")
    console.print(f"  Device: {device}")
    console.print(f"  Backend: {backend}")
    console.print(f"  Iterations: {iters}")
    console.print()

    result = benchmark_vietneu(
        device=device,
        backend=backend,
        iterations=iters,
        text=text,
        save_output=save_out,
    )

    if json_output:
        print(json.dumps(asdict(result), indent=2))
    else:
        print_benchmark_table([result])


# ---------------------------------------------------------------------------
# BENCHMARK-VIETNEU-ALL — compare all backends
# ---------------------------------------------------------------------------
@app.command("benchmark-vietneu-all")
def benchmark_vietneu_all_cmd(
    device: str = typer.Option("cuda", "--device", "-d",
                               help="Device to use (cuda, cpu, mps)"),
    iters: int = typer.Option(5, "--iters", "-i", help="Number of iterations"),
    text: str = typer.Option("Xin chao, day la bai kiem tra giong noi tieng Viet cua cac mo hinh TTS.",
                             "--text", "-t", help="Text to synthesize"),
    save_output: bool = typer.Option(False, "--save-output", help="Save output audio files"),
    skip_lmdeploy: bool = typer.Option(False, "--skip-lmdeploy", help="Skip LMDeploy run"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    out_dir: str = typer.Option("benchmark_runs", "--out-dir", help="Output directory for summaries"),
) -> None:
    """Benchmark VieNeu-TTS across ALL backends and compare results.

    Runs standard, torch-compile, and LMDeploy (if available) backends,
    then displays a comparison table with speedup metrics.
    """
    from dataclasses import asdict
    import json
    from datetime import datetime
    from pathlib import Path

    # Check CUDA availability
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]CUDA not available, falling back to CPU[/]")
        device = "cpu"

    console.print(f"[bold]VieNeu-TTS Multi-Backend Benchmark Comparison[/]")
    console.print(f"  Device: {device}")
    console.print(f"  Iterations: {iters}")
    console.print(f"  Skip LMDeploy: {skip_lmdeploy}")
    console.print()

    results = run_all_benchmarks(
        device=device,
        iterations=iters,
        text=text,
        save_output=save_output,
    )

    if json_output:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print_benchmark_table(results)

        # Write summary
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        runs = []
        for r in results:
            runs.append({
                "name": f"gpu_{r.backend}",
                "metrics": {
                    "avg_time_s": r.avg_inference_time_s,
                    "audio_dur_s": r.audio_duration_s,
                    "rtf": r.rtf_avg,
                    "peak_gpu_mb": r.peak_gpu_memory_mb,
                    "iters": r.iterations,
                }
            })

        # Import write_summary from benchmark_vietneu_all
        from tts_test.benchmark_vietneu_all import write_summary
        write_summary(out_path, ts, runs)




if __name__ == "__main__":
    app()

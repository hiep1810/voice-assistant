"""
TTS Benchmark Module — runs all (or selected) TTS models on the same text, collects RTF metrics.

Design Pattern: Template Method
ELI5: Same steps for each model (setup check → run → collect metrics), but each
      model fills in different implementation details via its own inference script.
Why Rich tables: Clear side-by-side comparison for deciding which model is viable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rich.console import Console
from rich.table import Table

from tts_test.env_manager import is_tts_env_ready, run_in_tts_env
from tts_test.registry import TTS_MODELS, get_tts_model

console = Console()


def run_tts_benchmark(
    text: str,
    model_names: list[str] | None = None,
    output_dir: str | None = None,
    speaker: str | None = None,
) -> list[dict]:
    """Run synthesis on the given text with each model and return results.

    Args:
        text: Vietnamese text to synthesize.
        model_names: Specific models to benchmark. If None, benchmarks all.
        output_dir: Optional directory to save synthesized audio files.
        speaker: Optional speaker/voice ID for models that require it.

    Returns:
        List of result dicts (one per model), each containing RTF metrics.
    """
    targets = model_names if model_names else list(TTS_MODELS.keys())
    results: list[dict] = []

    # Create temp directory for audio output if not specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="tts_benchmark_")
        output_path = Path(temp_dir)

    for name in targets:
        model = get_tts_model(name)

        if not is_tts_env_ready(name):
            console.print(
                f"  [yellow]WARNING Skipping {model.display_name}:[/] "
                f"environment not set up. Run: python -m tts_test setup {name}"
            )
            results.append({
                "model": model.display_name,
                "status": "skipped",
                "reason": "env not set up",
            })
            continue

        console.print(f"  [cyan]Running {model.display_name}...[/]")

        # Generate output path
        output_file = output_path / f"{name}_output.wav"

        try:
            result = run_in_tts_env(name, text, str(output_file), speaker)
            result["model"] = model.display_name
            result["status"] = "ok"
            result["output_path"] = str(output_file)
            results.append(result)
            console.print(f"  [green]OK[/] {model.display_name} done")
        except Exception as e:
            console.print(f"  [red]ERROR[/] {model.display_name} failed: {e}")
            results.append({
                "model": model.display_name,
                "status": "error",
                "reason": str(e),
            })

    return results


def print_tts_benchmark_table(results: list[dict]) -> None:
    """Print a Rich comparison table of TTS benchmark results.

    Columns: Model | Text Len | Audio(s) | Time(s) | RTF | Real-time? | Device | Output
    """
    table = Table(
        title="TTS Benchmark Results",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )

    table.add_column("Model", style="cyan", ratio=2)
    table.add_column("Text Len", justify="right", min_width=7)
    table.add_column("Audio", justify="right", min_width=6)
    table.add_column("Time", justify="right", min_width=6)
    table.add_column("RTF", justify="right", min_width=6)
    table.add_column("RT?", justify="center", min_width=4)
    table.add_column("Dev", justify="center", min_width=6)
    table.add_column("Output", ratio=1, min_width=15, overflow="fold")

    for r in results:
        if r.get("status") == "skipped":
            table.add_row(
                r["model"], "-", "-", "-", "-", "SKIP", "-",
                f"[dim]{r.get('reason', '')}[/]",
            )
        elif r.get("status") == "error":
            table.add_row(
                r["model"], "-", "-", "-", "-", "ERR", "-",
                f"[red]{r.get('reason', '')[:40]}[/]",
            )
        else:
            # Successful result — show all metrics
            rtf = r.get("rtf", 0)
            is_rt = "YES" if r.get("is_realtime", False) else "NO"

            # Output path preview
            output_preview = r.get("output_path", "N/A")
            if len(output_preview) > 30:
                output_preview = "..." + output_preview[-27:]

            table.add_row(
                r["model"],
                str(r.get("text_length", 0)),
                f"{r.get('audio_duration_s', 0):.2f}",
                f"{r.get('inference_time_s', 0):.2f}",
                f"{rtf:.3f}",
                is_rt,
                r.get("device", "?"),
                output_preview,
            )

    console.print()
    console.print(table)
    console.print()

    # Summary line
    ok_results = [r for r in results if r.get("status") == "ok"]
    if ok_results:
        fastest = min(ok_results, key=lambda r: r.get("rtf", float("inf")))
        console.print(
            f"  [bold green]Fastest:[/] {fastest['model']} "
            f"(RTF={fastest.get('rtf', 0):.3f})"
        )

        realtime_models = [r for r in ok_results if r.get("is_realtime", False)]
        console.print(
            f"  [bold]Real-time capable:[/] {len(realtime_models)}/{len(ok_results)} models"
        )

        # Show output directory
        if ok_results and ok_results[0].get("output_path"):
            output_dir = Path(ok_results[0]["output_path"]).parent
            console.print(f"  [dim]Audio saved to:[/] {output_dir}")

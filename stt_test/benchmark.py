"""
Benchmark Module — runs all (or selected) models on the same audio, collects RTF metrics.

Design Pattern: Template Method
ELI5: Same steps for each model (setup check → run → collect metrics), but each
      model fills in different implementation details via its own inference script.
Why Rich tables: Clear side-by-side comparison for deciding which model is viable
                 for real-time use.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from stt_test.env_manager import is_env_ready, run_in_env
from stt_test.registry import MODELS, get_model

console = Console()


def run_benchmark(
    audio_path: str,
    model_names: list[str] | None = None,
) -> list[dict]:
    """Run inference on the given audio with each model and return results.

    Args:
        audio_path: Path to the audio file.
        model_names: Specific models to benchmark. If None, benchmarks all.

    Returns:
        List of result dicts (one per model), each containing RTF metrics.
    """
    targets = model_names if model_names else list(MODELS.keys())
    results: list[dict] = []

    for name in targets:
        model = get_model(name)

        if not is_env_ready(name):
            console.print(
                f"  [yellow]⚠ Skipping {model.display_name}:[/] "
                f"environment not set up. Run: python -m stt_test setup {name}"
            )
            results.append({
                "model": model.display_name,
                "status": "skipped",
                "reason": "env not set up",
            })
            continue

        console.print(f"  [cyan]Running {model.display_name}...[/]")

        try:
            result = run_in_env(name, audio_path)
            result["model"] = model.display_name
            result["status"] = "ok"
            results.append(result)
            console.print(f"  [green]✓[/] {model.display_name} done")
        except Exception as e:
            console.print(f"  [red]✗[/] {model.display_name} failed: {e}")
            results.append({
                "model": model.display_name,
                "status": "error",
                "reason": str(e),
            })

    return results


def print_benchmark_table(results: list[dict]) -> None:
    """Print a Rich comparison table of benchmark results.

    Columns: Model | Audio(s) | Time(s) | RTF | Real-time? | Device | Text
    """
    table = Table(
        title="🎤 STT Benchmark Results",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )

    table.add_column("Model", style="cyan", ratio=2)
    table.add_column("Audio", justify="right", min_width=6)
    table.add_column("Time", justify="right", min_width=6)
    table.add_column("RTF", justify="right", min_width=6)
    table.add_column("RT?", justify="center", min_width=4)
    table.add_column("Dev", justify="center", min_width=6)
    table.add_column("Text Preview", ratio=1, min_width=10, overflow="fold")

    for r in results:
        if r.get("status") == "skipped":
            table.add_row(
                r["model"], "—", "—", "—", "⚠️ skip", "—",
                f"[dim]{r.get('reason', '')}[/]",
            )
        elif r.get("status") == "error":
            table.add_row(
                r["model"], "—", "—", "—", "❌ err", "—",
                f"[red]{r.get('reason', '')[:40]}[/]",
            )
        else:
            # Successful result — show all metrics
            rtf = r.get("rtf", 0)
            is_rt = "✅" if r.get("is_realtime", False) else "❌"
            # If the text is empty or missing, show [empty]
            text_preview = r.get("text", "").strip() or "[empty]"
            if len(text_preview) > 60:
                text_preview = text_preview[:57] + "..."

            table.add_row(
                r["model"],
                f"{r.get('audio_duration_s', 0):.2f}",
                f"{r.get('inference_time_s', 0):.2f}",
                f"{rtf:.3f}",
                is_rt,
                r.get("device", "?"),
                text_preview,
            )

    console.print()
    console.print(table)
    console.print()

    # Summary line
    ok_results = [r for r in results if r.get("status") == "ok"]
    if ok_results:
        fastest = min(ok_results, key=lambda r: r.get("rtf", float("inf")))
        console.print(
            f"  [bold green]🏆 Fastest:[/] {fastest['model']} "
            f"(RTF={fastest.get('rtf', 0):.3f})"
        )

        realtime_models = [r for r in ok_results if r.get("is_realtime", False)]
        console.print(
            f"  [bold]Real-time capable:[/] {len(realtime_models)}/{len(ok_results)} models"
        )

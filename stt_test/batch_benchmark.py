"""
Batch Benchmark — run models on multiple audio files and compute aggregate metrics.

Usage:
    python -m stt_test batch-benchmark ./data/vivos/test --models parakeet,gipformer
"""

import argparse
import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from stt_test.env_manager import is_env_ready, run_in_env
from stt_test.registry import get_model

app = typer.Typer(name="batch-benchmark")
console = Console()


def run_batch_benchmark(
    data_dir: str,
    model_names: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    """Run benchmark on multiple audio files in a directory.

    Args:
        data_dir: Directory containing .wav files and .txt transcriptions.
        model_names: Models to benchmark.
        limit: Limit number of samples to process.

    Returns:
        Dict with per-model metrics: {model_name: {wer, cer, avg_rtf, ...}}
    """
    data_path = Path(data_dir)

    # Find all audio files
    audio_files = sorted(data_path.glob("*.wav"))
    if not audio_files:
        raise ValueError(f"No .wav files found in {data_path}")

    if limit:
        audio_files = audio_files[:limit]

    console.print(f"\n[bold]🎤 Batch Benchmark[/]")
    console.print(f"  Directory: {data_path}")
    console.print(f"  Files: {len(audio_files)} samples")
    console.print(f"  Models: {model_names or 'all'}")
    console.print()

    # Load ground truth
    ground_truth = {}
    for audio_file in audio_files:
        txt_file = audio_file.with_suffix(".txt")
        if txt_file.exists():
            with open(txt_file, "r", encoding="utf-8") as f:
                ground_truth[audio_file.name] = f.read().strip()
        else:
            ground_truth[audio_file.name] = ""

    # Run benchmark for each model
    results = {}
    for model_name in (model_names or []):
        console.print(f"\n[cyan]Running {model_name}...[/]")

        if not is_env_ready(model_name):
            console.print(f"  [yellow]⚠ Skipping:[/] environment not set up")
            continue

        model = get_model(model_name)
        model_results = []

        for i, audio_file in enumerate(audio_files):
            try:
                result = run_in_env(model_name, str(audio_file))
                result["audio_file"] = audio_file.name
                result["ground_truth"] = ground_truth.get(audio_file.name, "")
                model_results.append(result)

                if (i + 1) % 10 == 0:
                    console.print(f"  [{i + 1}/{len(audio_files)}]...")
            except Exception as e:
                console.print(f"  [red]✗ {audio_file.name}:[/] {e}")
                model_results.append({
                    "audio_file": audio_file.name,
                    "error": str(e),
                })

        # Compute aggregate metrics
        results[model_name] = compute_metrics(model_results, model.display_name)

    return results


def compute_metrics(results: list[dict], model_name: str) -> dict:
    """Compute WER, CER, and average RTF from results."""
    # Simple WER/CER calculation (without normalization)
    total_wer = 0
    total_cer = 0
    total_rtf = 0
    count = 0

    for r in results:
        if "error" in r:
            continue

        pred = r.get("text", "").strip().lower()
        truth = r.get("ground_truth", "").strip().lower()

        if not truth:
            continue

        # Character Error Rate
        cer = compute_cer(pred, truth)
        total_cer += cer

        # Word Error Rate
        wer = compute_wer(pred, truth)
        total_wer += wer

        # RTF
        total_rtf += r.get("rtf", 0)
        count += 1

    if count == 0:
        return {"status": "error", "reason": "no valid results"}

    return {
        "status": "ok",
        "samples": count,
        "wer": total_wer / count,
        "cer": total_cer / count,
        "avg_rtf": total_rtf / count,
    }


def compute_cer(pred: str, truth: str) -> float:
    """Compute Character Error Rate."""
    if not truth:
        return 1.0

    # Simple edit distance
    edits = levenshtein_distance(pred, truth)
    return edits / len(truth)


def compute_wer(pred: str, truth: str) -> float:
    """Compute Word Error Rate."""
    pred_words = pred.split()
    truth_words = truth.split()

    if not truth_words:
        return 1.0

    edits = levenshtein_distance(pred_words, truth_words)
    return edits / len(truth_words)


def levenshtein_distance(s1: list | str, s2: list | str) -> int:
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def print_batch_results(results: dict) -> None:
    """Print batch benchmark results."""
    table = Table(
        title="📊 Batch Benchmark Results",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Model", style="cyan")
    table.add_column("Samples", justify="right")
    table.add_column("WER", justify="right")
    table.add_column("CER", justify="right")
    table.add_column("Avg RTF", justify="right")

    for model_name, metrics in results.items():
        if metrics.get("status") == "error":
            table.add_row(model_name, "—", "—", "—", "—")
            continue

        wer_color = "green" if metrics["wer"] < 0.1 else "yellow" if metrics["wer"] < 0.2 else "red"
        cer_color = "green" if metrics["cer"] < 0.05 else "yellow" if metrics["cer"] < 0.1 else "red"
        rtf_color = "green" if metrics["avg_rtf"] < 1.0 else "red"

        table.add_row(
            model_name,
            str(metrics["samples"]),
            f"[{wer_color}]{metrics['wer']:.2%}[/{wer_color}]",
            f"[{cer_color}]{metrics['cer']:.2%}[/{cer_color}]",
            f"[{rtf_color}]{metrics['avg_rtf']:.4f}[/{rtf_color}]",
        )

    console.print()
    console.print(table)
    console.print()


@app.command()
def main(
    data_dir: str = typer.Argument(help="Directory with .wav and .txt files"),
    models: str = typer.Option(None, "--models", help="Comma-separated model names"),
    limit: int = typer.Option(None, "--limit", help="Limit number of samples"),
):
    """Run batch benchmark on a directory of audio files."""
    model_names = models.split(",") if models else None

    results = run_batch_benchmark(data_dir, model_names, limit)
    print_batch_results(results)


if __name__ == "__main__":
    app()

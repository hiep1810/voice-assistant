"""
TTS Batch Benchmark Module — synthesizes multiple texts and computes aggregate metrics.

Extends single-text benchmark to handle datasets with ground truth reference audio.
Computes:
- Average RTF across all samples
- PESQ (if reference audio available)
- STOI (if reference audio available)
- MOS prediction scores
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from tts_test.env_manager import is_tts_env_ready, run_in_tts_env
from tts_test.registry import TTS_MODELS, get_tts_model
from tts_test.audio_quality import compute_tts_quality_metrics, format_quality_report

console = Console()


def run_batch_tts_benchmark(
    data_dir: str,
    model_names: List[str] | None = None,
    limit: Optional[int] = None,
    output_dir: str | None = None,
) -> dict:
    """Run TTS benchmark on multiple texts in a directory.

    Expects files named: 0000.txt, 0000.wav (optional reference), etc.

    Args:
        data_dir: Directory containing .txt files and optional .wav reference audio.
        model_names: Models to benchmark.
        limit: Limit number of samples to process.
        output_dir: Directory to save synthesized audio.

    Returns:
        Dict with per-model metrics: {model_name: {avg_rtf, pesq, stoi, mos, ...}}
    """
    data_path = Path(data_dir)

    # Find all text files
    text_files = sorted(data_path.glob("*.txt"))
    if not text_files:
        raise ValueError(f"No .txt files found in {data_path}")

    if limit:
        text_files = text_files[:limit]

    console.print(f"\n[bold]TTS Batch Benchmark[/]")
    console.print(f"  Directory: {data_path}")
    console.print(f"  Files: {len(text_files)} samples")
    console.print(f"  Models: {model_names or 'all'}")
    console.print()

    # Create output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        import tempfile
        output_path = Path(tempfile.mkdtemp(prefix="tts_batch_"))

    # Run benchmark for each model
    results = {}
    targets = model_names if model_names else list(TTS_MODELS.keys())

    for model_name in targets:
        console.print(f"\n[cyan]Running {model_name}...[/]")

        if not is_tts_env_ready(model_name):
            console.print(f"  [yellow]WARNING Skipping:[/] environment not set up")
            continue

        model = get_tts_model(model_name)
        model_results = []

        for i, text_file in enumerate(text_files):
            try:
                # Load text
                with open(text_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                # Generate output path
                output_file = output_path / f"{model_name}_{text_file.stem}.wav"

                # Run synthesis
                result = run_in_tts_env(model_name, text, str(output_file))
                result["text_file"] = text_file.name
                result["text"] = text
                result["output_path"] = str(output_file)

                # Find reference audio if exists
                ref_audio = text_file.with_suffix(".wav")
                if ref_audio.exists():
                    result["reference_path"] = str(ref_audio)

                model_results.append(result)

                if (i + 1) % 10 == 0:
                    console.print(f"  [{i + 1}/{len(text_files)}]...")

            except Exception as e:
                console.print(f"  [red]ERROR {text_file.name}:[/] {e}")
                model_results.append({
                    "text_file": text_file.name,
                    "error": str(e),
                })

        # Compute aggregate metrics
        results[model_name] = compute_batch_metrics(model_results, model.display_name)

    return results


def compute_batch_metrics(results: List[dict], model_name: str) -> dict:
    """Compute aggregate metrics from batch results."""
    total_rtf = 0
    total_mos = 0
    total_pesq = 0
    total_stoi = 0
    pesq_count = 0
    stoi_count = 0
    count = 0

    for r in results:
        if "error" in r:
            continue

        # RTF
        total_rtf += r.get("rtf", 0)

        # Audio quality metrics
        output_path = r.get("output_path")
        ref_path = r.get("reference_path")

        if output_path:
            try:
                quality = compute_tts_quality_metrics(output_path, ref_path)
                if quality.get('mos_predicted'):
                    total_mos += quality['mos_predicted']
                if quality.get('pesq') is not None:
                    total_pesq += quality['pesq']
                    pesq_count += 1
                if quality.get('stoi'):
                    total_stoi += quality['stoi']
                    stoi_count += 1
            except Exception:
                pass

        count += 1

    if count == 0:
        return {"status": "error", "reason": "no valid results"}

    return {
        "status": "ok",
        "samples": count,
        "avg_rtf": total_rtf / count,
        "avg_mos": total_mos / count if total_mos > 0 else None,
        "avg_pesq": total_pesq / pesq_count if pesq_count > 0 else None,
        "avg_stoi": total_stoi / stoi_count if stoi_count > 0 else None,
    }


def print_batch_tts_results(results: dict) -> None:
    """Print batch TTS benchmark results."""
    table = Table(
        title="TTS Batch Benchmark Results",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Model", style="cyan")
    table.add_column("Samples", justify="right")
    table.add_column("Avg RTF", justify="right")
    table.add_column("RT < 1?", justify="center")
    table.add_column("MOS", justify="right")
    table.add_column("PESQ", justify="right")
    table.add_column("STOI", justify="right")

    for model_name, metrics in results.items():
        if metrics.get("status") == "error":
            table.add_row(model_name, "-", "-", "-", "-", "-", "-")
            continue

        # RTF color
        rtf = metrics["avg_rtf"]
        rtf_color = "green" if rtf < 1.0 else "yellow" if rtf < 2.0 else "red"
        is_rt = "YES" if rtf < 1.0 else "NO"

        # MOS color and stars
        mos = metrics.get("avg_mos")
        if mos:
            mos_color = "green" if mos > 3.5 else "yellow" if mos > 2.5 else "red"
            mos_str = f"{mos:.2f}"
        else:
            mos_str = "N/A"

        # PESQ color
        pesq = metrics.get("avg_pesq")
        if pesq:
            pesq_color = "green" if pesq > 3.0 else "yellow" if pesq > 2.0 else "red"
            pesq_str = f"{pesq:.2f}"
        else:
            pesq_str = "N/A"

        # STOI color
        stoi = metrics.get("avg_stoi")
        if stoi:
            stoi_color = "green" if stoi > 0.8 else "yellow" if stoi > 0.7 else "red"
            stoi_str = f"{stoi:.2f}"
        else:
            stoi_str = "N/A"

        table.add_row(
            model_name,
            str(metrics["samples"]),
            f"[{rtf_color}]{rtf:.4f}[/{rtf_color}]",
            is_rt,
            f"[{mos_color}]{mos_str}[/{mos_color}]" if mos else mos_str,
            f"[{pesq_color}]{pesq_str}[/{pesq_color}]" if pesq else pesq_str,
            f"[{stoi_color}]{stoi_str}[/{stoi_color}]" if stoi else stoi_str,
        )

    console.print()
    console.print(table)
    console.print()

    # Summary
    ok_results = [(k, v) for k, v in results.items() if v.get("status") == "ok"]
    if ok_results:
        fastest = min(ok_results, key=lambda x: x[1]["avg_rtf"])
        console.print(
            f"  [bold green]Fastest:[/] {fastest[0]} "
            f"(RTF={fastest[1]['avg_rtf']:.4f})"
        )

        best_mos = max(
            [(k, v) for k, v in ok_results if v.get("avg_mos")],
            key=lambda x: x[1].get("avg_mos", 0),
            default=None
        )
        if best_mos:
            console.print(
                f"  [bold]Best MOS:[/] {best_mos[0]} "
                f"(MOS={best_mos[1]['avg_mos']:.2f})"
            )

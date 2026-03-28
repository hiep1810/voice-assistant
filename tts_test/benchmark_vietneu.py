"""
VieNeu-TTS Multi-Backend Benchmark

Benchmarks VieNeu-TTS across different backends:
- Standard (PyTorch/CUDA)
- Fast (LMDeploy with TurboMindEngine)
- torch.compile (PyTorch 2.0+ optimization)

Usage:
    python -m tts_test.benchmark_vietneu --device cuda --backend standard --iters 5
    python -m tts_test.benchmark_vietneu --device cuda --backend lmdeploy --iters 5
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch


@dataclass
class BenchmarkResult:
    """Benchmark result dataclass."""
    backend: str
    device: str
    iterations: int
    text_length: int
    audio_duration_s: float
    avg_inference_time_s: float
    std_inference_time_s: float
    min_inference_time_s: float
    max_inference_time_s: float
    rtf_avg: float
    rtf_p50: float
    rtf_p95: float
    rtf_p99: float
    peak_gpu_memory_mb: float
    is_realtime: bool
    speedup_vs_standard: float
    timestamp: str


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0.0

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)
    except ImportError:
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return 0.0


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def load_vietneu_model(device: str, backend: str, backbone_repo: str = "pnnbao-ump/VieNeu-TTS", codec_repo: str = "neuphonic/neucodec"):
    """Load VieNeu-TTS model with specified backend."""

    if backend == "standard":
        # Standard PyTorch backend
        from vieneu import VieNeuTTS
        return VieNeuTTS(
            backbone_repo=backbone_repo,
            backbone_device=device,
            codec_repo=codec_repo,
            codec_device=device,
        )

    elif backend == "lmdeploy":
        # LMDeploy backend using FastVieNeuTTS
        print("  [dim]Loading with LMDeploy backend (FastVieNeuTTS)...[/]")
        try:
            from vieneu.fast import FastVieNeuTTS

            return FastVieNeuTTS(
                backbone_repo=backbone_repo,
                backbone_device=device,
                codec_repo=codec_repo,
                codec_device=device,
            )
        except ImportError as e:
            print(f"  [red]LMDeploy not installed![/]")
            print(f"  [yellow]To install LMDeploy:[/]")
            print(f"    1. Enable Windows Long Path Support (Admin PowerShell):")
            print(f"       New-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force")
            print(f"    2. Reboot Windows")
            print(f"    3. pip install lmdeploy")
            print(f"  [yellow]Falling back to standard backend...[/]")
            from vieneu import VieNeuTTS
            return VieNeuTTS(
                backbone_repo=backbone_repo,
                backbone_device=device,
                codec_repo=codec_repo,
                codec_device=device,
            )

    elif backend == "torch-compile":
        # Load with torch.compile optimization
        from vieneu import VieNeuTTS
        model = VieNeuTTS(
            backbone_repo=backbone_repo,
            backbone_device=device,
            codec_repo=codec_repo,
            codec_device=device,
        )
        # Compile the backbone for faster inference
        if hasattr(model, 'backbone') and model.backbone is not None:
            model.backbone = torch.compile(model.backbone, mode="reduce-overhead")
        return model

    else:
        raise ValueError(f"Unknown backend: {backend}")


def benchmark_vietneu(
    device: str = "cuda",
    backend: Literal["standard", "torch-compile", "lmdeploy"] = "standard",
    iterations: int = 5,
    text: str = "Xin chao, day la bai kiem tra giong noi tieng Viet cua cac mo hinh TTS.",
    save_output: Optional[str] = None,
    backbone_repo: str = "pnnbao-ump/VieNeu-TTS",
    codec_repo: str = "neuphonic/neucodec",
) -> BenchmarkResult:
    """
    Benchmark VieNeu-TTS with specified backend.
    """
    print(f"\n{'='*60}")
    print(f"VieNeu-TTS Benchmark")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Backend: {backend}")
    print(f"Backbone: {backbone_repo}")
    print(f"Codec: {codec_repo}")
    print(f"Iterations: {iterations}")
    print(f"Text length: {len(text)} chars")
    print()

    # Reset GPU stats
    reset_gpu_memory_stats()

    # Load model
    print(f"[1] Loading VieNeu-TTS model...")
    start_load = time.perf_counter()
    model = load_vietneu_model(device, backend, backbone_repo, codec_repo)
    load_time = time.perf_counter() - start_load
    print(f"OK: Model loaded in {load_time:.2f}s")

    # Get preset voice
    voice = model.get_preset_voice()
    print(f"[+] Using preset voice")

    # Warmup run
    print(f"\n[2] Warmup...")
    _ = model.infer(text, voice=voice)
    if device == "cuda":
        torch.cuda.synchronize()
    print("OK: Warmup complete.")

    # Benchmark runs
    print(f"\n[3] Running {iterations} iterations...")
    inference_times = []
    rtfs = []
    audio_duration = 0

    for i in range(iterations):
        start = time.perf_counter()
        audio = model.infer(text, voice=voice)
        if device == "cuda":
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start

        inference_times.append(inference_time)
        audio_dur = len(audio) / model.sample_rate
        audio_duration = audio_dur
        rtf = inference_time / audio_dur
        rtfs.append(rtf)

        print(f"   Iter {i+1}: {inference_time:.3f}s")

    # Calculate statistics
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)

    avg_rtf = np.mean(rtfs)
    p50_rtf = np.percentile(rtfs, 50)
    p95_rtf = np.percentile(rtfs, 95)
    p99_rtf = np.percentile(rtfs, 99)

    # GPU memory
    peak_gpu_memory = get_gpu_memory_mb()

    # Real-time check
    is_realtime = avg_rtf < 1.0

    # Save audio if requested
    if save_output:
        output_path = Path(save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        sf.write(str(output_path), audio, model.sample_rate)
        print(f"\nSaved benchmark audio to: {output_path}")

    # Create result
    result = BenchmarkResult(
        backend=backend,
        device=device,
        iterations=iterations,
        text_length=len(text),
        audio_duration_s=round(audio_duration, 3),
        avg_inference_time_s=round(avg_inference_time, 3),
        std_inference_time_s=round(std_inference_time, 3),
        min_inference_time_s=round(min_inference_time, 3),
        max_inference_time_s=round(max_inference_time, 3),
        rtf_avg=round(avg_rtf, 4),
        rtf_p50=round(p50_rtf, 4),
        rtf_p95=round(p95_rtf, 4),
        rtf_p99=round(p99_rtf, 4),
        peak_gpu_memory_mb=round(peak_gpu_memory, 2),
        is_realtime=is_realtime,
        speedup_vs_standard=1.0,
        timestamp=datetime.now().isoformat(),
    )

    return result


def print_benchmark_table(results: list[BenchmarkResult]):
    """Print benchmark results in a table format."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(title="VieNeu-TTS Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Backend", style="cyan")
    table.add_column("Device", justify="center")
    table.add_column("Avg Time (s)", justify="right")
    table.add_column("Std Dev (ms)", justify="right")
    table.add_column("Audio (s)", justify="right")
    table.add_column("RTF Avg", justify="right")
    table.add_column("RTF P95", justify="right")
    table.add_column("RT?", justify="center")
    table.add_column("GPU Mem (MB)", justify="right")
    table.add_column("Speedup", justify="right")

    standard_result = next((r for r in results if r.backend == "standard"), None)

    for r in results:
        speedup = 1.0
        if standard_result and standard_result.avg_inference_time_s > 0:
            speedup = standard_result.avg_inference_time_s / r.avg_inference_time_s

        rtf_color = "green" if r.is_realtime else "red"
        speedup_str = f"{speedup:.1f}x" if speedup != 1.0 else "1.0x"

        table.add_row(
            r.backend,
            r.device,
            f"{r.avg_inference_time_s:.3f}",
            f"{r.std_inference_time_s*1000:.1f}",
            f"{r.audio_duration_s:.3f}",
            f"{r.rtf_avg:.4f}",
            f"{r.rtf_p95:.4f}",
            f"[{rtf_color}]{'YES' if r.is_realtime else 'NO'}[/{rtf_color}]",
            f"{r.peak_gpu_memory_mb:.1f}",
            speedup_str,
        )

    console.print()
    console.print(table)
    console.print()


def print_summary(results: list[BenchmarkResult]):
    """Print summary and insights."""
    from rich.console import Console

    console = Console()

    console.print("\n" + "="*70)
    console.print("[bold]Performance Summary[/]")
    console.print("="*70)

    if not results:
        return

    standard_result = next((r for r in results if r.backend == "standard"), None)
    lmdeploy_result = next((r for r in results if r.backend == "lmdeploy"), None)

    if standard_result:
        console.print(f"Average Time         {standard_result.avg_inference_time_s:.3f}s")
        console.print(f"Audio Duration       {standard_result.audio_duration_s:.3f}s")
        console.print(f"RTF (Avg)            {standard_result.rtf_avg:.3f}")
        console.print(f"Peak GPU Memory      {standard_result.peak_gpu_memory_mb:.2f} MB")
        console.print(f"Iterations           {standard_result.iterations}")

    console.print("")
    console.print("[bold]Insights[/]")
    console.print("")

    if standard_result and lmdeploy_result:
        speedup = standard_result.avg_inference_time_s / lmdeploy_result.avg_inference_time_s
        console.print(f"- Speedup (LMDeploy vs Standard): ~`{speedup:.1f}x`")

    if len(results) > 1:
        fastest = min(results, key=lambda r: r.avg_inference_time_s)
        console.print(f"- Fastest: {fastest.backend} (RTF={fastest.rtf_avg:.3f})")

        realtime = [r for r in results if r.is_realtime]
        if realtime:
            console.print(f"- Real-time capable: {len(realtime)}/{len(results)} backends")


def main():
    parser = argparse.ArgumentParser(description="VieNeu-TTS Multi-Backend Benchmark")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu", "mps"], help="Device to use")
    parser.add_argument("--backend", type=str, default="standard",
                       choices=["standard", "torch-compile", "lmdeploy"],
                       help="Backend to use")
    parser.add_argument("--iters", type=int, default=5, help="Number of iterations")
    parser.add_argument("--text", type=str,
                       default="Xin chao, day la bai kiem tra giong noi tieng Viet cua cac mo hinh TTS.",
                       help="Text to synthesize")
    parser.add_argument("--save_out", type=str, default=None,
                       help="Save output audio to file")
    parser.add_argument("--backbone", type=str, default="pnnbao-ump/VieNeu-TTS",
                       help="Backbone model repository")
    parser.add_argument("--codec", type=str, default="neuphonic/neucodec",
                       help="Codec model repository")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    result = benchmark_vietneu(
        device=args.device,
        backend=args.backend,
        iterations=args.iters,
        text=args.text,
        save_output=args.save_out,
        backbone_repo=args.backbone,
        codec_repo=args.codec,
    )

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print_benchmark_table([result])
        print_summary([result])


if __name__ == "__main__":
    main()

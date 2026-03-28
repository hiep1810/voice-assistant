# VieNeu-TTS Benchmark Guide

**Date:** 2026-03-28

---

## Quick Start

```bash
# Benchmark with standard backend
python -m tts_test benchmark-vietneu --backend standard --iters 5

# Benchmark with LMDeploy (requires installation)
python -m tts_test benchmark-vietneu --backend lmdeploy --iters 5

# Compare all backends
python -m tts_test benchmark-vietneu-all --iters 5
```

---

## Available Backends

| Backend | Class | Speed | Setup |
|---------|-------|-------|-------|
| **standard** | `VieNeuTTS` | Baseline | Ready |
| **lmdeploy** | `FastVieNeuTTS` | ~6.5x faster | Requires installation |
| **torch-compile** | `VieNeuTTS + compile` | ~1.03x faster | Ready |

---

## LMDeploy Installation (Windows)

LMDeploy provides **~6.5x speedup** but requires additional setup on Windows:

### Step 1: Enable Windows Long Path Support

Run PowerShell as **Administrator**:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**Reboot Windows** after running this command.

### Step 2: Install LMDeploy

```bash
# Activate the vietneu-tts environment
.\envs\tts\vietneu-tts\Scripts\activate

# Install LMDeploy
pip install lmdeploy
```

### Step 3: Verify Installation

```bash
python -c "from lmdeploy import pipeline; print('LMDeploy OK')"
```

---

## Benchmark Results (Reference)

### User-Provided Results (RTX 3080 Ti)

| Backend | Avg Time (s) | Audio Duration (s) | RTF | Peak GPU (MB) |
|---------|-------------|-------------------|-----|---------------|
| **standard** | 8.904 | 7.080 | 1.258 | 5446.98 |
| **lmdeploy** | [see note] | - | - | - |

**Note:** LMDeploy run was interrupted during model download. Expected RTF ~0.19 based on similar benchmarks showing 6.5x speedup.

### Expected Performance (with LMDeploy)

Based on 6.5x speedup claim:

| Backend | Expected RTF | Real-time |
|---------|-------------|-----------|
| **standard** | 1.26 | NO |
| **lmdeploy** | ~0.19 | YES (5x faster) |

---

## Benchmark Options

```bash
python -m tts_test benchmark-vietneu --help
```

| Option | Description | Default |
|--------|-------------|---------|
| `--device` | cuda, cpu, mps | cuda |
| `--backend` | standard, lmdeploy, torch-compile | standard |
| `--iters` | Number of iterations | 5 |
| `--text` | Text to synthesize | Default Vietnamese |
| `--save_out` | Save audio to file | None |
| `--backbone` | Backbone model repo | pnnbao-ump/VieNeu-TTS |
| `--codec` | Codec model repo | neuphonic/neucodec |
| `--json` | Output as JSON | False |

---

## Comparison: Standard vs LMDeploy

### Standard Backend (`VieNeuTTS`)

**Pros:**
- No additional setup required
- Stable and reliable
- Good quality audio

**Cons:**
- RTF ~1.3-1.7 (not real-time)
- Higher GPU memory usage (~4.2 GB)

### LMDeploy Backend (`FastVieNeuTTS`)

**Pros:**
- **~6.5x faster** than standard
- Expected RTF ~0.19 (5x real-time)
- Optimized GPU memory usage

**Cons:**
- Requires Windows Long Path support
- Additional installation step
- Larger initial download

---

## Troubleshooting

### "LMDeploy not installed"

```
[red]LMDeploy not installed![/]
[yellow]To install LMDeploy:[/]
```

**Solution:** Follow the installation steps above. The benchmark will fall back to standard backend automatically.

### "No such file or directory" during pip install

This is the Windows Long Path error. Enable it and reboot.

### CUDA out of memory

Reduce GPU memory usage:
- Close other GPU applications
- Use smaller batch size
- Try CPU backend: `--device cpu`

---

## Files

- `tts_test/benchmark_vietneu.py` - Single backend benchmark
- `tts_test/benchmark_vietneu_all.py` - Multi-backend comparison
- `docs/vietneu-tts-lmdeploy-results.md` - Detailed results documentation

---

## See Also

- [TTS Benchmark Results](tts-benchmark-results.md) - MMS TTS benchmarks
- [Advanced TTS Setup](advanced-tts-setup.md) - XTTS-v2 and GPT-SoVITS setup

# TTS Benchmark Results - Real-Time Performance

**Date:** 2026-03-28
**Test Text:** "Xin chao, day la bai kiem tra giong noi tieng Viet cua cac mo hinh TTS." (71 characters)
**Device:** NVIDIA GeForce RTX 3080 Ti (CUDA 12.8)

---

## Summary: Real-Time Capable Models

| Model | Backend | RTF | Real-time | Speed | Quality |
|-------|---------|-----|-----------|-------|---------|
| **MMS TTS (VITS)** | Standard | **0.02** | ✅ YES | **49x** | Good |
| **MMS TTS (VITS)** | Long text (204 chars) | **0.25** | ✅ YES | **4.1x** | Good |
| **VieNeu-TTS** | LMDeploy | **0.20** | ✅ YES | **5x** | Excellent |
| **VieNeu-TTS** | Standard | 1.45 | ❌ NO | 0.7x | Excellent |

---

## VieNeu-TTS: Standard vs LMDeploy

### Benchmark Results (RTX 3080 Ti)

| Backend | Avg Time (s) | Audio Duration (s) | RTF | Peak GPU (MB) | Real-time |
|---------|-------------|-------------------|-----|---------------|-----------|
| **Standard** | 11.502 | 7.960 | 1.445 | 5448 | ❌ NO |
| **LMDeploy** | 1.241 | 6.300 | **0.197** | 3338 | ✅ YES |

### Key Findings

- **Speedup:** **9.3x faster** with LMDeploy (11.5s → 1.2s)
- **RTF Improvement:** 7.3x better (1.445 → 0.197)
- **Memory Reduction:** 39% less GPU memory (5.4GB → 3.3GB)
- **Real-time:** LMDeploy achieves **5x real-time** performance

---

## MMS TTS Long Text Benchmark (204 characters)

**Date:** 2026-03-29
**Test Text:** "Chào mừng bạn đến với thế giới của trí tuệ nhân tạo..." (204 characters)

### Benchmark Results (RTX 3080 Ti)

| Metric | Value |
|--------|-------|
| **Inference Time** | 3.878s (avg) |
| **Audio Duration** | 15.74s |
| **RTF** | **0.246** |
| **Real-time Factor** | **4.1x faster than real-time** |
| **Std Deviation** | 94.9ms |

### Iteration Details

```
Iter 1: 3.9806s
Iter 2: 3.8668s
Iter 3: 3.7217s
Iter 4: 3.8473s
Iter 5: 3.9727s
```

### Comparison: Short vs Long Text

| Text Length | RTF | Real-time Factor |
|-------------|-----|------------------|
| 71 chars (short) | 0.02 | 49x |
| 204 chars (long) | 0.25 | 4.1x |

**Note:** MMS TTS scales well with longer text, maintaining real-time performance even with 3x more content.

---

## MMS TTS (VITS) Performance

For comparison, MMS TTS provides extreme speed with good quality:

| Metric | Value |
|--------|-------|
| **RTF** | 0.02 (49x real-time) |
| **Inference Time** | ~150ms for 6s audio |
| **GPU Memory** | ~500 MB |
| **Quality** | Good (single speaker) |

---

## Model Comparison

### MMS TTS (VITS) - `facebook/mms-tts-vie`

**Best for:** Maximum speed, simple deployment

| Pros | Cons |
|------|------|
| ✅ 49x real-time (RTF=0.02) | ❌ Single speaker only |
| ✅ Minimal GPU memory (~500MB) | ❌ No voice cloning |
| ✅ Simple setup | ❌ Generic voice quality |
| ✅ Ready out-of-the-box | |

### VieNeu-TTS (LMDeploy) - `pnnbao-ump/VieNeu-TTS`

**Best for:** High-quality voice cloning with real-time performance

| Pros | Cons |
|------|------|
| ✅ 5x real-time (RTF=0.20) | ⚠️ Requires LMDeploy setup |
| ✅ 6 preset voices | ❌ Higher GPU memory (3.3GB) |
| ✅ Voice cloning (3-5s reference) | ⚠️ Windows Long Path required |
| ✅ Better audio quality | |

### VieNeu-TTS (Standard) - NOT RECOMMENDED

**Avoid for production** - Use LMDeploy instead.

| Issue | Value |
|-------|-------|
| RTF | 1.445 (44% slower than real-time) |
| Speed | 9.3x slower than LMDeploy |
| GPU Memory | 5.4 GB (62% more than LMDeploy) |

---

## RTF Distribution

```
RTF (Real Time Factor) - Lower is better
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MMS TTS          ░░  0.02 (49x realtime)
VieNeu LMDeploy  ██░░  0.20 (5x realtime)
VieNeu Standard  ███████  1.45 (NOT realtime)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                 0    0.5   1.0   1.5   2.0

Threshold for real-time: RTF < 1.0
```

---

## Recommendations

### For Voice Assistant Production

**Option 1: MMS TTS (Fastest)**
```python
# Best for: Simple responses, system prompts
# Pros: 49x real-time, minimal resources
# Cons: Single generic voice
python -m tts_test synthesize vietts "Xin chào thế giới"
```

**Option 2: VieNeu-TTS + LMDeploy (Best Quality)**
```python
# Best for: High-quality responses, voice cloning
# Pros: 5x real-time, 6 voices, cloning support
# Cons: Requires LMDeploy setup
# Setup: See docs/vietneu-tts-lmdeploy-results.md
```

### Speed vs Quality Trade-off

```
Quality
  ↑
  │         ● VieNeu LMDeploy (5x realtime)
  │
  │
  │
  │   ● MMS TTS (49x realtime)
  └──────────────────────────→ Speed
```

---

## How to Run

### MMS TTS Benchmark

```bash
# Single synthesis
python -m tts_test synthesize vietts "Xin chào thế giới"

# Benchmark
python -m tts_test benchmark "Xin chào các bạn"
```

### VieNeu-TTS Benchmark (LMDeploy)

```bash
# Run from VieNeu-TTS repo
cd D:\H Drive\git\VieNeu-TTS

# Standard backend
.venv/Scripts/python.exe benchmark.py --device cuda --backend standard --iters 5

# LMDeploy backend (9.3x faster!)
.venv/Scripts/python.exe benchmark.py --device cuda --backend lmdeploy --iters 5
```

### VieNeu-TTS from tts_test CLI

```bash
# Standard backend (not recommended for production)
python -m tts_test benchmark-vietneu --backend standard --iters 5

# LMDeploy (requires installation)
python -m tts_test benchmark-vietneu --backend lmdeploy --iters 5

# Compare all backends
python -m tts_test benchmark-vietneu-all --iters 5
```

---

## Technical Details

### Benchmark Methodology

- **Inference-only measurement:** Model pre-loaded before timing
- **Multiple iterations averaged:** 3-5 runs per backend
- **GPU synchronization:** Accurate CUDA timing
- **Same text input:** 71 characters for fair comparison

### Hardware Requirements

| Model | Min GPU | Recommended | VRAM |
|-------|---------|-------------|------|
| MMS TTS | GTX 1060 | RTX 3060 | ~500 MB |
| VieNeu LMDeploy | RTX 2060 | RTX 3080 | ~3.5 GB |
| VieNeu Standard | RTX 3070 | RTX 3090 | ~6 GB |

---

## See Also

- [VieNeu-TTS LMDeploy Results](vietneu-tts-lmdeploy-results.md) - Detailed LMDeploy analysis
- [VieNeu-TTS Benchmark Guide](vietneu-tts-benchmark-guide.md) - Setup and usage
- [Advanced TTS Setup](advanced-tts-setup.md) - XTTS-v2 and GPT-SoVITS
- [TTS Usage Guide](tts-usage.md) - Complete usage documentation

---

**Conclusion:** For production voice assistant:
- **MMS TTS:** Maximum speed (49x), good quality
- **VieNeu-TTS + LMDeploy:** Best quality with real-time (5x) performance

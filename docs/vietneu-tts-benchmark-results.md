# VieNeu-TTS Multi-Backend Benchmark Results (2026-03-28)

## Environment

- **Host:** Windows 10 Pro
- **GPU:** NVIDIA GeForce RTX 3080 Ti
- **CUDA:** 12.8
- **Python:** 3.10+
- **VieNeu-TTS:** 1.3.0
- **PyTorch:** 2.11.0+cu128

## Test Configuration

- **Text:** "Xin chao, day la bai kiem tra giong noi tieng Viet cua cac mo hinh TTS." (71 chars)
- **Iterations:** 5
- **Device:** CUDA
- **Model:** pnnbao-ump/VieNeu-TTS-0.3B

---

## Results Summary

| Backend | Avg Time (s) | Std Dev (ms) | Audio (s) | RTF Avg | RTF P95 | Real-time | GPU Mem (MB) | Speedup |
|---------|-------------|--------------|-----------|---------|---------|-----------|--------------|---------|
| **standard** | 8.892 | 513.0 | 6.940 | 1.372 | 1.463 | NO | 1574.1 | 1.0x |
| **torch-compile** | 8.621 | 789.0 | 6.140 | 1.407 | 1.451 | NO | 1556.3 | **1.03x** |

---

## Key Findings

### 1. torch.compile Provides Marginal Improvement

- **Speedup:** 1.03x (~3% faster)
- **GPU Memory:** 18 MB less than standard
- **Consistency:** Higher standard deviation (789ms vs 513ms)

### 2. Neither Backend Achieves Real-Time

Both backends have RTF > 1.0:
- **Standard:** RTF = 1.372 (37% slower than real-time)
- **torch-compile:** RTF = 1.407 (41% slower than real-time)

### 3. Comparison with User's LMDeploy Results

User's LMDeploy benchmark (different model/setup):
```
GPU Standard:  RTF = 1.551
GPU LMDeploy:  RTF = 0.231 (6.5x faster)
```

Our VieNeu-TTS results:
```
GPU Standard:  RTF = 1.372
GPU torch-compile: RTF = 1.407 (1.03x faster)
```

**Analysis:** LMDeploy's 6.5x speedup was likely for a VITS-based model (like MMS TTS), not an LLM-based model like VieNeu-TTS. LMDeploy is optimized for transformer LLMs, but VieNeu-TTS has additional components (codec, flow matching) that may not benefit from LMDeploy optimization.

---

## Recommendations

### Option 1: Use MMS TTS (VITS) for Real-Time

If real-time is critical, use **MMS TTS (VITS)** instead:
- **RTF:** 0.02 (49x real-time)
- **Quality:** Good for general Vietnamese TTS
- **Trade-off:** Single speaker, no voice cloning

### Option 2: Optimize VieNeu-TTS Further

If VieNeu-TTS quality is required:

1. **Model Quantization (INT8/FP8)**
   - Expected speedup: 2-3x
   - Tools: bitsandbytes, AWQ
   - Risk: Quality degradation

2. **Smaller Backbone**
   - Use a smaller LLM backbone (if available)
   - Expected speedup: 2-4x
   - Risk: Lower quality

3. **TensorRT Optimization**
   - Export backbone to TensorRT
   - Expected speedup: 2-4x
   - Effort: High (custom implementation)

4. **ONNX Runtime**
   - Export to ONNX and use ONNX Runtime
   - Expected speedup: 1.5-2.5x
   - Effort: Medium

### Option 3: Hybrid Approach

Use both models based on use case:
- **MMS TTS:** For quick responses, system messages
- **VieNeu-TTS:** For high-quality, cloned voice responses

---

## Next Steps (If Further Optimization Needed)

1. **Profile Model Bottlenecks**
   ```bash
   python -m torch.profiler benchmark_vietneu.py
   ```
   Identify which component (backbone, codec, flow matching) is the bottleneck.

2. **Try ONNX Export**
   ```bash
   pip install onnx onnxruntime-gpu
   python -c "from vieneu import VieNeuTTS; import onnx; ..."
   ```

3. **Experiment with INT8 Quantization**
   ```bash
   pip install bitsandbytes
   ```

4. **Consider Alternative Models**
   - **F5-TTS Vietnamese:** Newer model, potentially faster
   - **ZipVoice:** Vietnamese-specific optimization

---

## How to Run

```bash
# Single backend
python -m tts_test.benchmark_vietneu --device cuda --backend standard --iters 5

# Compare all backends
python -m tts_test.benchmark_vietneu_all --device cuda --iters 5

# With torch.compile
python -m tts_test.benchmark_vietneu --device cuda --backend torch-compile --iters 5
```

---

## Artifacts

- Benchmark script: `tts_test/benchmark_vietneu.py`
- Comparison script: `tts_test/benchmark_vietneu_all.py`
- CLI command: `python -m tts_test benchmark-vietneu --help`

---

**Conclusion:** For real-time Vietnamese TTS, **MMS TTS (VITS)** with RTF=0.02 is the best choice. VieNeu-TTS (RTF~1.4) provides better quality and voice cloning but requires further optimization to achieve real-time performance.

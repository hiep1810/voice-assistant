# Advanced TTS Setup: XTTS-v2 and GPT-SoVITS

This guide explains how to set up the full versions of **Coqui XTTS-v2** and **GPT-SoVITS**, which require additional manual setup beyond the automatic environment creation.

---

## Current Status

Both models currently use **MMS TTS (Vietnamese)** as a fallback because their full implementations require additional setup:

| Model | Default Status | Full Setup Required |
|-------|---------------|---------------------|
| **XTTS-v2** | MMS TTS fallback | Coqui TTS with XTTS support |
| **GPT-SoVITS** | MMS TTS fallback | GPT-SoVITS inference code + pretrained weights |

---

## Coqui XTTS-v2 Full Setup

### Option 1: Use the HuggingFace Transformers version (Recommended)

XTTS-v2 is available on HuggingFace and can be loaded with custom code:

```bash
# Navigate to the project
cd D:\H Drive\git\voice-assistant

# Activate the xtts-v2 environment
.\envs\tts\xtts-v2\Scripts\activate

# Install required packages
pip install huggingface_hub transformers>=4.40.0

# Download the model
python -c "from huggingface_hub import snapshot_download; snapshot_download('coqui/XTTS-v2')"
```

**Note:** The Coqui TTS library (v0.14.3) does not include XTTS-v2 in its model registry. You'll need to load the model using custom code from HuggingFace.

### Option 2: Use a Community XTTS Implementation

Several community implementations are available:

1. **xtts-api-server**: A FastAPI server for XTTS-v2
   ```bash
   pip install xtts-api-server
   ```

2. **TTS (fork with XTTS support)**: Some forks include XTTS-v2 support
   ```bash
   pip install git+https://github.com/coqui-ai/TTS.git
   ```

### Using XTTS-v2 via API

If you prefer running XTTS-v2 as a separate service:

```bash
# Run an XTTS-v2 server (requires additional setup)
python -m xtts_api_server --model_name "coqui/XTTS-v2" --port 8020
```

Then modify `tts_test/scripts/run_xtts.py` to call the API endpoint.

---

## GPT-SoVITS Full Setup

GPT-SoVITS requires cloning the official repository and installing dependencies manually.

### Step 1: Clone the Repository

```bash
cd D:\H Drive\git
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
```

### Step 2: Create a Python 3.9 Environment

GPT-SoVITS works best with Python 3.9:

```bash
# Install Python 3.9 via pyenv-win
pyenv install 3.9.13
pyenv global 3.9.13

# Create virtual environment
~/.pyenv/pyenv-win/versions/3.9.13/python.exe -m venv D:\H Drive\git\voice-assistant\envs\tts\gpt-sovits-full
```

### Step 3: Install Dependencies

```bash
# Activate the environment
D:\H Drive\git\voice-assistant\envs\tts\gpt-sovits-full\Scripts\activate

# Install PyTorch (with CUDA)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install GPT-SoVITS dependencies
pip install -r requirements.txt
```

### Step 4: Download Pretrained Weights

```bash
# Download GPT-SoVITS weights
python -m gpt_sovits.download_models
```

### Step 5: Run Inference

```bash
# Run inference script
python GPT_SoVITS/inference_webui.py
```

Or use the API:

```python
import requests

response = requests.post(
    "http://localhost:9880/tts",
    json={
        "text": "Xin chào thế giới",
        "text_lang": "vi",
        "ref_audio_path": "path/to/reference.wav",
        "prompt_lang": "vi",
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

---

## Alternative: Use Docker

Both models can be run via Docker for easier setup:

### XTTS-v2 Docker

```bash
docker run -p 8020:8020 ghcr.io/daswer123/xtts-api-server:latest --model_name "coqui/XTTS-v2"
```

### GPT-SoVITS Docker

```bash
docker run -p 9880:9880 breakstring/gpt-sovits:latest
```

---

## Current Fallback Behavior

When the full models are not available, both scripts fall back to **Facebook MMS TTS for Vietnamese**:

- **Model:** `facebook/mms-tts-vie`
- **Framework:** Transformers (VITS)
- **Quality:** Good for general Vietnamese TTS
- **Speaker Support:** No voice cloning (uses default voice)

This fallback provides:
- Fast synthesis (RTF ~2.6)
- Good Vietnamese pronunciation
- No speaker customization

---

## Comparison: Fallback vs Full Models

| Feature | MMS Fallback | XTTS-v2 Full | GPT-SoVITS Full |
|---------|-------------|--------------|-----------------|
| Voice Cloning | ❌ No | ✅ Yes (3-5s ref) | ✅ Yes (few-shot) |
| Multilingual | ❌ Vietnamese only | ✅ 6 languages | ✅ Multiple |
| RTF | ~2.6 | ~1.0 | ~4.2 |
| Setup Complexity | ✅ Easy | ⚠️ Medium | ❌ Complex |
| Quality (MOS) | ~3.5 | ~4.2 | ~4.0 |

---

## Quick Reference

### For Most Users

Use the fallback models - they work out of the box:

```bash
python -m tts_test benchmark "Xin chào thế giới"
```

### For Advanced Users

1. Set up XTTS-v2 or GPT-SoVITS following this guide
2. Modify the respective script in `tts_test/scripts/` to use the full model
3. Run benchmarks to compare quality

---

## See Also

- [TTS Benchmark Results](tts-benchmark-results.md) - Latest benchmark results
- [TTS Usage Guide](tts-usage.md) - General TTS usage
- [HuggingFace XTTS-v2](https://huggingface.co/coqui/XTTS-v2) - Official model page
- [GPT-SoVITS GitHub](https://github.com/RVC-Boss/GPT-SoVITS) - Official repository

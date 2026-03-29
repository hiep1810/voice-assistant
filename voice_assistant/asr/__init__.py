"""Streaming ASR interface and implementations."""

import threading
import queue
from typing import Optional, Callable, Generator, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class ASRResult:
    """ASR transcription result."""
    text: str
    is_final: bool
    confidence: float = 1.0
    language: str = "vi"


class StreamingASRBase(ABC):
    """Base class for streaming ASR models."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the ASR model."""
        pass

    @abstractmethod
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[ASRResult]:
        """Process an audio chunk and return partial/final results."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset ASR state for new utterance."""
        pass


class ZipformerASR(StreamingASRBase):
    """
    Streaming ASR using Sherpa-onnx Zipformer.

    Optimized for Vietnamese transcription with real-time performance.
    """

    def __init__(
        self,
        model_name: str = "zipformer",
        language: str = "vi",
        sample_rate: int = 16000,
    ):
        self.model_name = model_name
        self.language = language
        self.sample_rate = sample_rate
        self._model = None
        self._stream = None
        self._tail_paddings = 0
        self._is_ready = False

    def load_model(self) -> None:
        """Load Zipformer model from HuggingFace."""
        try:
            import sherpa_onnx

            # Use pre-trained Vietnamese Zipformer model
            # Model: icefall-aishell2-zipformer-whisper
            # For Vietnamese: Use fine-tuned model if available
            model_config = self._get_model_config()

            self._model = sherpa_onnx.OnlineRecognizer(model_config)
            self._stream = self._model.create_stream()
            self._is_ready = True

        except ImportError:
            raise ImportError(
                "sherpa-onnx not installed. Run: pip install sherpa-onnx"
            )

    def _get_model_config(self):
        """Get model configuration for sherpa-onnx."""
        import sherpa_onnx

        # Use the Zipformer model configuration
        # For production, download models to local cache
        model_dir = self._download_model()

        return sherpa_onnx.OnlineRecognizerConfig(
            feat_config=sherpa_onnx.FeatureExtractorConfig(
                sampling_rate=self.sample_rate,
                feature_dim=80,
            ),
            model_config=sherpa_onnx.OnlineModelConfig(
                zipformer2_config=sherpa_onnx.OnlineZipformer2ModelConfig(
                    model=model_dir + "/encode_jit_trace.pt",
                    decoder=model_dir + "/decode_jit_trace.pt",
                    joiner=model_dir + "/joiner_jit_trace.pt",
                ),
                tokens=model_dir + "/tokens.txt",
                num_left_trailing=0,
                num_right_trailing=0,
            ),
            decoder_config=sherpa_onnx.ModifiedBeamSearchDecoderConfig(
                max_active_paths=4,
            ),
            enable_truncation=True,
        )

    def _download_model(self) -> str:
        """Download Zipformer model."""
        from huggingface_hub import snapshot_download

        # Use aishell2 zipformer as base (fine-tune for Vietnamese if needed)
        repo_id = "k2-fsa/sherpa-onnx-streaming-zipformer-aishell2-20230627"
        model_dir = snapshot_download(repo_id)
        return model_dir

    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[ASRResult]:
        """Process audio chunk and return result if available."""
        if not self._is_ready:
            return None

        # Convert to float32 if needed
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Accept samples from the stream
        self._stream.accept_waveform(audio_chunk, self.sample_rate)

        # Check if model has output
        while self._model.is_ready(self._stream):
            self._model.decode(self._stream, 1)

        # Get result
        result_text = self._model.get_result(self._stream).text

        if result_text:
            return ASRResult(
                text=result_text,
                is_final=False,  # Streaming results are partial
                language=self.language,
            )

        return None

    def finish(self) -> Optional[ASRResult]:
        """Finish the stream and get final result."""
        if not self._is_ready or self._stream is None:
            return None

        # Add tail padding
        tail_samples = int(self.sample_rate * 0.5)  # 500ms
        self._stream.input_waveform(
            np.zeros(tail_samples, dtype=np.float32)
        )

        # Process remaining
        while self._model.is_ready(self._stream):
            self._model.decode(self._stream, 1)

        # Get final result
        result_text = self._model.get_result(self._stream).text

        if result_text:
            return ASRResult(
                text=result_text,
                is_final=True,
                language=self.language,
            )

        return None

    def reset(self) -> None:
        """Reset the ASR stream."""
        if self._stream is not None:
            self._model.reset(self._stream)


class WhisperStreamingASR(StreamingASRBase):
    """
    Streaming ASR using Whisper with chunked processing.

    Note: Not truly streaming - processes complete chunks.
    Better for offline or near-real-time use cases.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        language: str = "vi",
        sample_rate: int = 16000,
    ):
        self.model_size = model_size
        self.language = language
        self.sample_rate = sample_rate
        self._model = None
        self._chunks = []
        self._buffer_seconds = 5.0  # Buffer 5 seconds before transcribing

    def load_model(self) -> None:
        """Load Whisper model."""
        try:
            import whisper

            self._model = whisper.load_model(self.model_size)

        except ImportError:
            raise ImportError("Whisper not installed. Run: pip install openai-whisper")

    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[ASRResult]:
        """Buffer audio and transcribe when enough is accumulated."""
        if self._model is None:
            return None

        self._chunks.append(audio_chunk)

        # Check if we have enough audio
        total_samples = sum(len(c) for c in self._chunks)
        total_seconds = total_samples / self.sample_rate

        if total_seconds >= self._buffer_seconds:
            # Transcribe buffered audio
            audio = np.concatenate(self._chunks)
            result = self._model.transcribe(
                audio,
                language=self.language,
                task="transcribe",
            )

            # Clear buffer
            self._chunks = []

            return ASRResult(
                text=result["text"],
                is_final=True,
                language=self.language,
            )

        return None

    def reset(self) -> None:
        """Clear audio buffer."""
        self._chunks = []


class StreamingASRHandler:
    """
    High-level handler for streaming ASR.

    Manages audio buffering, chunking, and result callbacks.
    """

    def __init__(
        self,
        asr: StreamingASRBase,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
    ):
        self.asr = asr
        self.on_partial = on_partial
        self.on_final = on_final
        self._is_running = False
        self._audio_queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the ASR handler thread."""
        self._is_running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the ASR handler."""
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def feed_audio(self, audio_chunk: np.ndarray) -> None:
        """Feed audio chunk to ASR handler."""
        self._audio_queue.put(audio_chunk)

    def _process_loop(self) -> None:
        """Main processing loop."""
        while self._is_running:
            try:
                audio_chunk = self._audio_queue.get(timeout=0.1)
                result = self.asr.process_chunk(audio_chunk)

                if result:
                    if result.is_final and self.on_final:
                        self.on_final(result.text)
                    elif not result.is_final and self.on_partial:
                        self.on_partial(result.text)

            except queue.Empty:
                pass

    def finish(self) -> None:
        """Finish streaming and get final result."""
        result = self.asr.finish()
        if result and self.on_final:
            self.on_final(result.text)


def create_streaming_asr(
    model: str = "zipformer",
    language: str = "vi",
    **kwargs
) -> StreamingASRBase:
    """Factory function to create streaming ASR instance."""
    if model == "zipformer":
        return ZipformerASR(language=language, **kwargs)
    elif model == "whisper":
        return WhisperStreamingASR(language=language, **kwargs)
    else:
        raise ValueError(f"Unknown ASR model: {model}")

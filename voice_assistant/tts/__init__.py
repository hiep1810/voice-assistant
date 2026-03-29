"""Streaming TTS interface for VieNeu-TTS with LMDeploy backend."""

import threading
import queue
import time
from typing import Optional, Callable, Generator, List, Dict
from dataclasses import dataclass
import numpy as np

from voice_assistant.config import TTSConfig


@dataclass
class TTSSegment:
    """TTS synthesis result."""
    audio: np.ndarray
    text: str
    sample_rate: int = 24000
    duration_s: float = 0.0


class VieNeuTTS:
    """
    VieNeu-TTS wrapper with LMDeploy backend for fast synthesis.

    Features:
    - LMDeploy TurboMind backend (9.3x faster than standard)
    - RTF ~0.20 (5x real-time)
    - 6 voice presets available
    - Voice cloning support (3-5s reference)

    Voice Presets:
    - neutrale: Neutral voice (default)
    - hanhphuc: Happy voice
    - leloi: Historical voice
    - nguyentruothanh: Poetic voice
    - chihanh: Gentle voice
    - khanhlinh: Clear voice
    """

    PRESET_VOICES = {
        "neutrale": "neutrale",
        "hanhphuc": "hanhphuc",
        "leloi": "leloi",
        "nguyentruothanh": "nguyentruothanh",
        "chihanh": "chihanh",
        "khanhlinh": "khanhlinh",
    }

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._model = None
        self._is_loaded = False
        self._sample_rate = 24000  # VieNeu-TTS native sample rate

    def load_model(self) -> None:
        """Load VieNeu-TTS model with LMDeploy backend."""
        if self.config.backend == "lmdeploy":
            self._load_lmdeploy()
        else:
            self._load_standard()

        self._is_loaded = True

    def _load_lmdeploy(self) -> None:
        """Load with LMDeploy backend (FastVieNeuTTS)."""
        try:
            from vieneu.fast import FastVieNeuTTS

            self._model = FastVieNeuTTS(
                backbone_repo="pnnbao-ump/VieNeu-TTS",
                backbone_device="cuda" if self.config.use_gpu else "cpu",
                codec_repo="neuphonic/neucodec",
                codec_device="cuda" if self.config.use_gpu else "cpu",
            )

        except ImportError:
            raise ImportError(
                "LMDeploy not installed. Run: pip install lmdeploy\n"
                "Also ensure Windows Long Path is enabled and reboot."
            )

    def _load_standard(self) -> None:
        """Load with standard PyTorch backend."""
        try:
            from vieneu import VieNeuTTS as StandardVieNeuTTS

            self._model = StandardVieNeuTTS(
                backbone_repo="pnnbao-ump/VieNeu-TTS",
                backbone_device="cuda" if self.config.use_gpu else "cpu",
                codec_repo="neuphonic/neucodec",
                codec_device="cuda" if self.config.use_gpu else "cpu",
            )

        except ImportError:
            raise ImportError("vieneu not installed. Run: pip install vieneu")

    def get_preset_voice(self, voice_name: Optional[str] = None) -> dict:
        """Get preset voice configuration."""
        voice = voice_name or self.config.speaker or "neutrale"

        if voice in self.PRESET_VOICES:
            voice_id = self.PRESET_VOICES[voice]
            return {"name": voice_id, "type": "preset"}

        raise ValueError(f"Unknown voice: {voice}. Available: {list(self.PRESET_VOICES.keys())}")

    def synthesize(
        self,
        text: str,
        voice: Optional[dict] = None,
        **kwargs
    ) -> TTSSegment:
        """
        Synthesize speech from text.

        Args:
            text: Vietnamese text to synthesize
            voice: Voice configuration (from get_preset_voice)
            **kwargs: Additional synthesis options

        Returns:
            TTSSegment with audio data
        """
        if not self._is_loaded:
            self.load_model()

        if voice is None:
            voice = self.get_preset_voice()

        # Synthesize
        audio = self._model.infer(text, voice=voice)

        return TTSSegment(
            audio=audio,
            text=text,
            sample_rate=self._sample_rate,
            duration_s=len(audio) / self._sample_rate,
        )

    def synthesize_streaming(
        self,
        text: str,
        voice: Optional[dict] = None,
        chunk_size: int = 50,  # Characters per chunk
    ) -> Generator[TTSSegment, None, None]:
        """
        Synthesize speech with sentence-level streaming.

        Splits text into sentences and yields audio chunks as they're synthesized.
        This enables double-buffered playback.

        Args:
            text: Vietnamese text to synthesize
            voice: Voice configuration
            chunk_size: Characters per chunk (for sentence splitting)

        Yields:
            TTSSegment for each sentence chunk
        """
        if not self._is_loaded:
            self.load_model()

        if voice is None:
            voice = self.get_preset_voice()

        # Split text into sentences (Vietnamese sentence boundaries)
        sentences = self._split_sentences(text, chunk_size)

        for sentence in sentences:
            if sentence.strip():
                audio = self._model.infer(sentence.strip(), voice=voice)
                yield TTSSegment(
                    audio=audio,
                    text=sentence.strip(),
                    sample_rate=self._sample_rate,
                    duration_s=len(audio) / self._sample_rate,
                )

    def _split_sentences(self, text: str, chunk_size: int = 50) -> List[str]:
        """Split Vietnamese text into sentences/chunks."""
        # Vietnamese sentence delimiters
        delimiters = ['.', '!', '?', '!', '.', '!', '?']

        sentences = []
        current = ""

        for char in text:
            current += char
            if char in delimiters and len(current) > chunk_size // 2:
                sentences.append(current.strip())
                current = ""

        # Add remaining text
        if current.strip():
            sentences.append(current.strip())

        # If sentences are too long, split by character count
        result = []
        for sentence in sentences:
            if len(sentence) > chunk_size * 2:
                # Split long sentences
                for i in range(0, len(sentence), chunk_size):
                    result.append(sentence[i:i + chunk_size])
            else:
                result.append(sentence)

        return result


class StreamingTTS:
    """
    High-level streaming TTS handler.

    Manages synthesis and audio output queue for double-buffered playback.
    """

    def __init__(
        self,
        tts: VieNeuTTS,
        on_segment: Optional[Callable[[TTSSegment], None]] = None,
    ):
        self.tts = tts
        self.on_segment = on_segment
        self._queue: queue.Queue = queue.Queue()
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the TTS handler thread."""
        self._is_running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the TTS handler."""
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def synthesize(self, text: str) -> None:
        """Queue text for synthesis."""
        self._queue.put(text)

    def _process_loop(self) -> None:
        """Main synthesis loop."""
        while self._is_running:
            try:
                text = self._queue.get(timeout=0.1)

                for segment in self.tts.synthesize_streaming(text):
                    if self.on_segment:
                        self.on_segment(segment)

            except queue.Empty:
                pass
            except Exception as e:
                # Log error but continue processing
                print(f"TTS error: {e}")

    def synthesize_blocking(self, text: str) -> List[TTSSegment]:
        """Synchronize synthesis and return all segments."""
        segments = []

        for segment in self.tts.synthesize_streaming(text):
            segments.append(segment)
            if self.on_segment:
                self.on_segment(segment)

        return segments


class DoubleBufferedTTSPlayer:
    """
    Double-buffered TTS player for seamless playback.

    While segment N is playing, segment N+1 is being synthesized.
    """

    def __init__(
        self,
        tts: VieNeuTTS,
        sample_rate: int = 24000,
    ):
        self.tts = tts
        self.sample_rate = sample_rate
        self._audio_output = None
        self._synthesis_queue: queue.Queue = queue.Queue()
        self._playback_queue: queue.Queue = queue.Queue()
        self._is_running = False
        self._synthesis_thread: Optional[threading.Thread] = None
        self._playback_thread: Optional[threading.Thread] = None
        self._current_text: str = ""

    def start(self) -> None:
        """Start the player."""
        from voice_assistant.audio import AudioOutput

        self._is_running = True
        self._audio_output = AudioOutput(sample_rate=self.sample_rate)
        self._audio_output.start()

        # Start synthesis thread
        self._synthesis_thread = threading.Thread(
            target=self._synthesis_loop,
            daemon=True
        )
        self._synthesis_thread.start()

        # Start playback thread
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            daemon=True
        )
        self._playback_thread.start()

    def stop(self) -> None:
        """Stop the player."""
        self._is_running = False

        if self._audio_output:
            self._audio_output.stop()

        if self._synthesis_thread:
            self._synthesis_thread.join(timeout=1.0)

        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)

    def speak(self, text: str) -> None:
        """Queue text for speech synthesis and playback."""
        self._current_text = text
        self._synthesis_queue.put(text)

    def _synthesis_loop(self) -> None:
        """Synthesis loop - runs in background."""
        while self._is_running:
            try:
                text = self._synthesis_queue.get(timeout=0.1)

                # Synthesize and queue for playback
                for segment in self.tts.synthesize_streaming(text):
                    self._playback_queue.put(segment.audio)

            except queue.Empty:
                pass
            except Exception as e:
                print(f"Synthesis error: {e}")

    def _playback_loop(self) -> None:
        """Playback loop - runs in background."""
        while self._is_running:
            try:
                audio_data = self._playback_queue.get(timeout=0.1)
                self._audio_output.queue_audio(audio_data)

            except queue.Empty:
                pass


# Factory function

def create_tts(
    model: str = "vietneu-tts",
    backend: str = "lmdeploy",
    speaker: Optional[str] = None,
    **kwargs
) -> VieNeuTTS:
    """Create TTS instance."""
    config = TTSConfig(
        model=model,
        backend=backend,
        speaker=speaker,
        **kwargs
    )
    return VieNeuTTS(config)

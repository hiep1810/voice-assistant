"""
Streaming VAD — Real-time Voice Activity Detection for voice assistants.

Provides a StreamingVAD class that can process audio chunks in real-time,
detecting when speech starts and stops. This is essential for:
- Voice trigger detection ("hey assistant")
- Auto-start/stop recording based on speech
- Streaming ASR integration

Based on Silero VAD's streaming API.
"""

from typing import List, Dict, Optional, Callable
import numpy as np


class StreamingVAD:
    """
    Real-time Voice Activity Detection for streaming audio.

    Usage:
        vad = StreamingVAD()
        for chunk in audio_chunks:
            if vad.is_speaking(chunk):
                # Speech detected!
                pass
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        onset: float = 0.5,
        offset: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        max_speech_duration_s: float = float('inf'),
    ):
        """
        Initialize Streaming VAD.

        Args:
            sampling_rate: Audio sample rate in Hz (8000 or 16000)
            onset: Threshold for speech start (0-1, higher = more strict)
            offset: Threshold for speech end (0-1, higher = more strict)
            min_speech_duration_ms: Ignore speech shorter than this
            min_silence_duration_ms: Wait this long before marking speech end
            max_speech_duration_s: Maximum speech duration before forced stop
        """
        self.sampling_rate = sampling_rate
        self.onset = onset
        self.offset = offset
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.max_speech_duration_s = max_speech_duration_s

        self._model = None
        self._state = None
        self._context = None
        self._is_speaking = False
        self._speech_start_time = None
        self._current_segment = None
        self._segments = []

        # Internal state
        self._silence_start = None
        self._speech_buffer = []

    def load_model(self, force_reload: bool = False):
        """Load Silero VAD model (lazy loading)."""
        if self._model is not None and not force_reload:
            return

        try:
            from silero_vad import (
                load_silero_vad,
                get_speech_timestamps,
                VADIterator,
            )
            self._model = load_silero_vad()
            self._vad_iterator = VADIterator(
                model=self._model,
                sampling_rate=self.sampling_rate,
                threshold=self.onset,
                min_speech_duration_ms=self.min_speech_duration_ms,
                max_speech_duration_s=self.max_speech_duration_s,
                min_silence_duration_ms=self.min_silence_duration_ms,
            )
        except ImportError:
            raise ImportError(
                "Silero VAD not installed. Run: pip install silero-vad"
            )

    def reset(self):
        """Reset VAD state for new audio stream."""
        self._is_speaking = False
        self._speech_start_time = None
        self._current_segment = None
        self._silence_start = None
        self._speech_buffer = []
        if hasattr(self, '_vad_iterator'):
            self._vad_iterator.reset_states()

    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict]:
        """
        Process a chunk of audio and return speech event if detected.

        Args:
            audio_chunk: Audio data (mono, at sampling_rate)

        Returns:
            Dict with speech event info, or None
            - {"event": "start", "time": timestamp}
            - {"event": "end", "time": timestamp, "segment": {...}}
        """
        self.load_model()

        # Convert to torch tensor if needed
        import torch
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk

        # Get speech probability
        speech_prob = self._model(audio_tensor, self.sampling_rate).item()

        current_time = self._get_current_time()

        # Simple threshold-based detection
        if speech_prob >= self.onset and not self._is_speaking:
            # Speech started
            self._is_speaking = True
            self._speech_start_time = current_time
            return {"event": "start", "time": current_time, "prob": speech_prob}

        elif speech_prob < self.offset and self._is_speaking:
            # Potential speech end
            if self._silence_start is None:
                self._silence_start = current_time
            else:
                silence_duration = (current_time - self._silence_start) * 1000
                if silence_duration >= self.min_silence_duration_ms:
                    # Confirmed speech end
                    self._is_speaking = False
                    segment = {
                        "start": self._speech_start_time,
                        "end": current_time,
                        "duration": current_time - self._speech_start_time,
                    }
                    self._segments.append(segment)
                    return {"event": "end", "time": current_time, "segment": segment}
        elif speech_prob >= self.offset and self._silence_start is not None:
            # Speech resumed before silence threshold
            self._silence_start = None

        return None

    def is_speaking(self, audio_chunk: np.ndarray) -> bool:
        """
        Check if speech is currently detected.

        Args:
            audio_chunk: Audio data to check

        Returns:
            True if speech is detected, False otherwise
        """
        self.process_chunk(audio_chunk)
        return self._is_speaking

    def get_current_segment(self) -> Optional[Dict]:
        """Get the current speech segment if speaking."""
        if self._is_speaking and self._speech_start_time is not None:
            return {
                "start": self._speech_start_time,
                "end": self._get_current_time(),
                "duration": self._get_current_time() - self._speech_start_time,
            }
        return None

    def get_segments(self) -> List[Dict]:
        """Get all completed speech segments."""
        return self._segments.copy()

    def _get_current_time(self) -> float:
        """Get current timestamp in seconds."""
        import time
        return time.time()


def create_vad_iterator(
    sampling_rate: int = 16000,
    threshold: float = 0.5,
    **kwargs
):
    """
    Create a Silero VAD iterator for advanced usage.

    Args:
        sampling_rate: Audio sample rate (8000 or 16000)
        threshold: Speech detection threshold (0-1)
        **kwargs: Additional arguments for VADIterator

    Returns:
        VADIterator instance
    """
    from silero_vad import load_silero_vad, VADIterator

    model = load_silero_vad()
    return VADIterator(
        model=model,
        sampling_rate=sampling_rate,
        threshold=threshold,
        **kwargs
    )

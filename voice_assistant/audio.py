"""Audio I/O with double-buffering for voice pipeline."""

import threading
import queue
import numpy as np
from typing import Optional, Callable, Generator
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Audio configuration."""
    sample_rate: int = 16000
    chunk_size: int = 512
    buffer_size: int = 2048
    channels: int = 1


class AudioInput:
    """Real-time audio input stream."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = device
        self._stream = None
        self._queue: queue.Queue = queue.Queue()
        self._is_recording = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start audio input stream."""
        import pyaudio

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback,
        )
        self._is_recording = True
        self._stream.start_stream()

    def stop(self) -> None:
        """Stop audio input stream."""
        self._is_recording = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if hasattr(self, '_pa'):
            self._pa.terminate()

    def _callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - called in separate thread."""
        if self._is_recording:
            # Convert to float32 and normalize
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            self._queue.put(audio_float)
        return (None, pyaudio.paContinue)

    def read_chunk(self) -> Optional[np.ndarray]:
        """Read next audio chunk."""
        try:
            return self._queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def iter_chunks(self) -> Generator[np.ndarray, None, None]:
        """Iterate over audio chunks."""
        while self._is_recording:
            chunk = self.read_chunk()
            if chunk is not None:
                yield chunk


class AudioOutput:
    """Double-buffered audio output for TTS playback."""

    def __init__(
        self,
        sample_rate: int = 22050,
        buffer_size: int = 2048,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.device = device
        self._stream = None
        self._queue: queue.Queue = queue.Queue()
        self._is_playing = False
        self._thread: Optional[threading.Thread] = None
        self._current_buffer = None
        self._next_buffer = None
        self._buffer_lock = threading.Lock()

    def start(self) -> None:
        """Start audio output stream."""
        import pyaudio

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.device,
            frames_per_buffer=self.buffer_size,
            stream_callback=self._callback,
        )
        self._is_playing = True
        self._stream.start_stream()

        # Start buffer filler thread
        self._thread = threading.Thread(target=self._fill_buffer, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop audio output stream."""
        self._is_playing = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if hasattr(self, '_pa'):
            self._pa.terminate()

    def _callback(self, in_data, out_data, frame_count, time_info, status):
        """PyAudio callback - called in separate thread."""
        import struct

        if self._current_buffer is not None and len(self._current_buffer) > 0:
            # Get next chunk from current buffer
            chunk = self._current_buffer[:frame_count]
            self._current_buffer = self._current_buffer[frame_count:]

            # Convert float32 to int16 for output
            audio_int16 = (chunk * 32767).astype(np.int16)
            out_data[:] = audio_int16.tobytes()
        else:
            # Silence if no data
            out_data[:] = b'\x00' * (frame_count * 2)

        return (None, pyaudio.paContinue)

    def _fill_buffer(self) -> None:
        """Fill buffers from queue (runs in background thread)."""
        while self._is_playing:
            try:
                audio_data = self._queue.get(timeout=0.1)
                with self._buffer_lock:
                    if self._current_buffer is None or len(self._current_buffer) == 0:
                        self._current_buffer = audio_data
                    else:
                        # Double-buffering: prepare next buffer while current plays
                        self._next_buffer = audio_data
                        # Switch buffers
                        if len(self._current_buffer) == 0:
                            self._current_buffer, self._next_buffer = self._next_buffer, None
            except queue.Empty:
                pass

    def queue_audio(self, audio_data: np.ndarray) -> None:
        """Queue audio data for playback."""
        self._queue.put(audio_data)

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing and not self._queue.empty()

    def play_file(self, file_path: str) -> None:
        """Play audio file."""
        import soundfile as sf
        audio_data, sample_rate = sf.read(file_path)

        # Resample if needed
        if sample_rate != self.sample_rate:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)

        self.queue_audio(audio_data)


class DoubleBufferedPlayer:
    """
    Double-buffered TTS player for seamless playback.

    While sentence N is playing, sentence N+1 is being synthesized.
    This eliminates gaps between sentences in TTS output.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self._output = AudioOutput(sample_rate=sample_rate)
        self._queue: queue.Queue = queue.Queue()
        self._is_active = False
        self._current_sentence = 0

    def start(self) -> None:
        """Start the player."""
        self._output.start()
        self._is_active = True

    def stop(self) -> None:
        """Stop the player."""
        self._is_active = False
        self._output.stop()

    def add_sentence(self, audio_data: np.ndarray, sentence_id: int) -> None:
        """Add synthesized sentence to playback queue."""
        self._queue.put((sentence_id, audio_data))
        self._play_next()

    def _play_next(self) -> None:
        """Play next sentence from queue."""
        try:
            sentence_id, audio_data = self._queue.get(timeout=0.01)
            self._output.queue_audio(audio_data)
            self._current_sentence = sentence_id
        except queue.Empty:
            pass

    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._is_active and self._output.is_playing()


def get_available_devices() -> list:
    """Get list of available audio devices."""
    import pyaudio

    pa = pyaudio.PyAudio()
    devices = []

    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            devices.append({
                "index": i,
                "name": info["name"],
                "input_channels": info["maxInputChannels"],
                "output_channels": info["maxOutputChannels"],
                "sample_rate": int(info["defaultSampleRate"]),
                "is_input": info["maxInputChannels"] > 0,
                "is_output": info["maxOutputChannels"] > 0,
            })
    finally:
        pa.terminate()

    return devices


def get_default_input_device() -> Optional[int]:
    """Get default input device index."""
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        info = pa.get_default_input_device_info()
        return info["index"]
    except Exception:
        return None
    finally:
        pa.terminate()


def get_default_output_device() -> Optional[int]:
    """Get default output device index."""
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        info = pa.get_default_output_device_info()
        return info["index"]
    except Exception:
        return None
    finally:
        pa.terminate()

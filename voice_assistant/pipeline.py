"""Main voice pipeline orchestrator with concurrent threads."""

import threading
import queue
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum

from voice_assistant.config import VoiceAssistantConfig, LLMConfig
from voice_assistant.state import (
    SharedState,
    PipelineState,
    EventType,
    Event,
    ConversationTurn,
    get_shared_state,
)
from voice_assistant.audio import AudioInput, DoubleBufferedPlayer
from voice_assistant.asr import StreamingASRHandler, create_streaming_asr, ZipformerASR
from voice_assistant.llm import LlamaCppLLM
from voice_assistant.tools import get_builtin_tools
from voice_assistant.tts import VieNeuTTS, DoubleBufferedTTSPlayer


@dataclass
class PipelineConfig:
    """Voice pipeline configuration."""
    enable_vad: bool = True
    enable_asr: bool = True
    enable_llm: bool = True
    enable_tts: bool = True

    # Threading
    num_worker_threads: int = 4

    # Audio
    audio_sample_rate: int = 16000
    audio_chunk_size: int = 512

    # VAD
    vad_onset: float = 0.5
    vad_offset: float = 0.5

    # ASR
    asr_model: str = "zipformer"

    # LLM
    llm_model: str = "qwen3-2b"

    # TTS
    tts_model: str = "vietneu-tts"
    tts_backend: str = "lmdeploy"  # lmdeploy (fast) or standard
    tts_speaker: Optional[str] = "neutrale"


class VoicePipeline:
    """
    Main voice assistant pipeline with concurrent VAD→STT→LLM→TTS threads.

    Architecture:
    ```
    [Microphone] → [VAD Thread] → [ASR Thread] → [LLM Thread] → [TTS Thread] → [Speaker]
                         ↓              ↓              ↓              ↓
                    [Shared State with Event Bus]
    ```

    Features:
    - Concurrent processing for low latency
    - Double-buffered TTS for seamless playback
    - Event-driven state management
    - Tool calling support
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        voice_config: Optional[VoiceAssistantConfig] = None,
    ):
        self.config = config or PipelineConfig()
        self.voice_config = voice_config or VoiceAssistantConfig()

        # Shared state
        self.state = get_shared_state()

        # Components (lazy initialization)
        self._vad = None
        self._asr_handler: Optional[StreamingASRHandler] = None
        self._llm: Optional[LlamaCppLLM] = None
        self._audio_in: Optional[AudioInput] = None
        self._audio_out: Optional[DoubleBufferedPlayer] = None

        # Threading
        self._is_running = False
        self._threads: List[threading.Thread] = []
        self._audio_queue: queue.Queue = queue.Queue()

        # Callbacks
        self._on_transcription: Optional[Callable[[str], None]] = None
        self._on_response: Optional[Callable[[str], None]] = None

        # Conversation context
        self._conversation_history: List[Dict[str, str]] = []

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        self._initialize_audio()
        self._initialize_vad()
        self._initialize_asr()
        self._initialize_llm()
        self._initialize_tts()

    def _initialize_audio(self) -> None:
        """Initialize audio input/output."""
        self._audio_in = AudioInput(
            sample_rate=self.config.audio_sample_rate,
            chunk_size=self.config.audio_chunk_size,
        )
        self._audio_out = DoubleBufferedPlayer(
            sample_rate=24000,  # VieNeu-TTS sample rate
        )

    def _initialize_vad(self) -> None:
        """Initialize Voice Activity Detection."""
        from vad_test.streaming import StreamingVAD

        self._vad = StreamingVAD(
            sampling_rate=self.config.audio_sample_rate,
            onset=self.config.vad_onset,
            offset=self.config.vad_offset,
        )
        self._vad.load_model()

    def _initialize_asr(self) -> None:
        """Initialize Automatic Speech Recognition."""
        asr = create_streaming_asr(
            model=self.config.asr_model,
            language="vi",  # Vietnamese
        )
        asr.load_model()

        self._asr_handler = StreamingASRHandler(
            asr=asr,
            on_partial=self._on_asr_partial,
            on_final=self._on_asr_final,
        )

    def _initialize_llm(self) -> None:
        """Initialize LLM."""
        self._llm = LlamaCppLLM(
            LLMConfig(
                model=self.config.llm_model,
            )
        )
        # Lazy load - call load_model() when needed

    def _initialize_tts(self) -> None:
        """Initialize Text-to-Speech with VieNeu-TTS and LMDeploy backend."""
        from voice_assistant.tts import VieNeuTTS, TTSConfig

        tts_config = TTSConfig(
            model=self.config.tts_model,
            backend=self.config.tts_backend,
            speaker=self.config.tts_speaker,
            use_gpu=True,  # Always use GPU for real-time performance
        )

        self._tts = VieNeuTTS(tts_config)
        # Pre-load model for faster first synthesis
        # self._tts.load_model()  # Lazy load on first use

        # Double-buffered player for seamless playback
        self._tts_player = DoubleBufferedTTSPlayer(
            tts=self._tts,
            sample_rate=24000,  # VieNeu-TTS native sample rate
        )

    def register_tool(self, name: str, description: str, parameters: Dict, handler: Callable) -> None:
        """Register a tool for LLM function calling."""
        if self._llm:
            self._llm.register_tool(name, description, parameters, handler)

    def set_transcription_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for partial/final transcriptions."""
        self._on_transcription = callback

    def set_response_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for LLM responses."""
        self._on_response = callback

    def start(self) -> None:
        """Start the voice pipeline."""
        if self._is_running:
            return

        self._is_running = True
        self.state.state = PipelineState.IDLE

        # Start audio input
        if self._audio_in:
            self._audio_in.start()

        # Start ASR handler
        if self._asr_handler:
            self._asr_handler.start()

        # Start worker threads
        self._start_worker_threads()

        # Start audio processing loop
        self._main_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
        self._main_thread.start()

        self.state.add_event(Event(
            type=EventType.ERROR,  # Reusing ERROR type for system events
            data={"message": "Pipeline started"}
        ))

    def stop(self) -> None:
        """Stop the voice pipeline."""
        self._is_running = False

        # Stop audio input
        if self._audio_in:
            self._audio_in.stop()

        # Stop ASR handler
        if self._asr_handler:
            self._asr_handler.stop()

        # Stop audio output
        if self._audio_out:
            self._audio_out.stop()

        # Wait for threads
        for thread in self._threads:
            thread.join(timeout=1.0)

        self.state.add_event(Event(
            type=EventType.ERROR,
            data={"message": "Pipeline stopped"}
        ))

    def _start_worker_threads(self) -> None:
        """Start worker threads for pipeline stages."""
        # VAD/ASR thread already running via StreamingASRHandler
        # LLM and TTS will run in separate threads when needed

    def _audio_processing_loop(self) -> None:
        """Main audio processing loop - reads from mic and feeds VAD/ASR."""
        while self._is_running:
            if self.state.is_muted:
                time.sleep(0.1)
                continue

            # Read audio chunk
            chunk = self._audio_in.read_chunk()
            if chunk is None:
                continue

            # Feed to VAD
            if self._vad and self.config.enable_vad:
                vad_event = self._vad.process_chunk(chunk)
                if vad_event:
                    self._handle_vad_event(vad_event)

            # Feed to ASR (only when VAD detects speech)
            if self._asr_handler and self.config.enable_asr:
                if self.state.state == PipelineState.LISTENING:
                    self._asr_handler.feed_audio(chunk)

    def _handle_vad_event(self, event: Dict) -> None:
        """Handle VAD event."""
        event_type = event.get("event")

        if event_type == "start":
            # Speech started
            self.state.state = PipelineState.LISTENING
            self.state.add_event(Event(
                type=EventType.VAD_SPEECH_START,
                data={"time": event.get("time"), "prob": event.get("prob")}
            ))

        elif event_type == "end":
            # Speech ended - ASR should have final result
            self.state.state = PipelineState.PROCESSING
            segment = event.get("segment", {})
            self.state.add_event(Event(
                type=EventType.VAD_SPEECH_END,
                data={"segment": segment}
            ))

    def _on_asr_partial(self, text: str) -> None:
        """Handle partial ASR result."""
        self.state.current_transcription = text
        self.state.add_event(Event(
            type=EventType.ASR_PARTIAL,
            data={"text": text}
        ))

        if self._on_transcription:
            self._on_transcription(text)

    def _on_asr_final(self, text: str) -> None:
        """Handle final ASR result."""
        self.state.current_transcription = text
        self.state.add_event(Event(
            type=EventType.ASR_FINAL,
            data={"text": text}
        ))

        # Add to conversation history
        self._conversation_history.append({"role": "user", "content": text})

        # Trigger LLM response
        self._trigger_llm_response(text)

    def _trigger_llm_response(self, user_text: str) -> None:
        """Trigger LLM response in separate thread."""
        llm_thread = threading.Thread(
            target=self._llm_response_thread,
            args=(user_text,),
            daemon=True,
        )
        llm_thread.start()
        self._threads.append(llm_thread)

    def _llm_response_thread(self, user_text: str) -> None:
        """Generate LLM response (runs in worker thread)."""
        if not self._llm:
            return

        # Load model if needed
        if self._llm._model is None:
            self.state.add_event(Event(
                type=EventType.ERROR,
                data={"message": "Loading LLM model..."}
            ))
            self._llm.load_model()

        # Generate response
        response_text = ""
        try:
            for token in self._llm.generate_streaming(
                user_text,
                self._conversation_history[-20:]  # Last 20 turns
            ):
                response_text += token

                # Stream tokens to TTS (sentence-level buffering)
                self._on_llm_token(token)

            # Response complete
            self.state.current_response = response_text
            self.state.add_event(Event(
                type=EventType.LLM_TOKEN,
                data={"text": response_text, "complete": True}
            ))

            # Add to conversation history
            self._conversation_history.append({"role": "assistant", "content": response_text})

            # Trigger TTS
            if self.config.enable_tts:
                self._trigger_tts(response_text)

        except Exception as e:
            self.state.add_event(Event(
                type=EventType.ERROR,
                data={"message": f"LLM error: {e}"}
            ))

    def _on_llm_token(self, token: str) -> None:
        """Handle LLM token (stream to TTS buffer)."""
        # Sentence-level buffering for TTS
        # This is where double-buffering happens
        pass

    def _trigger_tts(self, text: str) -> None:
        """Trigger TTS synthesis."""
        if not self.config.enable_tts:
            return

        tts_thread = threading.Thread(
            target=self._tts_synthesis_thread,
            args=(text,),
            daemon=True,
        )
        tts_thread.start()
        self._threads.append(tts_thread)

    def _tts_synthesis_thread(self, text: str) -> None:
        """Synthesize TTS audio (runs in worker thread)."""
        if not hasattr(self, '_tts') or self._tts is None:
            self.state.state = PipelineState.IDLE
            return

        self.state.state = PipelineState.SPEAKING

        try:
            # Load model on first use
            if not self._tts._is_loaded:
                self._tts.load_model()

            # Synthesize with streaming (sentence-level)
            voice = self._tts.get_preset_voice(self.config.tts_speaker)

            for segment in self._tts.synthesize_streaming(text, voice=voice):
                # Queue audio for playback
                self.state.add_event(Event(
                    type=EventType.TTS_SEGMENT,
                    data={"text": segment.text, "duration": segment.duration_s}
                ))

                # Play audio (double-buffered)
                if hasattr(self, '_tts_player') and self._tts_player:
                    self._tts_player._playback_queue.put(segment.audio)

        except Exception as e:
            self.state.add_event(Event(
                type=EventType.ERROR,
                data={"message": f"TTS error: {e}"}
            ))

        # When complete:
        self.state.state = PipelineState.IDLE
        self.state.add_event(Event(
            type=EventType.TTS_DONE,
            data={"text": text}
        ))

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self._conversation_history.copy()

    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history = []

    def mute(self) -> None:
        """Mute the pipeline."""
        self.state.is_muted = True

    def unmute(self) -> None:
        """Unmute the pipeline."""
        self.state.is_muted = False

    def toggle_mute(self) -> bool:
        """Toggle mute state."""
        self.state.is_muted = not self.state.is_muted
        return self.state.is_muted


# Global pipeline instance
_pipeline: Optional[VoicePipeline] = None


def get_pipeline() -> VoicePipeline:
    """Get or create global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = VoicePipeline()
    return _pipeline


def reset_pipeline() -> None:
    """Reset global pipeline instance."""
    global _pipeline
    if _pipeline:
        _pipeline.stop()
    _pipeline = None

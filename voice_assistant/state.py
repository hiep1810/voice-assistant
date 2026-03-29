"""Shared state management for voice pipeline threads."""

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from collections import deque
import time


class PipelineState(Enum):
    """Voice pipeline state."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    PAUSED = "paused"
    ERROR = "error"


class EventType(Enum):
    """Pipeline event types."""
    VAD_SPEECH_START = "vad_speech_start"
    VAD_SPEECH_END = "vad_speech_end"
    ASR_PARTIAL = "asr_partial"
    ASR_FINAL = "asr_final"
    LLM_TOKEN = "llm_token"
    LLM_TOOL_CALL = "llm_tool_call"
    TTS_SEGMENT = "tts_segment"
    TTS_DONE = "tts_done"
    ERROR = "error"


@dataclass
class Event:
    """Pipeline event."""
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationTurn:
    """Single conversation turn."""
    role: str  # "user" or "assistant"
    text: str
    timestamp: float = field(default_factory=time.time)
    audio_path: Optional[str] = None


class SharedState:
    """
    Thread-safe shared state for voice pipeline.

    All access is protected by a reentrant lock to prevent deadlocks.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._state = PipelineState.IDLE
        self._events: deque[Event] = deque(maxlen=1000)
        self._conversation: deque[ConversationTurn] = deque(maxlen=100)
        self._current_transcription: str = ""
        self._current_response: str = ""
        self._is_muted: bool = False
        self._last_activity: float = time.time()

        # Event listeners
        self._event_listeners: List[callable] = []

        # Metrics
        self._metrics: Dict[str, Any] = {
            "vad_detections": 0,
            "asr_transcriptions": 0,
            "llm_responses": 0,
            "tts_syntheses": 0,
            "total_speech_time": 0.0,
            "total_idle_time": 0.0,
        }

    @property
    def state(self) -> PipelineState:
        with self._lock:
            return self._state

    @state.setter
    def state(self, value: PipelineState) -> None:
        with self._lock:
            old_state = self._state
            self._state = value
            self._last_activity = time.time()

            # Emit state change event
            self._emit_event(Event(
                type=EventType.ERROR if value == PipelineState.ERROR else EventType.VAD_SPEECH_START,
                data={"old_state": old_state.value, "new_state": value.value}
            ))

    @property
    def is_muted(self) -> bool:
        with self._lock:
            return self._is_muted

    @is_muted.setter
    def is_muted(self, value: bool) -> None:
        with self._lock:
            self._is_muted = value

    @property
    def current_transcription(self) -> str:
        with self._lock:
            return self._current_transcription

    @current_transcription.setter
    def current_transcription(self, value: str) -> None:
        with self._lock:
            self._current_transcription = value

    @property
    def current_response(self) -> str:
        with self._lock:
            return self._current_response

    @current_response.setter
    def current_response(self, value: str) -> None:
        with self._lock:
            self._current_response = value

    def add_event(self, event: Event) -> None:
        """Add event to event queue."""
        with self._lock:
            self._events.append(event)
            self._emit_event(event)

    def get_events(self, clear: bool = False) -> List[Event]:
        """Get recent events."""
        with self._lock:
            events = list(self._events)
            if clear:
                self._events.clear()
            return events

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add conversation turn."""
        with self._lock:
            self._conversation.append(turn)
            self._last_activity = time.time()

    def get_conversation(self, max_turns: Optional[int] = None) -> List[ConversationTurn]:
        """Get conversation history."""
        with self._lock:
            turns = list(self._conversation)
            if max_turns is not None:
                turns = turns[-max_turns:]
            return turns

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        with self._lock:
            self._conversation.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        with self._lock:
            return self._metrics.copy()

    def update_metrics(self, updates: Dict[str, Any]) -> None:
        """Update metrics."""
        with self._lock:
            for key, value in updates.items():
                if key in self._metrics:
                    if isinstance(value, (int, float)):
                        self._metrics[key] += value
                    else:
                        self._metrics[key] = value

    def register_listener(self, callback: callable) -> None:
        """Register event listener."""
        with self._lock:
            self._event_listeners.append(callback)

    def unregister_listener(self, callback: callable) -> None:
        """Unregister event listener."""
        with self._lock:
            if callback in self._event_listeners:
                self._event_listeners.remove(callback)

    def _emit_event(self, event: Event) -> None:
        """Emit event to all listeners (must be called with lock held)."""
        for listener in self._event_listeners:
            try:
                listener(event)
            except Exception:
                pass  # Don't let listener errors break the pipeline

    def get_idle_time(self) -> float:
        """Get time since last activity."""
        with self._lock:
            return time.time() - self._last_activity


# Global shared state instance
_shared_state: Optional[SharedState] = None


def get_shared_state() -> SharedState:
    """Get or create global shared state instance."""
    global _shared_state
    if _shared_state is None:
        _shared_state = SharedState()
    return _shared_state


def reset_shared_state() -> None:
    """Reset global shared state (for testing)."""
    global _shared_state
    _shared_state = None

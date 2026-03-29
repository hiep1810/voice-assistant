"""TUI (Terminal UI) for voice assistant using Rich Live."""

from typing import Optional, Callable
import threading
import time
import sys

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.spinner import Spinner
from rich.progress import Progress

from voice_assistant.pipeline import VoicePipeline, PipelineConfig
from voice_assistant.state import PipelineState, EventType, get_shared_state


# Status emoji mapping
STATUS_EMOJI = {
    PipelineState.IDLE: "⏸️",
    PipelineState.LISTENING: "🎤",
    PipelineState.PROCESSING: "💭",
    PipelineState.SPEAKING: "🔊",
    PipelineState.PAUSED: "⏸️",
    PipelineState.ERROR: "❌",
}


class VoiceAssistantTUI:
    """
    Rich-based TUI for voice assistant with live updates.

    Features:
    - Real-time transcription display
    - Conversation history with scrollback
    - Status indicators (mic, speaker, VAD)
    - Hotkeys: M (mic), V (camera), S (screen), Q (quit)
    """

    def __init__(
        self,
        pipeline: Optional[VoicePipeline] = None,
        model: Optional[str] = None,
        enable_tts: bool = True,
        enable_vad: bool = True,
        server_url: Optional[str] = None,
    ):
        self.console = Console()
        self.pipeline = pipeline or VoicePipeline()
        self.server_url = server_url

        # State
        self._transcription = ""
        _response = ""
        self._status = PipelineState.IDLE
        self._is_muted = False
        self._conversation_history = []
        self._last_event = None
        self._running = False

        # Layout
        self._layout = self._create_layout()

        # Event listener
        self._state = get_shared_state()
        self._state.register_listener(self._on_event)

    def _create_layout(self) -> Layout:
        """Create the TUI layout."""
        layout = Layout()

        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2),
        )

        layout["left"].split(
            Layout(name="transcription", size=10),
            Layout(name="status"),
        )

        return layout

    def _on_event(self, event) -> None:
        """Handle pipeline events."""
        self._last_event = event

        if event.type == EventType.ASR_PARTIAL:
            self._transcription = event.data.get("text", "")
        elif event.type == EventType.ASR_FINAL:
            self._transcription = event.data.get("text", "")
        elif event.type == EventType.LLM_TOKEN:
            if event.data.get("complete"):
                _response = event.data.get("text", "")
        elif event.type == EventType.VAD_SPEECH_START:
            self._status = PipelineState.LISTENING
        elif event.type == EventType.VAD_SPEECH_END:
            self._status = PipelineState.PROCESSING
        elif event.type == EventType.TTS_SEGMENT:
            self._status = PipelineState.SPEAKING
        elif event.type == EventType.TTS_DONE:
            self._status = PipelineState.IDLE

    def render_header(self) -> Panel:
        """Render header panel."""
        emoji = STATUS_EMOJI.get(self._status, "❓")
        status_text = self._status.value if self._status else "unknown"

        title = f"[bold]Voice Assistant[/]"
        subtitle = f"Model: qwen3-2b | {emoji} {status_text}"

        return Panel(
            Text(title, style="bold blue"),
            subtitle=subtitle,
            style="blue",
        )

    def render_footer(self) -> Panel:
        """Render footer with hotkeys and status."""
        hotkeys = "[M] Toggle Mic  [V] Camera  [S] Screen  [Q] Quit"
        mute_status = "[red]🔇 MUTED[/]" if self._is_muted else "[green]🎤 LIVE[/]"

        # Pipeline state
        state_info = f"State: {emoji} {status_text}" if hasattr(self, 'status_text') else ""

        return Panel(
            f"{mute_status} | {hotkeys}",
            style="dim",
        )

    def render_transcription(self) -> Panel:
        """Render live transcription panel."""
        if self._transcription:
            text = Text(self._transcription, style="bold yellow")
            return Panel(
                text,
                title="[bold]🎤 Transcription[/]",
                border_style="yellow",
            )
        else:
            status = "Listening..." if not self._is_muted else "Muted - Press M to unmute"
            return Panel(
                f"[dim]{status}[/]",
                title="[bold]🎤 Transcription[/]",
                border_style="dim",
            )

    def render_status(self) -> Panel:
        """Render status panel."""
        lines = []

        # VAD status
        if hasattr(self.pipeline, '_vad') and self.pipeline._vad:
            vad_status = "✅ Active" if self.pipeline.config.enable_vad else "❌ Disabled"
            lines.append(f"VAD: {vad_status}")

        # ASR status
        asr_status = "✅ Ready" if self.pipeline._asr_handler else "❌ Not loaded"
        lines.append(f"ASR: {asr_status}")

        # LLM status
        llm_status = "✅ Loaded" if self.pipeline._llm and self.pipeline._llm._model else "❌ Not loaded"
        lines.append(f"LLM: {llm_status}")

        # Metrics
        metrics = self._state.get_metrics()
        lines.append("")
        lines.append(f"Transcriptions: {metrics.get('asr_transcriptions', 0)}")
        lines.append(f"Responses: {metrics.get('llm_responses', 0)}")

        return Panel(
            "\n".join(lines),
            title="[bold]Status[/]",
            border_style="green",
        )

    def render_history(self) -> Panel:
        """Render conversation history panel."""
        history = self.pipeline.get_conversation_history()

        if not history:
            return Panel(
                "[dim]No conversation yet. Start speaking![/]",
                title="[bold]💬 Conversation History[/]",
                border_style="dim",
            )

        # Show last 10 turns
        recent_history = history[-10:]

        lines = []
        for turn in recent_history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")

            if role == "user":
                lines.append(f"[bold green]You:[/green] {content[:100]}")
            else:
                lines.append(f"[bold blue]Assistant:[/blue] {content[:100]}")
            lines.append("")

        return Panel(
            "\n".join(lines),
            title="[bold]💬 Conversation History[/]",
            border_style="blue",
        )

    def update_layout(self) -> None:
        """Update layout with current state."""
        self._layout["header"].update(self.render_header())
        self._layout["transcription"].update(self.render_transcription())
        self._layout["status"].update(self.render_status())
        self._layout["right"].update(self.render_history())
        self._layout["footer"].update(self.render_footer())

    def run(self) -> None:
        """Run the TUI main loop."""
        self._running = True

        # Start pipeline
        self.pipeline.initialize()
        self.pipeline.start()

        # Set up callbacks
        self.pipeline.set_transcription_callback(self._on_transcription)
        self.pipeline.set_response_callback(self._on_response)

        # Handle keyboard input in separate thread
        input_thread = threading.Thread(target=self._handle_input, daemon=True)
        input_thread.start()

        # Main display loop with Rich Live
        with Live(self._layout, console=self.console, refresh_per_second=10, screen=True) as live:
            while self._running:
                self.update_layout()
                live.update(self._layout)
                time.sleep(0.1)

        # Cleanup
        self.pipeline.stop()

    def _handle_input(self) -> None:
        """Handle keyboard input (runs in separate thread)."""
        while self._running:
            try:
                # Non-blocking key read
                if sys.platform == 'win32':
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                        self._handle_key(key)
                else:
                    # Unix-like systems - simplified implementation
                    # Full implementation needs proper terminal handling with termios/tty
                    pass

            except Exception:
                pass

            time.sleep(0.1)

    def _handle_key(self, key: str) -> None:
        """Handle key press."""
        if key == 'q':
            self._running = False
        elif key == 'm':
            self._is_muted = self.pipeline.toggle_mute()
        elif key == 'v':
            # Camera - placeholder
            pass
        elif key == 's':
            # Screen - placeholder
            pass

    def _on_transcription(self, text: str) -> None:
        """Handle transcription update."""
        self._transcription = text

    def _on_response(self, text: str) -> None:
        """Handle response update."""
        _response = text


def run_tui(
    model: Optional[str] = None,
    enable_tts: bool = True,
    enable_vad: bool = True,
    config_path: Optional[str] = None,
    server_url: Optional[str] = None,
) -> None:
    """
    Run voice assistant with TUI interface.

    This is the main entry point for the TUI.
    """
    console = Console()
    console.print("[bold blue]Voice Assistant TUI[/]")
    console.print("")

    if server_url:
        console.print(f"[bold]Using remote server:[/] {server_url}")

    # Create pipeline
    config = PipelineConfig(
        enable_vad=enable_vad,
        enable_tts=enable_tts,
        llm_model=model or "qwen3-2b",
        llm_server_url=server_url,
    )

    pipeline = VoicePipeline(config)

    # Create and run TUI
    tui = VoiceAssistantTUI(
        pipeline=pipeline,
        model=model,
        enable_tts=enable_tts,
        enable_vad=enable_vad,
        server_url=server_url,
    )

    try:
        tui.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/]")

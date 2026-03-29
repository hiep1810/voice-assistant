"""TUI (Terminal UI) for voice assistant using Rich."""

from typing import Optional
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

console = Console()


def run_tui(
    model: Optional[str] = None,
    enable_tts: bool = True,
    enable_vad: bool = True,
    config_path: Optional[str] = None,
) -> None:
    """
    Run voice assistant with TUI interface.

    Features:
    - Live transcription display
    - Conversation history
    - Status indicators (mic, speaker)
    - Hotkeys: V (camera), S (screen), M (mic), Q (quit)
    """
    console.print("[yellow]TUI Interface[/]")
    console.print("")
    console.print("The TUI interface is under development.")
    console.print("")
    console.print("For now, use CLI mode:")
    console.print("  [dim]voice-assistant cli[/]")
    console.print("")

    # Placeholder - full TUI implementation coming
    # This will use Rich Live for real-time updates

    from voice_assistant.cli import run_cli
    run_cli(
        model=model,
        enable_tts=enable_tts,
        config_path=config_path,
    )


class VoiceAssistantTUI:
    """Rich-based TUI for voice assistant."""

    def __init__(self):
        self._console = Console()
        self._transcription = ""
        self._response = ""
        self._status = "idle"
        self._is_muted = False
        self._conversation_history = []

    def create_layout(self) -> Layout:
        """Create the TUI layout."""
        layout = Layout()

        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(
            Layout(name="transcription"),
            Layout(name="history", ratio=2),
        )

        return layout

    def update_transcription(self, text: str) -> None:
        """Update live transcription display."""
        self._transcription = text

    def update_response(self, text: str) -> None:
        """Update response display."""
        self._response = text

    def update_status(self, status: str) -> None:
        """Update status (idle, listening, speaking, etc.)."""
        self._status = status

    def render_header(self) -> Panel:
        """Render header panel."""
        status_emoji = {
            "idle": "⏸️",
            "listening": "🎤",
            "processing": "💭",
            "speaking": "🔊",
            "paused": "⏸️",
        }.get(self._status, "❓")

        return Panel(
            f"[bold]Voice Assistant[/] | Status: {status_emoji} {self._status}",
            style="bold blue",
        )

    def render_footer(self) -> Panel:
        """Render footer with hotkeys."""
        hotkeys = "[M] Mic | [V] Camera | [S] Screen | [Q] Quit"
        mute_status = "[red]MUTED[/]" if self._is_muted else "[green]LIVE[/]"
        return Panel(f"{mute_status} | {hotkeys}", style="dim")

    def render_transcription(self) -> Panel:
        """Render transcription panel."""
        if self._transcription:
            return Panel(
                Text(self._transcription, style="bold yellow"),
                title="[bold]Transcription[/]",
                border_style="yellow",
            )
        else:
            return Panel(
                "[dim]Listening...[/]",
                title="[bold]Transcription[/]",
                border_style="dim",
            )

    def render_history(self) -> Panel:
        """Render conversation history panel."""
        if not self._conversation_history:
            return Panel(
                "[dim]No conversation yet[/]",
                title="[bold]History[/]",
                border_style="dim",
            )

        history_text = ""
        for turn in self._conversation_history[-5:]:  # Last 5 turns
            role = "[bold green]You[/]" if turn["role"] == "user" else "[bold blue]Assistant[/]"
            history_text += f"{role}: {turn['content']}\n\n"

        return Panel(
            history_text,
            title="[bold]Conversation History[/]",
            border_style="blue",
        )

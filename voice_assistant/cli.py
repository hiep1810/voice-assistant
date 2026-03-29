"""Voice Assistant CLI entry point."""

import typer
from typing import Optional
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="voice-assistant",
    help="Real-time voice assistant with VAD, STT, LLM, and TTS.",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="LLM model to use (default: qwen3-2b)",
    ),
    no_tts: bool = typer.Option(
        False, "--no-tts",
        help="Disable TTS output (text-only responses)",
    ),
    no_vad: bool = typer.Option(
        False, "--no-vad",
        help="Disable VAD (continuous listening mode)",
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to configuration file",
    ),
    server_url: Optional[str] = typer.Option(
        None, "--server-url", "-s",
        help="Remote llama.cpp server URL (e.g., http://localhost:8000)",
    ),
) -> None:
    """
    Launch the voice assistant with TUI interface.

    This starts the full pipeline:
    - VAD for voice activity detection
    - Streaming ASR for transcription
    - LLM for responses
    - TTS for voice output
    """
    console.print("[yellow]Voice Assistant (TUI mode)[/]")
    console.print("")
    console.print("Starting pipeline...")

    # Try to import and launch TUI
    try:
        from voice_assistant.tui.app import run_tui
        run_tui(
            model=model,
            enable_tts=not no_tts,
            enable_vad=not no_vad,
            config_path=config,
            server_url=server_url,
        )
    except ImportError as e:
        console.print(f"[red]TUI not available: {e}[/]")
        console.print("[yellow]Falling back to CLI mode...[/]")
        run_cli(
            model=model,
            enable_tts=not no_tts,
            config_path=config,
            server_url=server_url,
        )


@app.command("cli")
def run_cli(
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="LLM model to use",
    ),
    enable_tts: bool = typer.Option(
        True, "--tts/--no-tts",
        help="Enable TTS output",
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to configuration file",
    ),
    server_url: Optional[str] = typer.Option(
        None, "--server-url", "-s",
        help="Remote llama.cpp server URL (e.g., http://localhost:8000)",
    ),
) -> None:
    """
    Run voice assistant in CLI mode (no TUI).

    Simple text-based interaction without the terminal UI.
    """
    console.print("[yellow]Voice Assistant (CLI mode)[/]")
    console.print("")

    from pathlib import Path
    from voice_assistant.config import VoiceAssistantConfig
    from voice_assistant.llm import LlamaCppLLM
    from voice_assistant.tools import get_builtin_tools

    # Load configuration
    if config_path:
        config = VoiceAssistantConfig.load(Path(config_path))
    else:
        config = VoiceAssistantConfig()

    if model:
        config.llm.model = model

    if server_url:
        config.llm.server_url = server_url
        console.print(f"[bold]Using remote server:[/] {server_url}")
    else:
        console.print(f"[bold]Loading LLM:[/] {config.llm.model} (local)")

    # Initialize LLM
    llm = LlamaCppLLM(config.llm)
    llm.load_model()
    console.print("[green]LLM ready[/]")

    # Initialize tools
    tools = get_builtin_tools()
    for tool_def in tools.get_all_definitions():
        llm.register_tool(
            name=tool_def["function"]["name"],
            description=tool_def["function"]["description"],
            parameters=tool_def["function"]["parameters"],
            handler=tools._handlers[tool_def["function"]["name"]],
        )
    console.print("[green]Tools registered[/]")

    # Chat loop
    console.print("")
    console.print("[bold]Ready! Type your message (or 'quit' to exit)[/]")
    console.print("")

    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("")
            console.print("[yellow]Goodbye![/]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            console.print("[yellow]Goodbye![/]")
            break

        # Generate response
        console.print("[dim]Thinking...[/]", end="")

        response = ""
        try:
            for token in llm.generate_streaming(user_input, conversation_history):
                if not response:
                    console.print("\r[bold blue]Assistant:[/] ", end="")
                console.print(token, end="")
                response += token

            console.print("")  # Newline after response

        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
            continue

        # Update conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})

        # Limit history
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]


@app.command("list-models")
def list_models() -> None:
    """List available LLM models."""
    from voice_assistant.config import LLM_MODELS, VLM_MODELS

    console.print("[bold]Available LLM Models[/]")
    console.print("")

    table = Table(title="LLM Models")
    table.add_column("Model", style="cyan")
    table.add_column("HuggingFace ID")
    table.add_column("Context")
    table.add_column("Tool Calling")
    table.add_column("Vietnamese")

    for name, info in LLM_MODELS.items():
        table.add_row(
            name,
            info["hf_id"],
            str(info.get("context", "N/A")),
            "Yes" if info.get("tool_calling") else "No",
            "Yes" if info.get("vietnamese") else "No",
        )

    console.print(table)

    console.print("")
    console.print("[bold]Available VLM Models[/]")
    console.print("")

    table = Table(title="VLM Models")
    table.add_column("Model", style="cyan")
    table.add_column("HuggingFace ID")
    table.add_column("Context")

    for name, info in VLM_MODELS.items():
        table.add_row(
            name,
            info["hf_id"],
            str(info.get("context", "N/A")),
        )

    console.print(table)


@app.command("benchmark")
def benchmark(
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="LLM model to benchmark",
    ),
) -> None:
    """
    Benchmark pipeline latency.

    Measures:
    - VAD detection latency
    - ASR transcription time
    - LLM token generation rate
    - TTS synthesis time
    """
    console.print("[yellow]Pipeline Benchmark[/]")
    console.print("")
    console.print("This feature is under development.")


if __name__ == "__main__":
    app()

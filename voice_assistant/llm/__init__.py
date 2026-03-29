"""LLM integration via llama.cpp for GGUF models."""

import threading
from typing import Optional, Dict, Any, List, Generator, Callable
from dataclasses import dataclass
import json

from voice_assistant.config import LLMConfig, LLM_MODELS


@dataclass
class ToolDefinition:
    """Tool definition for LLM function calling."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Optional[Callable] = None


class LlamaCppLLM:
    """
    LLM wrapper using llama-cpp-python for GGUF model inference.

    Features:
    - Load GGUF models from HuggingFace or local path
    - KV cache continuation for fast multi-turn
    - Tool call parsing (Qwen3, LFM2 formats)
    - Streaming token generation
    - Optional: Use remote llama.cpp server via HTTP API
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._model = None
        self._chat_format = "chatml"
        self._tools: Dict[str, ToolDefinition] = {}
        self._lock = threading.Lock()

    @property
    def is_remote(self) -> bool:
        """Check if using remote server mode."""
        return self.config.server_url is not None

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load GGUF model or connect to remote server.

        Args:
            model_path: Path to GGUF file. If None, downloads from HuggingFace.
                        Ignored if using remote server mode.
        """
        if self.is_remote:
            self._connect_to_server()
            return

        from llama_cpp import Llama

        if model_path is None:
            model_path = self._download_model()

        self._model = Llama(
            model_path=model_path,
            n_ctx=self.config.context_length,
            n_gpu_layers=self.config.n_gpu_layers,
            n_threads=self.config.n_threads,
            flash_attention=self.config.flash_attention,
            verbose=False,
        )

        # Detect model type and set chat format
        self._detect_model_type()

    def _connect_to_server(self) -> None:
        """Connect to remote llama.cpp server."""
        import requests

        # Test connection
        try:
            response = requests.get(
                f"{self.config.server_url}/health",
                timeout=5,
            )
            if response.status_code == 200:
                print(f"Connected to llama.cpp server at {self.config.server_url}")
            else:
                print(f"Warning: Server returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Could not connect to llama.cpp server at {self.config.server_url}\n"
                f"Error: {e}\n\n"
                f"Please ensure:\n"
                f"1. The llama.cpp server is running\n"
                f"2. The URL is correct (e.g., http://localhost:8000)\n"
                f"3. No firewall is blocking the connection\n\n"
                f"To start a local llama.cpp server:\n"
                f"  llama-server -m your-model.gguf --host 0.0.0.0 --port 8000\n"
                f"\n"
                f"Or install llama-cpp-python for local inference:\n"
                f"  uv pip install llama-cpp-python"
            )

    def _download_model(self) -> str:
        """Download model from HuggingFace."""
        from huggingface_hub import hf_hub_download

        model_info = LLM_MODELS.get(self.config.model)
        if not model_info:
            raise ValueError(f"Unknown model: {self.config.model}")

        model_path = hf_hub_download(
            repo_id=model_info["hf_id"],
            filename=model_info["file"],
        )
        return model_path

    def _detect_model_type(self) -> None:
        """Detect model type and configure chat format."""
        model_lower = self.config.model.lower()

        if "qwen3" in model_lower:
            self._chat_format = "chatml-function-calling"
        elif "lfm2" in model_lower or "liquid" in model_lower:
            self._chat_format = "chatml"
        else:
            self._chat_format = "chatml"

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
    ) -> None:
        """Register a tool for function calling."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]

    def get_tool_definitions(self) -> List[Dict]:
        """Get tool definitions in LLM-native format."""
        tools = []
        for tool in self._tools.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            })
        return tools

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
        tools: Optional[List[Dict]] = None,
    ) -> Generator[str, None, None]:
        """
        Generate response with streaming tokens.

        Args:
            messages: Chat history in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: If True, yield tokens as they're generated
            tools: Tool definitions for function calling

        Yields:
            Generated tokens (may be partial for streaming)
        """
        if self.is_remote:
            yield from self._generate_remote(messages, max_tokens, temperature, stream, tools)
            return

        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        with self._lock:
            # Prepare tool definitions if provided
            if tools is None and self._tools:
                tools = self.get_tool_definitions()

            # Create chat completion
            kwargs = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
            }

            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            # Generate
            response = self._model.create_chat_completion(**kwargs)

            if stream:
                for chunk in response:
                    delta = chunk["choices"][0]["delta"]

                    # Check for tool calls
                    if "tool_calls" in delta:
                        yield json.dumps({
                            "type": "tool_call",
                            "tool_calls": delta["tool_calls"],
                        })
                    elif "content" in delta:
                        yield delta["content"]
            else:
                content = response["choices"][0]["message"]["content"]
                yield content

    def _generate_remote(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
        tools: Optional[List[Dict]] = None,
    ) -> Generator[str, None, None]:
        """
        Generate response using remote llama.cpp server via HTTP API.

        Uses OpenAI-compatible API endpoint.
        """
        import requests

        url = f"{self.config.server_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.server_api_key:
            headers["Authorization"] = f"Bearer {self.config.server_api_key}"

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if stream:
            # Streaming request using requests with stream=True
            response = requests.post(url, json=payload, headers=headers, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                            elif "tool_calls" in delta:
                                yield json.dumps({
                                    "type": "tool_call",
                                    "tool_calls": delta["tool_calls"],
                                })
                        except (json.JSONDecodeError, KeyError):
                            continue
        else:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            yield content

    def generate_response(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Generate a complete response (non-streaming).

        Args:
            user_message: User's message
            conversation_history: Previous conversation turns

        Returns:
            Generated response text
        """
        messages = self._build_messages(user_message, conversation_history)
        return "".join(self.generate(messages, stream=False))

    def generate_streaming(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Generator[str, None, None]:
        """
        Generate response with streaming tokens.

        Args:
            user_message: User's message
            conversation_history: Previous conversation turns

        Yields:
            Generated tokens
        """
        messages = self._build_messages(user_message, conversation_history)
        yield from self.generate(messages, stream=True, tools=self.get_tool_definitions())

    def _build_messages(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> List[Dict[str, str]]:
        """Build messages list for llama.cpp."""
        messages = []

        # Add system prompt
        messages.append({
            "role": "system",
            "content": "Bạn là một trợ lý AI hữu ích và thân thiện. Hãy trả lời bằng tiếng Việt khi được hỏi.",
        })

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add user message
        messages.append({
            "role": "user",
            "content": user_message,
        })

        return messages

    def execute_tool_call(self, tool_name: str, arguments: Dict) -> Any:
        """Execute a registered tool."""
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool = self._tools[tool_name]
        return tool.handler(**arguments)


def get_available_devices() -> str:
    """Get available device info for LLM."""
    import torch

    info = []

    if torch.cuda.is_available():
        info.append(f"CUDA: {torch.cuda.get_device_name(0)}")
        info.append(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        info.append("Apple Metal (MPS) available")
    else:
        info.append("CPU only")

    return "\n".join(info)

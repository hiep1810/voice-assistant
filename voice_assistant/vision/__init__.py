"""Vision integration for VLM (Vision Language Models)."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np
from pathlib import Path


@dataclass
class VisionResult:
    """Result from VLM inference."""
    text: str
    confidence: float = 1.0
    processing_time_ms: float = 0.0


class VisionLLM:
    """
    Vision Language Model wrapper using llama.cpp for GGUF VLM models.

    Supports:
    - Qwen3 VL 2B
    - Liquid LFM2 VL 1.6B
    - SmolVLM 500M
    """

    def __init__(self, model: str = "smolvlm"):
        self.model = model
        self._llm = None
        self._is_loaded = False

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load VLM model from GGUF."""
        from voice_assistant.config import VLM_MODELS
        from voice_assistant.llm import LlamaCppLLM, LLMConfig

        model_info = VLM_MODELS.get(self.model)
        if not model_info:
            raise ValueError(f"Unknown VLM model: {self.model}")

        # Configure for VLM
        config = LLMConfig(
            model=self.model,
            model_path=model_path,
            hf_id=model_info.get("hf_id"),
            context_length=model_info.get("context", 4096),
        )

        self._llm = LlamaCppLLM(config)
        self._llm.load_model()
        self._is_loaded = True

    def analyze_image(
        self,
        image_path: str,
        prompt: str = "What's in this image?",
    ) -> VisionResult:
        """
        Analyze an image and return description.

        Args:
            image_path: Path to image file
            prompt: Question about the image

        Returns:
            VisionResult with analysis
        """
        import time
        start = time.time()

        if not self._is_loaded:
            self.load_model()

        # Generate response with image
        # llama.cpp VLM API
        response = self._llm._model.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"file://{image_path}"},
                    ]
                }
            ],
            max_tokens=512,
        )

        text = response["choices"][0]["message"]["content"]

        return VisionResult(
            text=text,
            processing_time_ms=(time.time() - start) * 1000,
        )

    def analyze_image_streaming(
        self,
        image_path: str,
        prompt: str = "What's in this image?",
    ):
        """Analyze image with streaming tokens."""
        if not self._is_loaded:
            self.load_model()

        for token in self._llm._model.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"file://{image_path}"},
                    ]
                }
            ],
            max_tokens=512,
            stream=True,
        ):
            delta = token["choices"][0]["delta"]
            if "content" in delta:
                yield delta["content"]


class ScreenCapture:
    """Screen capture utility."""

    @staticmethod
    def capture(region: Optional[Dict[str, int]] = None) -> np.ndarray:
        """
        Capture screen or region.

        Args:
            region: Dict with x, y, width, height (or None for full screen)

        Returns:
            Screen image as numpy array (RGB)
        """
        try:
            import pyautogui
            screen = pyautogui.screenshot()

            if region:
                screen = screen.crop((
                    region["x"],
                    region["y"],
                    region["x"] + region["width"],
                    region["y"] + region["height"],
                ))

            return np.array(screen)

        except ImportError:
            raise ImportError("pyautogui not installed. Run: pip install pyautogui")

    @staticmethod
    def save(filepath: str, region: Optional[Dict[str, int]] = None) -> str:
        """Capture screen and save to file."""
        import PIL.Image

        image = ScreenCapture.capture(region)
        pil_image = PIL.Image.fromarray(image)
        pil_image.save(filepath)
        return filepath

    @staticmethod
    def list_monitors() -> List[Dict]:
        """List available monitors."""
        try:
            import pyautogui
            # pyautogui doesn't have multi-monitor support built-in
            # Use screeninfo instead
            try:
                from screeninfo import get_monitors
                monitors = get_monitors()
                return [
                    {
                        "name": m.name,
                        "x": m.x,
                        "y": m.y,
                        "width": m.width,
                        "height": m.height,
                    }
                    for m in monitors
                ]
            except ImportError:
                # Fallback to single monitor
                return [{
                    "name": "Primary",
                    "x": 0,
                    "y": 0,
                    "width": pyautogui.size().width,
                    "height": pyautogui.size().height,
                }]
        except Exception as e:
            return [{"error": str(e)}]


class CameraCapture:
    """Camera capture utility."""

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self._cap = None

    def open(self) -> bool:
        """Open camera."""
        try:
            import cv2
            self._cap = cv2.VideoCapture(self.camera_id)
            return self._cap.isOpened()
        except ImportError:
            raise ImportError("opencv-python not installed. Run: pip install opencv-python")

    def capture(self) -> Optional[np.ndarray]:
        """Capture single frame from camera."""
        if self._cap is None:
            self.open()

        if self._cap is not None and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                # Convert BGR to RGB
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return None

    def save(self, filepath: str) -> Optional[str]:
        """Capture frame and save to file."""
        import PIL.Image

        frame = self.capture()
        if frame is not None:
            pil_image = PIL.Image.fromarray(frame)
            pil_image.save(filepath)
            return filepath
        return None

    def close(self) -> None:
        """Close camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# High-level vision API

def analyze_screen(prompt: str = "What's on this screen?") -> str:
    """Capture screen and analyze with VLM."""
    # Save screenshot
    screenshot_path = Path.home() / ".voice_assistant" / "screenshots" / "current.png"
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)

    ScreenCapture.save(str(screenshot_path))

    # Analyze with VLM
    vlm = VisionLLM(model="smolvlm")
    result = vlm.analyze_image(str(screenshot_path), prompt)

    return result.text


def analyze_camera(prompt: str = "What do you see?") -> str:
    """Capture camera frame and analyze with VLM."""
    # Capture frame
    capture_path = Path.home() / ".voice_assistant" / "captures" / "camera.png"
    capture_path.parent.mkdir(parents=True, exist_ok=True)

    with CameraCapture() as cam:
        cam.save(str(capture_path))

    # Analyze with VLM
    vlm = VisionLLM(model="smolvlm")
    result = vlm.analyze_image(str(capture_path), prompt)

    return result.text


def analyze_image(image_path: str, prompt: str = "What's in this image?") -> str:
    """Analyze image file with VLM."""
    vlm = VisionLLM(model="smolvlm")
    result = vlm.analyze_image(image_path, prompt)
    return result.text

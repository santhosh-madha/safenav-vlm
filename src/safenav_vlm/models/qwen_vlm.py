"""Qwen vision-language model wrapper for SafeNav-VLM."""

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

DEFAULT_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"


class QwenVLM:
    """Local Qwen VLM running through MLX."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name

        print(f"Loading model: {model_name}")
        self.model, self.processor = load(model_name)
        self.config = load_config(model_name)

    def generate_response(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 100,
    ) -> str:
        """Generate a response for one image and prompt."""

        formatted_prompt = apply_chat_template(
            self.processor,
            self.config,
            prompt,
            num_images=1,
        )

        result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            image=[image_path],
            max_tokens=max_tokens,
            verbose=False,
        )

        return result.text.strip()

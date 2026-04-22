import time
import logging
from typing import Any, List, Optional
from google import genai
from google.genai.types import GenerateContentConfig, Part, Content
from llm.base import BaseLLMProvider
from utils.config import LlmConfig

logger = logging.getLogger(__name__)

class GeminiProvider(BaseLLMProvider):
    def __init__(self, config: LlmConfig) -> None:
        self.config = config.gemini
        self.client = genai.Client(api_key=self.config.api_key)
        logger.info(f"Gemini {self.config.model} готов")

    async def generate_response(self, user_text: str, image_bytes: Optional[bytes] = None, prompt: str = "", history: Optional[List[Any]] = None) -> str:
        try:
            start = time.perf_counter()
            config = GenerateContentConfig(
                system_instruction=prompt,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                thinking_config=self.config.thinking_config
            )
            contents: List[Content] = []
            if history:
                contents = [Content.model_validate(entry) for entry in history]
            user_parts: List[Part] = [Part.from_text(text=user_text)]
            if image_bytes:
                user_parts.append(
                    Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                )
            user_content = Content(role="user", parts=user_parts)
            contents.append(user_content)
            response = await self.client.aio.models.generate_content(
                model=self.config.model,
                contents=contents, # pyright: ignore[reportArgumentType]
                config=config
            )
            text = response.text or "Gemini вернул пустой ответ."
            logger.info(f"LLM: Gemini ответил за {time.perf_counter() - start:.2f} сек")
            return text.strip()

        except Exception as e:
            logger.error(f"Ошибка Gemini: {str(e)}")
            return f"Ошибка Gemini: {str(e)}"
    
    async def unload(self) -> None:
        pass

    async def warmup(self) -> None:
        pass
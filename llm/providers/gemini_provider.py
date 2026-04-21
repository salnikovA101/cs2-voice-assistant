import time
import logging
from typing import Any, Dict, List, Optional
from google import genai
from google.genai.types import GenerateContentConfig, Part, ThinkingConfig, ThinkingLevel
from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class GeminiProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config["llm"]
        self.model_name = self.config["model"]
        self.client = genai.Client(api_key=self.config["api_key"])
        logger.info(f"Gemini {self.model_name} готов")

    async def generate_response(self, user_text: str, image_bytes: Optional[bytes] = None, prompt: str = "", history: Optional[List[Any]] = None) -> str:
        try:
            start = time.perf_counter()
            config = GenerateContentConfig(
                system_instruction=prompt,
                temperature=self.config.get("temperature", 0.9),
                max_output_tokens=self.config.get("max_output_tokens", 2000),
                thinking_config=ThinkingConfig(thinking_level=ThinkingLevel.HIGH)
            )
            contents = history or []
            contents.append(user_text)
            if image_bytes:
                contents.append(Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            text = response.text or "Gemini вернул пустой ответ."
            logger.info(f"LLM: Gemini ответил за {time.perf_counter() - start:.2f} сек")
            return text.strip()

        except Exception as e:
            logger.error(f"Ошибка Gemini: {str(e)}")
            return f"Ошибка Gemini: {str(e)}"
    
    def unload(self) -> None:
        pass

    def warmup(self) -> None:
        pass
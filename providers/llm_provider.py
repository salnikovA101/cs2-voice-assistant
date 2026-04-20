import asyncio
import time
import logging
from typing import Any, Dict, Deque
from collections import deque

from google import genai
from google.genai.types import GenerateContentConfig, Part, ThinkingConfig, ThinkingLevel

logger = logging.getLogger(__name__)

class LLMProvider:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config["llm"]
        self.model_name = self.config["model"]
        self.client = genai.Client(api_key=self.config["api_key"])
        self.history: Deque[Dict[str, str]] = deque(maxlen=self.config.get("history_len", 4))
        logger.info(f"Gemini {self.model_name} готов")

    async def generate_response(self, prompt: str, user_text: str, image_bytes: bytes) -> str:
        try:
            start = time.perf_counter()
            config = GenerateContentConfig(
                system_instruction=prompt,
                temperature=self.config.get("temperature", 0.9),
                max_output_tokens=self.config.get("max_output_tokens", 2000),
                thinking_config=ThinkingConfig(thinking_level="high")
            )

            contents = []

            for elem in self.history:
                contents.append(f"User: {elem['q']}")
                contents.append(f"Assistant: {elem['a']}")

            contents.append(user_text)
            contents.append(Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )

            text = response.text or "Gemini вернул пустой ответ."
            self.history.append({'q': user_text, 'a': text.strip()})
            logger.info(f"LLM: Gemini ответил за {time.perf_counter() - start:.2f} сек")
            return text.strip()

        except Exception as e:
            logger.error(f"Ошибка Gemini: {str(e)}")
            return f"Ошибка Gemini: {str(e)}"
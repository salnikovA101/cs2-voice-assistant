import asyncio
import time
import logging
from typing import Any, Dict

from google import genai
from google.genai.types import GenerateContentConfig, Part, ThinkingConfig, ThinkingLevel

logger = logging.getLogger(__name__)

class LLMProvider:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config["llm"]
        self.model_name = self.config["model"]
        self.system_prompt = self._load_prompt(config)
        self.client = genai.Client(api_key=self.config["api_key"])
        logger.info(f"Gemini {self.model_name} готов")

    def _load_prompt(self, config: Dict[str, Any]) -> str:
        if config["llm"].get("prompt_mode", "smart") == "smart":
            with open("core/prompts/smart.txt", encoding="utf-8") as file:
                prompt = file.read().strip()
        else:
            with open("core/prompts/humor.txt", encoding="utf-8") as file:
                prompt = file.read().strip()

        if config["tts"].get("mode", "speed") == "speed":
            with open("core/prompts/silero_fix.txt", encoding="utf-8") as file:
                prompt += f"\n{file.read().strip()}"
        
        return prompt

    async def generate_response(self, user_text: str, image_bytes: bytes) -> str:
        try:
            start = time.perf_counter()
            
            config = GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.config.get("temperature", 0.9),
                max_output_tokens=self.config.get("max_output_tokens", 2000),
                thinking_config=ThinkingConfig(thinking_level="high")
            )

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=[user_text, Part.from_bytes(data=image_bytes, mime_type="image/jpeg")],
                config=config
            )

            logger.info(f"LLM: Gemini ответил за {time.perf_counter() - start:.2f} сек")
            text = response.text or "Gemini вернул пустой ответ."
            return text.strip()

        except Exception as e:
            logger.error(f"Ошибка Gemini: {str(e)}")
            return f"Ошибка Gemini: {str(e)}"
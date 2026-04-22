import time
import logging
from typing import Any, Dict, List, Optional
from ollama import AsyncClient
from llm.base import BaseLLMProvider
from utils.config import LlmConfig

logger = logging.getLogger(__name__)

class OllamaProvider(BaseLLMProvider):
    def __init__(self, config: LlmConfig) -> None:
        self.config = config.ollama
        self.client = AsyncClient(host=self.config.host)
        logger.info(f"Ollama {self.config.model} готов")

    async def generate_response(self, user_text: str, image_bytes: Optional[bytes] = None, prompt: str = "", history: Optional[List[Any]] = None) -> str:
        try:
            start = time.perf_counter()
            messages: List[Any] = []
            if prompt:
                messages.append({"role": "system", "content": prompt})
            if history:
                messages.extend(history)
            user_msg: Dict[str, Any] = {"role": "user", "content": user_text}
            if image_bytes:
                user_msg["images"] = [image_bytes]
            messages.append(user_msg)
            response = await self.client.chat(
                model=self.config.model,
                messages=messages,
                think=True,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_output_tokens,
                    "num_ctx": self.config.num_ctx
                }
            )
            text = response["message"]["content"].strip() or "Ollama вернул пустой ответ."
            logger.info(f"LLM: Ollama ответил за {time.perf_counter() - start:.2f} сек")
            return text

        except Exception as e:
            logger.error(f"Ошибка Ollama: {e}")
            return f"Ошибка Ollama: {e}"
    
    async def unload(self) -> None:
        try:
            await self.client.generate(model=self.config.model, prompt="", keep_alive=0)
            logger.info(f"Ollama: модель {self.config.model} выгружена.")
        except Exception as e:
            logger.error(f"Ошибка выгрузки Ollama: {e}")

    async def warmup(self) -> None:
        try:
            await self.client.generate(model=self.config.model, prompt="", keep_alive=-1)
            logger.info(f"Ollama: модель {self.config.model} прогрета")
        except Exception as e:
            logger.error(f"Ошибка прогрева Ollama: {e}")
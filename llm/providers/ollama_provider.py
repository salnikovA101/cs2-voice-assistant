import logging
import httpx

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Провайдер Ollama.

    Lifecycle через нативный Ollama REST API:
      - unload: POST {host}/api/generate  {keep_alive: 0}
      - warmup: POST {host}/api/generate  {keep_alive: -1, options: {num_ctx: ...}}
    """

    async def unload(self) -> None:
        """
        Выгружает модель из памяти Ollama (устанавливает keep_alive: 0).
        """
        url = f"{self.profile.host}/api/generate"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url,
                    json={"model": self.profile.model, "keep_alive": 0},
                )
                resp.raise_for_status()
            logger.info(f"[OllamaProvider] Модель '{self.profile.model}' выгружена")
        except Exception as e:
            logger.warning(f"[OllamaProvider] Не удалось выгрузить модель: {e}")

    async def warmup(self) -> None:
        """
        Загружает модель в память Ollama (устанавливает keep_alive: -1).
        """
        url = f"{self.profile.host}/api/generate"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    url,
                    json={
                        "model": self.profile.model,
                        "prompt": "",
                        "keep_alive": -1,
                        "options": {
                            "num_ctx": self.profile.context_length,
                        },
                    },
                )
                resp.raise_for_status()
            logger.info(
                f"[OllamaProvider] '{self.profile.model}' загружена "
                f"(num_ctx={self.profile.context_length})"
            )
        except Exception as e:
            logger.warning(f"[OllamaProvider] Не удалось прогреть модель: {e}")

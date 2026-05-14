import logging
from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    Провайдер для OpenAI-совместимых API (OpenAI, Gemini, Groq и др.).

    Управление жизненным циклом модели (load/unload) не поддерживается —
    модель живёт на стороне провайдера.
    """

    async def unload(self) -> None:
        """
        Метод-заглушка. OpenAI-совместимые API не требуют ручной выгрузки.
        """
        logger.debug("[OpenAIProvider] выгрузка не нужна")

    async def warmup(self) -> None:
        """
        Метод-заглушка. OpenAI-совместимые API не требуют ручного прогрева.
        """
        logger.debug("[OpenAIProvider] прогрев не нужен")

import time
import logging
from typing import Any, List, Optional, Callable, Dict
from google import genai
from google.genai.types import GenerateContentConfig, Part, Content
from llm.base import BaseLLMProvider
from utils.config import LlmConfig

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """
    Провайдер для взаимодействия с моделями Google Gemini через GenAI SDK.

    Поддерживает мультимодальные запросы (текст + изображения), системные инструкции
    и работу с историей диалога.
    """

    def __init__(self, config: LlmConfig) -> None:
        """
        Инициализирует клиент Gemini с использованием API ключа.

        Args:
            config (LlmConfig): Общий объект конфигурации LlmConfig, содержащий секцию .gemini.
        """

        self.config = config.gemini
        self.client = genai.Client(api_key=self.config.api_key)
        logger.info(f"Gemini {self.config.model} готов")

    async def generate_response(
        self,
        user_text: str,
        image_bytes: Optional[bytes] = None,
        prompt: str = "",
        history: Optional[List[Any]] = None,
        tools: Optional[List[Callable]] = None,
        tool_map: Optional[Dict[str, Callable]] = None,
    ) -> str:
        """
        Отправляет запрос к модели Gemini и возвращает сгенерированный текст.

        Метод формирует конфигурацию генерации, собирает историю сообщений,
        добавляет пользовательский текст и опциональное изображение.

        Args:
            user_text (str): Текст сообщения пользователя.
            image_bytes (Optional[bytes]): Бинарные данные изображения (JPEG).
            prompt (str): Системная инструкция для модели.
            history (Optional[List[Any]]): Список предыдущих сообщений для контекста.
            tools (Optional[List[Callable]]): Список доступных инструментов (функций).
            tool_map (Optional[Dict[str, Callable]]): Карта сопоставления имен функций и их реализаций.

        Returns:
            str: Ответ модели или сообщение об ошибке.
        """
        try:
            start = time.perf_counter()
            config = GenerateContentConfig(
                system_instruction=prompt,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                thinking_config=self.config.thinking_config,
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
                contents=contents,  # pyright: ignore[reportArgumentType]
                config=config,
            )
            logger.debug(response)
            text = response.text or "Gemini вернул пустой ответ."
            logger.debug(
                f"LLM: Gemini ответил за {time.perf_counter() - start:.2f} сек"
            )
            return text.strip()

        except Exception as e:
            logger.error(f"Ошибка Gemini: {str(e)}")
            return f"Ошибка Gemini: {str(e)}"

    async def unload(self) -> None:
        """
        Заглушка для метода выгрузки модели (для Gemini не требуется локальное освобождение).
        """
        pass

    async def warmup(self) -> None:
        """
        Заглушка для метода прогрева модели (подготовка сетевого соединения или кеша).
        """
        pass

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Callable, Dict


class BaseLLMProvider(ABC):
    """Абстрактный базовый класс для реализации провайдеров LLM."""

    @abstractmethod
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
        Генерирует текстовый ответ на основе входных данных.

        Args:
            user_text (str): Текст запроса пользователя.
            image_bytes (Optional[bytes]): Опциональное изображение.
            prompt (str): Системный промпт.
            history (Optional[List[Any]]): История диалога.
            tools (Optional[List[Callable]]): Список доступных инструментов.
            tool_map (Optional[Dict[str, Callable]]): Карта функций для вызова.

        Returns:
            str: Ответ модели.
        """
        pass

    @abstractmethod
    async def unload(self) -> None:
        """
        Освобождает ресурсы текущей активной модели.
        """
        pass

    @abstractmethod
    async def warmup(self) -> None:
        """Прогрев/запуск модели."""
        pass

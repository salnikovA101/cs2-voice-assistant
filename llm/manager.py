import logging
from typing import Any, Dict, List, Optional
from llm.history_manager import HistoryManager
from llm.base import BaseLLMProvider
from llm.providers.gemini_provider import GeminiProvider
from llm.providers.ollama_provider import OllamaProvider
from llm.prompt_loader import PromptLoader
from tools.registry import Tools
from utils.constants import LLMModelNames
from utils.config import LlmConfig

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Менеджер для переключения между различными LLM провайдерами и управления контекстом.

    Реализует динамическую смену моделей (например, с Ollama на Gemini)
    и интеграцию инструментов (tools).
    """

    MODELS: Dict[LLMModelNames, Any] = {
        LLMModelNames.GEMINI: GeminiProvider,
        LLMModelNames.OLLAMA: OllamaProvider,
    }

    HISTORY: Dict[LLMModelNames, Any] = {
        LLMModelNames.GEMINI: "get_gemini",
        LLMModelNames.OLLAMA: "get_ollama",
    }

    def __init__(self, config: LlmConfig) -> None:
        """
        Инициализирует менеджер LLM, загружает конфигурацию, промпты и инструменты.

        Args:
            config (LlmConfig): Объект конфигурации с настройками моделей.
        """
        self.config = config
        self.loaded_model = config.current_model
        self.prompt_manager = PromptLoader(config.prompt_folder)
        self.history_manager = HistoryManager(config.history_len)
        self.tools = Tools(switch_model_callback=self._switch_to_game_mode)
        self.model: BaseLLMProvider = self._load(config.current_model)

    def _switch_to_game_mode(self):
        """
        Callback-метод для принудительного переключения текущей модели на Gemini.
        Вызывается через инструменты (tools).
        """
        self.config.current_model = LLMModelNames.GEMINI

    async def generate_response(
        self, user_text: str, image_bytes: Optional[bytes] = None
    ) -> str:
        """
        Генерирует ответ от LLM, учитывая контекст, промпты и доступные инструменты.

        Если в процессе работы `self.config.current_model` изменился, метод
        автоматически выгружает старую модель и инициализирует новую.

        Args:
            user_text (str): Текстовый запрос пользователя.
            image_bytes (Optional[bytes]): Бинарные данные изображения (если есть).

        Returns:
            str: Текстовый ответ, сгенерированный моделью.
        """
        if self.loaded_model != self.config.current_model:
            await self.model.unload()
            self.loaded_model = self.config.current_model
            self.model = self._load(self.config.current_model)
            await self.model.warmup()
        prompt = self.prompt_manager.get_prompt(self.config.prompt_mode)
        history = self._get_history(self.config.current_model)
        logger.debug(prompt)
        logger.debug(history)
        text = await self.model.generate_response(
            user_text=user_text,
            image_bytes=image_bytes,
            prompt=prompt,
            history=history,
            tools=self.tools.get_tools_list(),
            tool_map=self.tools.get_tool_map(),
        )
        self.history_manager.add_entry(user_text, text)
        return text

    async def unload(self) -> None:
        """
        Освобождает ресурсы текущей активной модели.
        """
        await self.model.unload()

    def _load(self, name: LLMModelNames) -> BaseLLMProvider:
        """
        Инстанцирует провайдера LLM на основе переданного имени.

        Args:
            name (LLMModelNames): Название модели для загрузки.

        Returns:
            BaseLLMProvider: Экземпляр класса провайдера.
        """
        model_class = self.MODELS.get(name, GeminiProvider)
        model = model_class(self.config)
        return model

    def _get_history(self, name: LLMModelNames) -> List[Any]:
        """
        Получает историю диалога, отформатированную под конкретного провайдера.

        Args:
            name (LLMModelNames): Название модели, для которой нужна история.

        Returns:
            List[Any]: Список объектов истории в формате, который понимает SDK провайдера.
        """
        method_name = self.HISTORY.get(name, "get_gemini")
        history_type = getattr(self.history_manager, method_name)
        return history_type()

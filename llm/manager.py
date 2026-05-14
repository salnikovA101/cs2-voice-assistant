import logging
from datetime import datetime
from typing import Optional, Tuple

from llm.base import BaseLLMProvider
from llm.history_manager import HistoryManager
from llm.prompt_loader import PromptLoader
from llm.providers.lm_studio_provider import LMStudioProvider
from llm.providers.ollama_provider import OllamaProvider
from llm.providers.openai_provider import OpenAIProvider
from tools.registry import ToolRegistry
from utils.constants import LLMProviderType
from utils.config import AppConfig

logger = logging.getLogger(__name__)

_PROVIDER_MAP: dict[LLMProviderType, type[BaseLLMProvider]] = {
    LLMProviderType.OPENAI: OpenAIProvider,
    LLMProviderType.OLLAMA: OllamaProvider,
    LLMProviderType.LM_STUDIO: LMStudioProvider,
}


class LLMManager:
    """
    Менеджер для работы с LLM провайдером.

    Инициализирует нужный провайдер по полю `current_profile` из конфига (LlmConfig).
    Делегирует вызовы generate_response, unload и warmup активному провайдеру.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Инициализирует менеджер LLM, промпты, историю и реестр инструментов.

        Args:
            config (AppConfig): Полная конфигурация приложения.
        """
        self.config = config.llm
        self.loaded_profile = self.config.current_profile
        self.prompt_manager = PromptLoader(self.config.prompt_folder, config.tts.mode)
        self.history_manager = HistoryManager(self.config.history_len)
        self.tools = ToolRegistry(switch_model_callback=self._switch_to_game_mode)
        self.model: BaseLLMProvider = self._load(self.loaded_profile)

    def _switch_to_game_mode(self):
        """
        Callback-метод для принудительного переключения текущей модели на профиль gemini.
        Вызывается через инструменты (tools).
        """
        logger.info(f"Переключение в игровой режим (профиль: {self.config.game_mode})")
        self.config.current_profile = self.config.game_mode

    async def generate_response(
        self, user_text: str, image_bytes: Optional[bytes] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Основной метод генерации ответа. Обрабатывает смену профилей,
        подготовку промпта и истории, а также вызов инструментов.

        Args:
            user_text (str): Текст запроса пользователя.
            image_bytes (bytes, optional): Данные изображения в байтах.

        Returns:
            Tuple[str, Optional[str]]: Очищенный текст и тег инструкции.
        """
        if self.loaded_profile != self.config.current_profile:
            await self.model.unload()
            self.loaded_profile = self.config.current_profile
            self.model = self._load(self.loaded_profile)
            await self.model.warmup()

        prompt = self.prompt_manager.get_system_prompt()
        now = datetime.now().strftime("%d.%m.%Y %H:%M")
        prompt = prompt.replace("{current_datetime}", now)
        history = self.history_manager.get_history()
        logger.debug(prompt)
        logger.debug(history)

        clean_text, instruct = await self.model.generate_response(
            user_text=user_text,
            image_bytes=image_bytes,
            prompt=prompt,
            history=history,
            tools=self.tools.get_openai_tools(),
            tool_map=self.tools.get_tool_map(),
        )
        history_text = (
            f"<instruct>{instruct}</instruct>\n{clean_text}" if instruct else clean_text
        )
        self.history_manager.add_entry(user_text, history_text)
        return clean_text, instruct

    async def unload(self) -> None:
        """Выгружает текущую модель из памяти."""
        await self.model.unload()

    async def warmup(self) -> None:
        """Прогревает текущую модель для ускорения первого ответа."""
        await self.model.warmup()

    def _load(self, name: str) -> BaseLLMProvider:
        """
        Загружает конкретный экземпляр провайдера по имени профиля.

        Args:
            name (str): Имя профиля из конфигурации.

        Returns:
            BaseLLMProvider: Инициализированный провайдер.
        """
        profile = getattr(self.config.profiles, name, None)
        if not profile:
            raise ValueError(f"Профиль LLM '{name}' не найден в конфигурации")

        cls = _PROVIDER_MAP.get(profile.provider)
        if cls is None:
            raise ValueError(
                f"Неизвестный провайдер '{profile.provider}'. "
                f"Доступные: {[e.value for e in LLMProviderType]}"
            )

        logger.info(f"LLM провайдер: {cls.__name__} (профиль: '{name}')")
        return cls(
            profile,
            max_output_tokens=self.config.max_output_tokens,
            max_turns=self.config.max_turns,
        )

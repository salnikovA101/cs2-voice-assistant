from typing import Dict, Any
from tts.base import BaseTTSProvider
from tts.providers.quality_tts import QualityTTSProvider
from tts.providers.speed_tts import FastTTSProvider
from utils.constants import TTSModes
from utils.config import TtsConfig


class TTSManager:
    """
    Менеджер для управления провайдерами синтеза речи (TTS).

    Класс отвечает за инициализацию, динамическое переключение между
    различными режимами озвучки (скорость vs качество) и управление
    жизненным циклом моделей.
    """

    MODELS: Dict[TTSModes, Any] = {
        TTSModes.SPEED: FastTTSProvider,
        TTSModes.QUALITY: QualityTTSProvider,
    }

    def __init__(self, config: TtsConfig) -> None:
        """
        Инициализирует менеджер TTS и загружает модель по умолчанию.

        Args:
            config (TtsConfig): Объект конфигурации, содержащий текущий режим
                и параметры для TTS-провайдеров.
        """
        self.config = config
        self.loaded_mode = config.mode
        self.model: BaseTTSProvider = self._load_model(self.config.mode)

    async def voiceover(self, text: str) -> None:
        """
        Озвучивает переданный текст, используя активную модель.

        Если в конфигурации изменился режим (например, с 'SPEED' на 'QUALITY'),
        метод автоматически выгрузит старую модель и загрузит новую перед озвучкой.

        Args:
            text (str): Текст, который необходимо преобразовать в речь.
        """
        if self.loaded_mode != self.config.mode:
            self.unload()
            self.loaded_mode = self.config.mode
            self.model = self._load_model(self.config.mode)
        await self.model.voiceover(text)

    def _load_model(self, mode: TTSModes) -> BaseTTSProvider:
        """
        Внутренний метод для создания экземпляра провайдера и его подготовки.

        Args:
            mode (TTSModes): Режим работы, определяющий выбор класса провайдера.

        Returns:
            BaseTTSProvider: Инициализированный и "прогретый" экземпляр провайдера.
        """
        model_class = self.MODELS.get(mode, FastTTSProvider)
        model = model_class(self.config)
        model.warmup()
        return model

    def unload(self) -> None:
        """
        Освобождает ресурсы текущей загруженной модели.

        Вызывает внутренний метод `unload()` провайдера для очистки памяти
        или корректного завершения работы процессов.
        """
        self.model.unload()

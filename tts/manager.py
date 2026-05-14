from typing import Dict, Any, Optional
from tts.base import BaseTTSProvider
from tts.providers.copy_tts import CopyTTSProvider
from tts.providers.speed_tts import FastTTSProvider
from tts.providers.emotional_tts import EmotionalTTSProvider
from utils.constants import TTSModes
from utils.config import TtsConfig


class TTSManager:
    """
    Менеджер для управления провайдерами синтеза речи (TTS).

    Класс отвечает за инициализацию, динамическое переключение между
    различными режимами озвучки (скорость, копия, дефолт) и управление
    жизненным циклом моделей.
    """

    MODELS: Dict[TTSModes, Any] = {
        TTSModes.SPEED: FastTTSProvider,
        TTSModes.CLONE: CopyTTSProvider,
        TTSModes.EMOTIONAL: EmotionalTTSProvider,
    }

    def __init__(self, config: TtsConfig, ptt_key: str = "right ctrl") -> None:
        """
        Инициализирует менеджер TTS и загружает модель по умолчанию.

        Args:
            config (TtsConfig): Объект конфигурации, содержащий текущий режим
                и параметры для TTS-провайдеров.
            ptt_key (str): Клавиша Push-to-Talk для прерывания озвучки.
        """
        self.config = config
        self.ptt_key = ptt_key
        self.loaded_mode = config.mode
        self.model: BaseTTSProvider = self._load_model(self.config.mode)

    async def voiceover(self, text: str, instruct: Optional[str] = None) -> None:
        """
        Озвучивает переданный текст, используя активную модель.

        Если в конфигурации изменился режим (например, с 'SPEED' на 'COPY'),
        метод автоматически выгрузит старую модель и загрузит новую перед озвучкой.

        Args:
            text (str): Текст, который необходимо преобразовать в речь.
        """
        if self.loaded_mode != self.config.mode:
            self.unload()
            self.loaded_mode = self.config.mode
            self.model = self._load_model(self.config.mode)
        await self.model.voiceover(text, instruct=instruct)

    def _load_model(self, mode: TTSModes) -> BaseTTSProvider:
        """
        Внутренний метод для создания экземпляра провайдера и его подготовки.

        Args:
            mode (TTSModes): Режим работы, определяющий выбор класса провайдера.

        Returns:
            BaseTTSProvider: Инициализированный и "прогретый" экземпляр провайдера.
        """
        model_class = self.MODELS.get(mode, FastTTSProvider)
        model = model_class(self.config, ptt_key=self.ptt_key)
        model.warmup()
        return model

    def unload(self) -> None:
        """
        Освобождает ресурсы текущей загруженной модели.

        Вызывает внутренний метод `unload()` провайдера для очистки памяти
        или корректного завершения работы процессов.
        """
        self.model.unload()

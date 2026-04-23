import sounddevice as sd
import torch
import logging
import time
import asyncio
from silero import silero_tts
from tts.base import BaseTTSProvider
from utils.config import TtsConfig, SpeedTtsConfig

logger = logging.getLogger(__name__)


class FastTTSProvider(BaseTTSProvider):
    """
    Провайдер быстрого синтеза речи на базе Silero TTS.

    Обеспечивает минимальную задержку (latency) генерации, и нагрузку на видеокарту.
    """

    def __init__(self, config: TtsConfig) -> None:
        """
        Инициализирует Silero TTS, загружая модель и перемещая её на целевое устройство.

        Args:
            config (TtsConfig): Общий объект конфигурации, из которого извлекаются
                настройки секции 'speed'.
        """
        self.config: SpeedTtsConfig = config.speed
        self.device = torch.device(self.config.device)
        logger.info("Загрузка Silero TTS V5...")
        self.model, _ = silero_tts(  # type: ignore
            language=self.config.language, speaker=self.config.speaker_type
        )
        self.model.to(self.device)
        logger.info(f"Silero TTS готов. Спикер: {self.config.speaker_name}")

    async def voiceover(self, text: str) -> None:
        """
        Асинхронно запускает процесс озвучивания текста.

        Метод использует asyncio.to_thread для выполнения блокирующих операций
        генерации и воспроизведения аудио, сохраняя отзывчивость приложения.

        Args:
            text (str): Текст для синтеза речи.
        """
        if not text:
            return
        await asyncio.to_thread(self._voiceover_sync, text)

    def unload(self) -> None:
        """
        Удаляет модель из памяти и очищает кэш CUDA.

        Метод гарантирует освобождение ресурсов VRAM при переключении
        между быстрым и качественным режимами озвучки.
        """
        if hasattr(self, "model"):
            del self.model
            torch.cuda.empty_cache()
            logger.debug("Silero TTS выгружена из VRAM")

    def warmup(self) -> None:
        """
        Выполняет 'прогрев' модели тестовой фразой.

        Инициализирует веса модели в памяти, чтобы первое реальное
        сообщение озвучивалось без стартовой задержки.
        """
        self._voiceover_sync("Слушаю")

    def _voiceover_sync(self, text: str) -> None:
        """
        Синхронный метод генерации и немедленного воспроизведения аудио.

        Выполняет полный цикл: генерация тензора, конвертация в NumPy и
        блокирующее воспроизведение через sounddevice.

        Args:
            text (str): Текст для синтеза.
        """
        start = time.perf_counter()
        audio = self.model.apply_tts(
            text=text,
            speaker=self.config.speaker_name,
            sample_rate=self.config.sample_rate,
        )
        logger.debug(f"Сгенерировано за {time.perf_counter() - start:.3f} сек")
        audio_np = audio.cpu().numpy()

        sd.play(audio_np, self.config.sample_rate)
        sd.wait()

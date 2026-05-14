import sounddevice as sd
import torch
import keyboard
import logging
import time
import asyncio
import warnings
from typing import Optional
from silero import silero_tts
from tts.base import BaseTTSProvider
from utils.config import TtsConfig, SpeedTtsConfig

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


class FastTTSProvider(BaseTTSProvider):
    """
    Провайдер быстрого синтеза речи на базе Silero TTS.

    Обеспечивает минимальную задержку (latency) генерации, и нагрузку на видеокарту.
    """

    def __init__(self, config: TtsConfig, ptt_key: str = "right ctrl") -> None:
        """
        Инициализирует Silero TTS, загружая модель и перемещая её на целевое устройство.

        Args:
            config (TtsConfig): Общий объект конфигурации, из которого извлекаются
                настройки секции 'speed'.
            ptt_key (str): Клавиша Push-to-Talk для прерывания озвучки.
        """
        self.config: SpeedTtsConfig = config.speed
        self.ptt_key = ptt_key
        self.device = torch.device(self.config.device)
        logger.info("Загрузка Silero TTS V5...")
        self.model, _ = silero_tts(  # type: ignore
            language=self.config.language, speaker=self.config.speaker_type
        )
        self.model.to(self.device)
        logger.info(f"Silero TTS готов. Спикер: {self.config.speaker_name}")

    async def voiceover(self, text: str, instruct: Optional[str] = None) -> None:
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
        Выполняет тихий 'прогрев' модели тестовой фразой.

        Генерирует аудио без воспроизведения, чтобы инициализировать
        веса модели в памяти и исключить задержку при первом реальном запросе.
        """
        self.model.apply_tts(
            text="Прогрев",
            speaker=self.config.speaker_name,
            sample_rate=self.config.sample_rate,
        )
        logger.info("Silero TTS прогрет (тихо)")

    def _voiceover_sync(self, text: str) -> None:
        """
        Синхронный метод генерации и потокового воспроизведения аудио.

        Генерирует аудио целиком, затем воспроизводит чанками (~100мс)
        через OutputStream. Во время воспроизведения проверяет нажатие
        клавиши PTT для возможности прерывания (barge-in).

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

        chunk_size = self.config.sample_rate // 10  # ~100мс чанки
        with sd.OutputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
        ) as stream:
            for i in range(0, len(audio_np), chunk_size):
                if keyboard.is_pressed(self.ptt_key):
                    logger.info("Озвучка прервана пользователем")
                    break
                chunk = audio_np[i : i + chunk_size]
                stream.write(chunk.reshape(-1, 1))

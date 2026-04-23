from faster_whisper import WhisperModel
import time
import asyncio
import numpy as np
import logging
import numpy.typing as npt
from utils.config import SttConfig
from typing import Optional

logger = logging.getLogger(__name__)


class STTProvider:
    """
    Класс для обеспечения работы системы распознавания речи (Speech-to-Text)
    на базе модели Faster Whisper.
    """

    def __init__(self, config: SttConfig) -> None:
        """
        Инициализирует модель Whisper с заданными параметрами конфигурации.

        Args:
            config (SttConfig): Объект конфигурации, содержащий параметры модели.
        """
        self.config: SttConfig = config
        logger.info(f"Загрузка Whisper {self.config.model} на {self.config.device}...")
        self.model = WhisperModel(
            self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        self._warmup()
        logger.info("Whisper готов")

    def _warmup(self) -> None:
        """
        Выполняет 'прогрев' модели путем обработки пустого аудиосигнала.
        Это необходимо для инициализации весов и кэша, чтобы последующие
        реальные запросы обрабатывались без задержек.
        """
        logger.debug("Прогрев STT модели...")
        start = time.perf_counter()
        dummy_audio = np.zeros(16000, dtype=np.float32)
        self.model.transcribe(dummy_audio, beam_size=5, language="ru")
        logger.debug(f"Прогрев STT завершен за {time.perf_counter() - start:.3f} сек")

    async def transcribe(self, audio: npt.NDArray[np.float32]) -> Optional[str]:
        """
        Асинхронно преобразует аудиопоток в текст.

        Метод запускает блокирующую операцию транскрибации в отдельном потоке,
        чтобы не блокировать цикл событий asyncio.

        Args:
            audio (npt.NDArray[np.float32]): Аудиоданные в формате массива NumPy (float32).

        Returns:
            str: Распознанный текст. Возвращает None, если входной массив пуст.
        """
        if len(audio) == 0:
            return None

        start = time.perf_counter()
        text = await asyncio.to_thread(self._transcribe_sync, audio)
        logger.debug(f"STT: {time.perf_counter() - start:.3f} сек")
        return text.strip()

    def _transcribe_sync(self, audio: npt.NDArray[np.float32]) -> str:
        """
        Внутренний синхронный метод для выполнения транскрибации.

        Использует VAD (Voice Activity Detection) для фильтрации тишины и
        ограничивает распознавание русским языком.

        Args:
            audio (npt.NDArray[np.float32]): Аудиоданные для обработки.

        Returns:
            str: Склеенный текст всех распознанных сегментов аудио.
        """
        segments, _ = self.model.transcribe(
            audio, beam_size=5, vad_filter=True, language="ru"
        )
        return " ".join(seg.text for seg in segments)

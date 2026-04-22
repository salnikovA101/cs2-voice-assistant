from faster_whisper import WhisperModel
import time
import asyncio
import numpy as np
import logging
import numpy.typing as npt
from utils.config import SttConfig

logger = logging.getLogger(__name__)

class STTProvider:
    def __init__(self, config: SttConfig) -> None:
        self.config: SttConfig = config
        logger.info(f"Загрузка Whisper {self.config.model} на {self.config.device}...")
        self.model = WhisperModel(
            self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type
        )
        self._warmup()
        logger.info("Whisper готов")

    def _warmup(self) -> None:
        logger.info("Прогрев STT модели...")
        start = time.perf_counter()
        dummy_audio = np.zeros(16000, dtype=np.float32)
        self.model.transcribe(
            dummy_audio,
            beam_size=5,
            language="ru"
        )
        logger.info(f"Прогрев STT завершен за {time.perf_counter() - start:.3f} сек")

    async def transcribe(self, audio: npt.NDArray[np.float32]) -> str:
        if len(audio) == 0:
            return ""

        start = time.perf_counter()
        text = await asyncio.to_thread(self._transcribe_sync, audio)
        logger.info(f"STT: {time.perf_counter() - start:.3f} сек")
        return text.strip()

    def _transcribe_sync(self, audio: npt.NDArray[np.float32]) -> str:
        segments, _ = self.model.transcribe(
            audio,
            beam_size=5,
            vad_filter=True,
            language="ru"
        )
        return " ".join(seg.text for seg in segments)
    
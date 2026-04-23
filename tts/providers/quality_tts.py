import sounddevice as sd
import numpy as np
import torch
import logging
from faster_qwen3_tts import FasterQwen3TTS
import time
import asyncio
from tts.base import BaseTTSProvider
from utils.config import TtsConfig, QualityTtsConfig

logger = logging.getLogger(__name__)


class QualityTTSProvider(BaseTTSProvider):
    """
    Провайдер высококачественного синтеза речи на базе модели FasterQwen3TTS.

    Ориентирован на естественное звучание и клонирование голоса, используя
    возможности видеокарт NVIDIA для инференса в формате bfloat16.
    """

    def __init__(self, config: TtsConfig) -> None:
        """
        Инициализирует модель QwenTTS с загрузкой весов в видеопамять.

        Args:
            config (TtsConfig): Общий объект конфигурации, из которого извлекаются
                настройки секции 'quality'.
        """
        self.config: QualityTtsConfig = config.quality
        logger.info(f"Загрузка TTS {self.config.model}...")
        self.model = FasterQwen3TTS.from_pretrained(
            model_name=self.config.model,
            device=self.config.device,
            attn_implementation=self.config.attn_implementation,
            max_seq_len=self.config.max_seq_len,
            dtype=torch.bfloat16,
        )
        logger.info("QwenTTS готов")

    async def voiceover(self, text: str) -> None:
        """
        Асинхронно запускает процесс озвучивания текста.

        Метод перенаправляет выполнение в отдельный поток через asyncio.to_thread,
        чтобы генерация аудио и блокирующий вывод в sounddevice не фризили основной цикл.

        Args:
            text (str): Текст для преобразования в речь.
        """
        if not text:
            return
        await asyncio.to_thread(self._voiceover_sync, text)

    def unload(self) -> None:
        """
        Удаляет модель из памяти и принудительно очищает кэш CUDA.

        Это важно при переключении на другой провайдер, чтобы
        освободить VRAM для других задач или моделей.
        """
        if hasattr(self, "model"):
            del self.model
            torch.cuda.empty_cache()
            logger.debug("FasterQwen3TTS выгружена из VRAM")

    def warmup(self) -> None:
        """
        Выполняет 'прогрев' модели коротким тестовым словом.

        Это заставляет CUDA инициализировать ядра и аллоцировать память,
        чтобы избежать задержки при первом реальном обращении.
        """
        self._voiceover_sync("Слушаю")

    def _voiceover_sync(self, text: str) -> None:
        """
        Синхронный метод генерации и потокового воспроизведения аудио.

        Использует генератор модели для получения аудио-чанков и сразу пишет
        их в выходной поток sounddevice. Реализует логику замера времени
        до появления первого фрагмента звука.

        Args:
            text (str): Текст для синтеза.
        """
        start = time.perf_counter()
        first_chunk = True
        with sd.OutputStream(
            samplerate=24000,
            channels=1,
            dtype="float32",
            blocksize=2048,
            latency="high",
        ) as stream:
            for audio_chunk, _, _ in self.model.generate_voice_clone_streaming(
                text=text,
                language=self.config.language,
                ref_audio=self.config.ref_voice,
                ref_text=self.config.ref_text,
                chunk_size=self.config.chunk_size,
                xvec_only=True,
            ):
                stream.write(np.asarray(audio_chunk, dtype=np.float32).reshape(-1, 1))

                if first_chunk:
                    logger.debug(f"TTS: {time.perf_counter() - start:.2f} сек")
                    first_chunk = False

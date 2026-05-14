import sounddevice as sd
import numpy as np
import torch
import keyboard
import logging
from faster_qwen3_tts import FasterQwen3TTS
import time
import asyncio
from typing import Optional
from tts.base import BaseTTSProvider
from utils.config import TtsConfig, EmotionalTtsConfig

logger = logging.getLogger(__name__)


class EmotionalTTSProvider(BaseTTSProvider):
    """
    Провайдер TTS на базе Qwen3TTS, использующий предобученный голос (CustomVoice).
    """

    def __init__(self, config: TtsConfig, ptt_key: str = "right ctrl") -> None:
        """
        Инициализирует эмоциональный TTS на базе Qwen3TTS.

        Args:
            config (TtsConfig): Конфигурация TTS.
            ptt_key (str): Клавиша для прерывания озвучки.
        """
        self.config: EmotionalTtsConfig = config.emotional
        self.ptt_key = ptt_key
        logger.info(f"Загрузка Emotional TTS {self.config.model}...")
        self.model = FasterQwen3TTS.from_pretrained(
            model_name=self.config.model,
            device=self.config.device,
            attn_implementation=self.config.attn_implementation,
            max_seq_len=self.config.max_seq_len,
            dtype=torch.bfloat16,
        )
        logger.info("Emotional QwenTTS готов")

    async def voiceover(self, text: str, instruct: Optional[str] = None) -> None:
        """
        Озвучивает текст с учетом переданной инструкции (эмоции).
        """
        if not text:
            return
        await asyncio.to_thread(self._voiceover_sync, text, instruct)

    def unload(self) -> None:
        """
        Выгружает модель и очищает VRAM.
        """
        if hasattr(self, "model"):
            del self.model
            torch.cuda.empty_cache()
            logger.debug("FasterQwen3TTS (Emotional) выгружена из VRAM")

    def warmup(self) -> None:
        """
        Прогрев модели с использованием предустановленного голоса (voice_id).
        """
        for _ in self.model.generate_custom_voice_streaming(
            text="Прогрев",
            language=self.config.language,
            speaker=self.config.voice_id,
            chunk_size=self.config.chunk_size,
        ):
            break
        logger.info("Emotional QwenTTS прогрет (тихо)")

    def _voiceover_sync(self, text: str, instruct: Optional[str] = None) -> None:
        """
        Синхронная генерация и воспроизведение аудиопотока.
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
            for audio_chunk, _, _ in self.model.generate_custom_voice_streaming(
                text=text,
                language=self.config.language,
                speaker=self.config.voice_id,
                chunk_size=self.config.chunk_size,
                instruct=instruct,
            ):
                if keyboard.is_pressed(self.ptt_key):
                    logger.info("Озвучка прервана пользователем")
                    break

                stream.write(np.asarray(audio_chunk, dtype=np.float32).reshape(-1, 1))

                if first_chunk:
                    logger.debug(
                        f"Emotional TTS: {time.perf_counter() - start:.2f} сек"
                    )
                    first_chunk = False

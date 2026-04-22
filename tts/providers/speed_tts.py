import sounddevice as sd
import numpy as np
import torch
import logging
import time
import asyncio
from typing import Any, Dict
from silero import silero_tts
from tts.base import BaseTTSProvider
from utils.config import TtsConfig, SpeedTtsConfig

logger = logging.getLogger(__name__)

class FastTTSProvider(BaseTTSProvider):
    def __init__(self, config: TtsConfig) -> None:
        self.config: SpeedTtsConfig = config.speed
        self.device = torch.device(self.config.device)
        logger.info("Загрузка Silero TTS V5...")
        self.model, _ = silero_tts(language=self.config.language, speaker=self.config.speaker_type)
        self.model.to(self.device)
        logger.info(f"Fast TTS готов. Спикер: {self.config.speaker_name}")

    async def voiceover(self, text: str) -> None:
        if not text:
            return
        await asyncio.to_thread(self._voiceover_sync, text)

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model
            torch.cuda.empty_cache()
            logger.info("Silero TTS выгружена из VRAM")

    def warmup(self) -> None:
        self._voiceover_sync("Слушаю")

    def _voiceover_sync(self, text: str) -> None:
        start = time.perf_counter()
        audio = self.model.apply_tts(
            text=text,
            speaker=self.config.speaker_name,
            sample_rate=self.config.sample_rate
        )
        logger.info(f"Сгенерировано за {time.perf_counter() - start:.3f} сек")
        audio_np = audio.cpu().numpy()

        sd.play(audio_np, self.config.sample_rate)
        sd.wait()

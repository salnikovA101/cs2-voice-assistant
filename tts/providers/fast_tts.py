import sounddevice as sd
import numpy as np
import torch
import logging
import time
import asyncio
from typing import Any, Dict
from silero import silero_tts
from tts.base import BaseTTSProvider

logger = logging.getLogger(__name__)

class FastTTSProvider(BaseTTSProvider):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.tts_config = config["tts"]
        self.device = torch.device('cuda')
        logger.info("Загрузка Silero TTS V5...")
        self.model, _ = silero_tts(language='ru', speaker='v5_ru')
        self.model.to(self.device)
        self.speaker = self.tts_config.get("silero_speaker", "baya")
        self.sample_rate = self.tts_config.get("sample_rate", 24000)
        logger.info(f"Fast TTS готов. Спикер: {self.speaker}")

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
            speaker=self.speaker,
            sample_rate=self.sample_rate
        )
        logger.info(f"Сгенерировано за {time.perf_counter() - start:.3f} сек")
        audio_np = audio.cpu().numpy()

        sd.play(audio_np, self.sample_rate)
        sd.wait()

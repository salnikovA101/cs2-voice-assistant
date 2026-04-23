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
    def __init__(self, config: TtsConfig) -> None:
        self.config: QualityTtsConfig = config.quality
        logger.info(f"Загрузка TTS {self.config.model}...")
        self.model = FasterQwen3TTS.from_pretrained(
            model_name=self.config.model,
            device=self.config.device,
            attn_implementation=self.config.attn_implementation,
            max_seq_len=self.config.max_seq_len,
            dtype=torch.bfloat16
        )
        logger.info("QwenTTS готов")

    async def voiceover(self, text: str) -> None:
        if not text:
            return
        await asyncio.to_thread(self._voiceover_sync, text)
    
    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model
            torch.cuda.empty_cache()
            logger.debug("FasterQwen3TTS выгружена из VRAM")

    def warmup(self) -> None:
        self._voiceover_sync("Слушаю")

    def _voiceover_sync(self, text: str) -> None:
        start = time.perf_counter()
        first_chunk = True
        with sd.OutputStream(samplerate=24000, channels=1, dtype="float32", blocksize=2048, latency="high") as stream:
            for audio_chunk, _, _ in self.model.generate_voice_clone_streaming(
                text=text,
                language=self.config.language,
                ref_audio=self.config.ref_voice,
                ref_text=self.config.ref_text,
                chunk_size=self.config.chunk_size,
                xvec_only=True
            ):
                stream.write(np.asarray(audio_chunk, dtype=np.float32).reshape(-1, 1))

                if first_chunk:
                    logger.debug(f"TTS: {time.perf_counter() - start:.2f} сек")
                    first_chunk = False

import asyncio
import logging
import cv2
import numpy as np
from typing import Any, Dict
from tts.manager import TTSManager
from providers.stt_provider import STTProvider
from providers.ocr_provider import OCRProvider
from llm.manager import LLMManager
from utils.audio_recorder import AudioRecorder
from utils.constants import LLMModelNames, TTSModes

logger = logging.getLogger(__name__)

class Assistant:
    def __init__(self, config: Dict[str, Any], tts_mode: TTSModes = TTSModes.SPEED) -> None:
        self.config: Dict[str, Any] = config
        self.stt = STTProvider(config)
        self.tts = TTSManager(config, tts_mode)
        self.llm = LLMManager(config)
        self.ocr = OCRProvider()
        self.recorder = AudioRecorder()
        self.tts_mode = tts_mode
    
    def _process_image(self, image_tensor: np.ndarray) -> bytes:
        image_bgr = cv2.cvtColor(image_tensor, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buffer.tobytes()

    async def run_pipeline(self) -> None:
        key: str = self.config["general"]["push_to_talk_key"]
        audio_data, image_tensor = await asyncio.gather(
            self.recorder.record(key),
            self.ocr.get_screen()
        )

        if len(audio_data) == 0:
            return

        text, image_bytes = await asyncio.gather(
            self.stt.transcribe(audio_data),
            asyncio.to_thread(self._process_image, image_tensor)
        )

        if not text:
            return

        logger.info(f"Ты сказал: {text}")

        answer = await self.llm.generate_response(
            user_text=text,
            image_bytes=image_bytes,
        )

        logger.info(f"Ответ LLM: {answer}")

        await self.tts.voiceover(
            text=answer,
            mode=self.tts_mode
        )
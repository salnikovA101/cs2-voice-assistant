import logging
import cv2
import numpy as np
from providers.stt_provider import STTProvider
from providers.tts_provider import TTSProvider
from providers.ocr_provider import OCRProvider
from providers.llm_provider import LLMProvider
from utils.audio_recorder import AudioRecorder
from typing import Any, Dict

logger = logging.getLogger(__name__)

class Assistant:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.stt = STTProvider(config)
        self.tts = TTSProvider(config)
        self.llm = LLMProvider(config, "")
        self.ocr = OCRProvider()
        self.recorder = AudioRecorder()
        self.tts._voiceover_sync("Слушаю")

    async def run_pipeline(self) -> None:
        key: str = self.config["general"]["push_to_talk_key"]
        audio_data = await self.recorder.record(key)

        if len(audio_data) == 0:
            return
        
        image_tensor = await self.ocr.get_screen()

        text = await self.stt.transcribe(audio_data)
        if not text:
            return

        logger.info(f"Ты сказал: {text}")
        
        image_bgr = cv2.cvtColor(image_tensor, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        image_bytes = buffer.tobytes()
        
        answer = await self.llm.generate_response(text, image_bytes)

        logger.info(f"Ответ LLM: {answer}")

        await self.tts.voiceover(answer)
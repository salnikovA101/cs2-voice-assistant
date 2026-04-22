import asyncio
import logging
import cv2
import numpy as np
from tts.manager import TTSManager
from providers.stt_provider import STTProvider
from providers.ocr_provider import OCRProvider
from llm.manager import LLMManager
from utils.audio_recorder import AudioRecorder
from utils.constants import LLMModelNames, TTSModes
from utils.config import AppConfig

logger = logging.getLogger(__name__)

class Assistant:
    def __init__(self, config: AppConfig) -> None:
        self.recorder = AudioRecorder(config.general)
        self.stt = STTProvider(config.stt)
        self.llm = LLMManager(config.llm)
        self.tts = TTSManager(config.tts)
        self.ocr = OCRProvider()
    
    def _process_image(self, image_tensor: np.ndarray) -> bytes:
        image_bgr = cv2.cvtColor(image_tensor, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buffer.tobytes()

    async def run_pipeline(self) -> None:
        audio_data, image_tensor = await asyncio.gather(
            self.recorder.record(),
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
            model_name=LLMModelNames.GEMINI
        )

        logger.info(f"Ответ LLM: {answer}")

        await self.tts.voiceover(
            text=answer,
            mode=TTSModes.SPEED
        )
import asyncio
import logging
import cv2
import numpy as np
from tts.manager import TTSManager
from providers.stt_provider import STTProvider
from providers.ocr_provider import OCRProvider
from llm.manager import LLMManager
from utils.audio_recorder import AudioRecorder
from utils.config import AppConfig

logger = logging.getLogger(__name__)


class Assistant:
    """
    Главный класс ассистента, координирующий работу STT, LLM, TTS и OCR.

    Attributes:
        config (AppConfig): Объект конфигурации приложения.
        recorder (AudioRecorder): Модуль для захвата аудио с микрофона.
        stt (STTProvider): Провайдер для распознавания речи.
        llm (LLMManager): Менеджер управления языковыми моделями.
        tts (TTSManager): Менеджер управления моделями озвучки.
        ocr (OCRProvider): Модуль для захвата изображения с экрана.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Инициализирует компоненты ассистента на основе конфигурации.

        Args:
            config (AppConfig): Конфигурация для всех подсистем.
        """
        self.config = config
        self.recorder = AudioRecorder(config.general)
        self.stt = STTProvider(config.stt)
        self.llm = LLMManager(config.llm)
        self.tts = TTSManager(config.tts)
        self.ocr = OCRProvider()

    def _process_image(self, image_tensor: np.ndarray) -> bytes:
        """
        Преобразует тензор изображения в формат JPEG для отправки в LLM.

        Args:
            image_tensor (np.ndarray): Исходное изображение с экрана (RGB).

        Returns:
            bytes: Сжатое изображение в формате JPEG.
        """
        image_bgr = cv2.cvtColor(image_tensor, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buffer.tobytes()

    async def run_pipeline(self) -> None:
        """
        Запускает полный цикл обработки: ожидание нажатия, запись,
        распознавание, генерация ответа и озвучка.
        """
        await self.recorder.wait_for_press()

        image_tensor = await self.ocr.get_screen()

        audio_data = await self.recorder.record()

        if len(audio_data) == 0:
            return

        text, image_bytes = await asyncio.gather(
            self.stt.transcribe(audio_data),
            asyncio.to_thread(self._process_image, image_tensor),
        )

        if not text:
            return

        logger.info(f"Ты сказал: {text}")

        answer = await self.llm.generate_response(
            user_text=text, image_bytes=image_bytes
        )

        logger.info(f"Ответ LLM: {answer}")

        await self.tts.voiceover(text=answer)

import asyncio
import logging
import threading
import queue
from typing import Optional
import keyboard
import cv2
import numpy as np

from tts.manager import TTSManager
from providers.stt_provider import STTProvider
from providers.screen_capture import ScreenCapture
from llm.manager import LLMManager
from utils.audio_recorder import AudioRecorder
from utils.config import AppConfig

logger = logging.getLogger(__name__)


class Assistant:
    """
    Главный класс ассистента, координирующий работу STT, LLM, TTS и ScreenCapture.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Инициализирует основные компоненты ассистента: LLM, STT, TTS и захват экрана.

        Args:
            config (AppConfig): Объект полной конфигурации приложения.
        """
        self.config = config
        self.llm = LLMManager(config)
        self.screen: Optional[ScreenCapture] = None
        self.recorder: Optional[AudioRecorder] = None
        self.stt: Optional[STTProvider] = None
        self.tts: Optional[TTSManager] = None

        if self.config.general.image:
            self.screen = ScreenCapture()

        if self.config.general.enable_voice_input:
            self.recorder = AudioRecorder(config.general)
            self.stt = STTProvider(config.stt)

        if self.config.general.enable_voice_output:
            self.tts = TTSManager(config.tts, ptt_key=config.general.push_to_talk_key)

        self.text_queue: queue.Queue[str] = queue.Queue()
        if self.config.general.enable_text_input:
            self.input_thread = threading.Thread(target=self._read_console, daemon=True)
            self.input_thread.start()

    def _read_console(self):
        """
        Фоновый поток для чтения текстового ввода из консоли.
        Позволяет пользователю вводить команды текстом параллельно с голосовым вводом.
        """
        while True:
            try:
                line = input()
                if line.strip():
                    self.text_queue.put(line.strip())
            except EOFError:
                break

    def _process_image(self, image_tensor: np.ndarray) -> bytes:
        """
        Преобразует тензор изображения в формат JPEG для отправки в LLM.
        """
        image_bgr = cv2.cvtColor(image_tensor, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buffer.tobytes()

    async def run_pipeline(self) -> None:
        """
        Запускает полный цикл обработки: ожидание нажатия или ввода текста,
        захват экрана, распознавание, генерация ответа и озвучка.
        """
        text = None
        image_bytes = None

        while True:
            if self.config.general.enable_text_input and not self.text_queue.empty():
                text = self.text_queue.get()
                logger.info(f"Введен текст: {text}")
                if self.config.general.image:
                    image_tensor = await self.screen.get_screen()
                    image_bytes = await asyncio.to_thread(
                        self._process_image, image_tensor
                    )
                break

            if self.config.general.enable_voice_input and self.recorder is not None and keyboard.is_pressed(
                self.recorder.ptt_key
            ):
                if self.config.general.image and self.screen is not None:
                    audio_data, image_tensor = await asyncio.gather(
                        self.recorder.record(), self.screen.get_screen()
                    )
                else:
                    audio_data = await self.recorder.record()
                    image_tensor = None

                if len(audio_data) > 0:
                    try:
                        if self.config.general.image and self.stt is not None and image_tensor is not None:
                            text, image_bytes = await asyncio.wait_for(
                                asyncio.gather(
                                    self.stt.transcribe(audio_data),
                                    asyncio.to_thread(
                                        self._process_image, image_tensor
                                    ),
                                ),
                                timeout=30,
                            )
                        elif self.stt is not None:
                            text = await asyncio.wait_for(
                                self.stt.transcribe(audio_data), timeout=30
                            )
                            image_bytes = None
                        logger.info(f"Ты сказал: {text}")
                    except asyncio.TimeoutError:
                        logger.error("STT/image_process завис, пропуск итерации")
                break

            await asyncio.sleep(0.01)

        if not text:
            return

        try:
            answer, instruct = await asyncio.wait_for(
                self.llm.generate_response(user_text=text, image_bytes=image_bytes),
                timeout=120,
            )
        except asyncio.TimeoutError:
            answer = "LLM не смог ответить за 120 сек"
            instruct = None
            logger.error(answer)

        logger.info(f"Ответ LLM: {answer}")
        if instruct:
            logger.info(f"Эмоция (instruct): {instruct}")

        if self.config.general.enable_voice_output:
            await self.tts.voiceover(text=answer, instruct=instruct)

    async def startup(self) -> None:
        """
        Инициализация: загружает LLM модель с параметрами из конфига.
        """
        await self.llm.warmup()

    async def shutdown(self) -> None:
        """
        Завершение: выгружает все модели и освобождает ресурсы.
        """
        await self.llm.unload()
        if self.config.general.enable_voice_output and self.tts is not None:
            self.tts.unload()
        if self.config.general.image and self.screen is not None:
            self.screen.release()
        logger.info("Ресурсы освобождены")

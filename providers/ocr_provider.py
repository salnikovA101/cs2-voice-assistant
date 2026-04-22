import numpy as np
import dxcam
import asyncio
import logging

logger = logging.getLogger(__name__)

class OCRProvider:
    def __init__(self) -> None:
        logger.info("Инициализация DXcam...")
        try:
            self.camera = dxcam.create(output_color="RGB")
        except Exception as e:
            logger.error(f"Ошибка инициализации DXcam: {e}")
            raise RuntimeError("DXcam недоступен.")
            
        logger.info("OCR (Screen Capture) готов")

    async def get_screen(self) -> np.ndarray:
        return await asyncio.to_thread(self._get_screen_sync)
    
    def _get_screen_sync(self) -> np.ndarray:
        frame = self.camera.grab()
        while frame is None:
            frame = self.camera.grab()
        logger.info("OCR: сделал снимок")
        return frame
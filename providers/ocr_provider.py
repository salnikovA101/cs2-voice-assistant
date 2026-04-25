import numpy as np
import dxcam
import asyncio
import logging

logger = logging.getLogger(__name__)


class OCRProvider:
    """
    Класс для высокоскоростного захвата экрана с использованием библиотеки DXcam.
    """

    def __init__(self) -> None:
        """
        Инициализирует объект камеры DXcam для захвата изображения в формате RGB.

        Raises:
            RuntimeError: Если DXcam не удается инициализировать (например,
            из-за отсутствия графического устройства или прав доступа).
        """
        logger.info("Инициализация DXcam...")
        try:
            self.camera = dxcam.create(output_color="RGB")
        except Exception as e:
            logger.error(f"Ошибка инициализации DXcam: {e}")
            raise RuntimeError("DXcam недоступен.")

        logger.info("DXcam готов")

    async def get_screen(self) -> np.ndarray:
        """
        Асинхронно захватывает текущий кадр экрана.

        Использует `asyncio.to_thread`, чтобы блокирующий вызов захвата кадра
        не останавливал основной цикл событий (event loop).

        Returns:
            np.ndarray: Изображение экрана в виде массива NumPy (формат RGB).
        """
        return await asyncio.to_thread(self._get_screen_sync)

    def _get_screen_sync(self) -> np.ndarray:
        """
        Синхронный метод захвата кадра.

        В случае, если камера возвращает пустой кадр (например, при временном
        отсутствии обновлений на экране), метод продолжает попытки захвата
        до получения валидного изображения.

        Returns:
            np.ndarray: Данные изображения кадра.
        """
        frame = self.camera.grab()
        while frame is None:
            frame = self.camera.grab()
        logger.debug("OCR: сделал снимок")
        return frame
    
    def release(self) -> None:
        """
        Освобождает DXcam и DirectX ресурсы.
        """
        try:
            if hasattr(self, "camera") and self.camera:
                self.camera.release()
                logger.debug("DXcam освобождён")
        except Exception as e:
            logger.error(f"Ошибка при освобождении DXcam: {e}")

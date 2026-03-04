import numpy as np
import cv2
import base64
import mss
import asyncio

class OCRProvider:
    def __init__(self) -> None:
        print("OCR готов")

    async def get_screen(self) -> str:
        return await asyncio.to_thread(self._get_screen_sync)
    
    def _get_screen_sync(self) -> str:
        with mss.mss() as sct:
            screenshot = sct.grab(sct.monitors[1])
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        _, buffer = cv2.imencode(
            '.jpg',
            img,
            [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        print("OCR: сделал снимок")
        return base64.b64encode(buffer.tobytes()).decode('utf-8')
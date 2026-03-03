from faster_whisper import WhisperModel
import time
import asyncio
import numpy as np
from typing import Any, Dict
import numpy.typing as npt


class STTProvider:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config["stt"]
        print(f"Загрузка Whisper {self.config['model']} на {self.config['device']}...")
        self.model = WhisperModel(
            self.config["model"],
            device=self.config["device"],
            compute_type=self.config["compute_type"],
        )
        print("Whisper готов")

    async def transcribe(self, audio: npt.NDArray[np.float32]) -> str:
        if len(audio) == 0:
            return ""

        start = time.perf_counter()
        text = await asyncio.to_thread(self._transcribe_sync, audio)
        print(f"STT: {time.perf_counter() - start:.3f} сек")
        return text.strip()

    def _transcribe_sync(self, audio: npt.NDArray[np.float32]) -> str:
        segments, _ = self.model.transcribe(
            audio,
            beam_size=5,
            vad_filter=True,
            language="ru"
        )
        return " ".join(seg.text for seg in segments)

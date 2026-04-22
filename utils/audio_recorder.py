import numpy as np
import sounddevice as sd
import asyncio
import keyboard
import logging
from typing import List
import numpy.typing as npt
from utils.config import GeneralConfig

logger = logging.getLogger(__name__)

class AudioRecorder:
    def __init__(self, config: GeneralConfig, samplerate: int = 16000) -> None:
        self.ptt_key = config.push_to_talk_key
        self.samplerate: int = samplerate
        self.recording: bool = False
        self.chunks: List[np.ndarray] = []

    def _callback(self, indata: np.ndarray, frames, time_info, status):
        if self.recording:
            self.chunks.append(indata.copy())

    async def record(self) -> npt.NDArray[np.float32]:
        self.chunks.clear()
        self.recording = False

        while not keyboard.is_pressed(self.ptt_key):
            await asyncio.sleep(0.01)

        logger.info(f"Отпусти '{self.ptt_key}', чтобы закончить")
        self.recording = True

        with sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            callback=self._callback,
            dtype="float32",
        ):
            while keyboard.is_pressed(self.ptt_key):
                await asyncio.sleep(0.03)

        self.recording = False

        if not self.chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(self.chunks, axis=0).flatten()
        return audio.astype(np.float32)
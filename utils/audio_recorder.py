import numpy as np
import sounddevice as sd
import asyncio
import keyboard
from typing import List
import numpy.typing as npt

class AudioRecorder:
    def __init__(self, samplerate: int = 16000) -> None:
        self.samplerate: int = samplerate
        self.recording: bool = False
        self.chunks: List[np.ndarray] = []

    def _callback(self, indata: np.ndarray, frames, time, status):
        if self.recording:
            self.chunks.append(indata.copy())

    async def record(self, key: str) -> npt.NDArray[np.float32]:
        self.chunks.clear()
        self.recording = False

        while not keyboard.is_pressed(key):
            await asyncio.sleep(0.01)

        print(f"Отпусти '{key}' чтобы закончить")
        self.recording = True

        with sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            callback=self._callback,
            dtype="float32",
        ):
            while keyboard.is_pressed(key):
                await asyncio.sleep(0.03)

        self.recording = False

        if not self.chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(self.chunks, axis=0).flatten()
        return audio.astype(np.float32)
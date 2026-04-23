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
    """
    Класс для управления захватом аудио через микрофон с использованием механики Push-to-Talk.

    Обеспечивает ожидание нажатия горячей клавиши, циклическую запись аудиоданных
    в буфер и последующую склейку в единый массив для обработки STT.
    """

    def __init__(self, config: GeneralConfig, samplerate: int = 16000) -> None:
        """
        Инициализирует рекордер и настраивает параметры захвата.

        Args:
            config (GeneralConfig): Общая конфигурация приложения, содержащая клавишу активации.
            samplerate (int, optional): Частота дискретизации аудио. По умолчанию 16000 Гц,
                что является стандартом для моделей семейства Whisper.
        """
        self.ptt_key = config.push_to_talk_key
        self.samplerate: int = samplerate
        self.recording: bool = False
        self.chunks: List[np.ndarray] = []

    async def wait_for_press(self) -> None:
        """
        Асинхронно ожидает нажатия заданной клавиши Push-to-Talk.

        Использует короткие паузы (sleep), чтобы не перегружать процессор
        в цикле ожидания.
        """
        while not keyboard.is_pressed(self.ptt_key):
            await asyncio.sleep(0.01)

    def _callback(self, indata: np.ndarray, frames, time_info, status):
        """
        Внутренний обработчик (callback) для входящего аудиопотока.

        Вызывается библиотекой sounddevice каждый раз, когда заполнится
        внутренний буфер захвата.

        Args:
            indata (np.ndarray): Массив с захваченными аудиоданными.
            frames (int): Количество кадров в буфере.
            time_info (Any): Временные метки потока.
            status (sd.CallbackFlags): Статус ошибок потока (например, переполнение).
        """
        if self.recording:
            self.chunks.append(indata.copy())

    async def record(self) -> npt.NDArray[np.float32]:
        """
        Запускает и останавливает запись аудио по удержанию клавиши.

        Метод открывает входной поток микрофона, записывает данные, пока клавиша
        удерживается, и по окончании объединяет все накопленные фрагменты
        в один плоский массив NumPy.

        Returns:
            npt.NDArray[np.float32]: Объединенный аудиосигнал в формате float32.
                Возвращает пустой массив, если запись не была произведена.
        """
        self.chunks.clear()

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

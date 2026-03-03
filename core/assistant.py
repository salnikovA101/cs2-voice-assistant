from providers.stt_provider import STTProvider
from utils.audio_recorder import AudioRecorder
from typing import Any, Dict


class Assistant:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.stt = STTProvider(config)
        self.recorder = AudioRecorder()

    async def run_pipeline(self) -> None:
        key: str = self.config["general"]["push_to_talk_key"]
        audio_data = await self.recorder.record(key)

        if len(audio_data) == 0:
            return

        text = await self.stt.transcribe(audio_data)
        if not text:
            return

        print(f"Ты сказал: {text}")
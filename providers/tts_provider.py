import sounddevice as sd
import numpy as np
import torch
from faster_qwen3_tts import FasterQwen3TTS
from typing import Any, Dict
import time
import asyncio

class TTSProvider:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config["tts"]
        print(f"Загрузка TTS {self.config['model']}...")
        self.model = FasterQwen3TTS.from_pretrained(
            model_name=self.config["model"],
            attn_implementation=self.config["attn_implementation"],
            max_seq_len=self.config["max_seq_len"],
            dtype=torch.bfloat16
        )
        print("TTS готов")

    async def voiceover(self, text: str) -> None:
        if not text:
            return
        await asyncio.to_thread(self._voiceover_sync, text)

    def _voiceover_sync(self, text: str) -> None:
        start = time.time()
        first_chunk = True
        with sd.OutputStream(samplerate=24000, channels=1, dtype="float32") as stream:
            for audio_chunk, _, _ in self.model.generate_voice_clone_streaming(
                text=text,
                language="Russian",
                ref_audio=self.config["voice"],
                ref_text="",
                chunk_size=6,
                xvec_only=True
            ):
                stream.write(np.asarray(audio_chunk, dtype=np.float32).reshape(-1, 1))

                if first_chunk:
                    print(f"TTS: {time.time() - start:.2f} сек")
                    first_chunk = False
        torch.cuda.empty_cache()

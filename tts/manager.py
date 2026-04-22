from typing import Optional, Dict, Any
from tts.base import BaseTTSProvider
from tts.providers.quality_tts import QualityTTSProvider
from tts.providers.speed_tts import FastTTSProvider
from utils.constants import TTSModes
from utils.config import TtsConfig

class TTSManager:
    MODELS: Dict[TTSModes, Any] = {
        TTSModes.SPEED: FastTTSProvider,
        TTSModes.QUALITY: QualityTTSProvider
    }

    def __init__(self, config: TtsConfig) -> None:
        self.config = config
        self.loaded_mode = config.mode
        self.model: BaseTTSProvider = self._load_model(self.config.mode)
    
    async def voiceover(self, text: str) -> None:
        if self.loaded_mode != self.config.mode:
            self._unload()
            self.loaded_mode = self.config.mode
            self.model = self._load_model(self.config.mode)
        await self.model.voiceover(text)

    def _load_model(self, mode: TTSModes) -> BaseTTSProvider:
        model_class = self.MODELS.get(mode, FastTTSProvider)
        model = model_class(self.config)
        model.warmup()
        return model

    def _unload(self) -> None:
        self.model.unload()

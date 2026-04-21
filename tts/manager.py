from typing import Optional, Dict, Any
from tts.base import BaseTTSProvider
from tts.providers.quality_tts import QualityTTSProvider
from tts.providers.fast_tts import FastTTSProvider
from utils.constants import TTSModes

class TTSManager:
    MODELS: Dict[TTSModes, Any] = {
        TTSModes.SPEED: FastTTSProvider,
        TTSModes.QUALITY: QualityTTSProvider
    }

    def __init__(self, config: Dict[str, Any], mode: TTSModes = TTSModes.SPEED) -> None:
        self.mode: TTSModes = mode
        self.config = config
        self.model: BaseTTSProvider = self._load_model(mode)
    
    async def voiceover(self, text: str, mode: Optional[TTSModes] = None) -> None:
        if mode and mode != self.mode:
            self._unload()
            self.model = self._load_model(mode)
            self.mode = mode
        await self.model.voiceover(text)

    def _load_model(self, mode: TTSModes) -> BaseTTSProvider:
        model_class = self.MODELS.get(mode, FastTTSProvider)
        model = model_class(self.config)
        model.warmup()
        return model

    def _unload(self) -> None:
        self.model.unload()

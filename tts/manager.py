from typing import Optional, Dict, Any
from tts.providers.quality_tts import QualityTTSProvider
from tts.providers.fast_tts import FastTTSProvider
from tts.base import BaseTTSProvider

class TTSManager:
    def __init__(self, config: Dict[str, Any], mode: str = "speed") -> None:
        self.mode: str = mode
        self.config = config
        self.model: BaseTTSProvider = self._load_model(mode)
    
    async def voiceover(self, text: str, mode: Optional[str] = None) -> None:
        if mode and mode != self.mode:
            self._unload()
            self.model = self._load_model(mode)
            self.mode = mode
        await self.model.voiceover(text)

    def _load_model(self, mode: str) -> BaseTTSProvider:
        models = {
            "speed": FastTTSProvider,
            "quality": QualityTTSProvider
        }
        model_class = models.get(mode, FastTTSProvider)
        model = model_class(self.config)
        model.warmup()
        return model

    def _unload(self) -> None:
        self.model.unload()

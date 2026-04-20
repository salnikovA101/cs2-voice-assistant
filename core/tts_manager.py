from typing import Optional, Dict, Any
from tts.providers.quality_tts import TTSProvider
from tts.providers.fast_tts import FastTTSProvider

class TTSManager:
    def __init__(self, config: Dict[str, Any], mode: str = "speed") -> None:
        self.mode: str = mode
        self.config = config
        self.model: FastTTSProvider|TTSProvider = self._load_model(mode)
    
    async def voiceover(self, text: str, mode: Optional[str] = None) -> None:
        if mode and mode != self.mode:
            self._unload(self.model)
            self.model = self._load_model(mode)
            self.mode = mode
        await self.model.voiceover(text)

    def _load_model(self, mode: str) -> FastTTSProvider|TTSProvider:
        if mode == "speed":
            model = FastTTSProvider(self.config)
        else:
            model = TTSProvider(self.config)
        model._voiceover_sync("Мхм") # warming up
        return model

    def _unload(self, model: Optional[FastTTSProvider|TTSProvider]) -> None:
        if model:
            model.unload()

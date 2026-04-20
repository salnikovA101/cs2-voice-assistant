from abc import ABC, abstractmethod

class BaseTTSProvider(ABC):
    @abstractmethod
    async def voiceover(self, text: str) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass
    
    @abstractmethod
    def warmup(self) -> None:
        pass
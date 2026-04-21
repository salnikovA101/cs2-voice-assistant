from abc import ABC, abstractmethod
from typing import List, Any, Optional

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_response(self, user_text: str, image_bytes: Optional[bytes] = None, prompt: str = "", history: Optional[List[Any]] = None) -> str:
        pass

    @abstractmethod
    async def unload(self) -> None:
        pass

    @abstractmethod
    async def warmup(self) -> None:
        pass
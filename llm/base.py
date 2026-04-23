from abc import ABC, abstractmethod
from typing import List, Any, Optional, Callable, Dict

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_response(
        self,
        user_text: str,
        image_bytes: Optional[bytes] = None,
        prompt: str = "",
        history: Optional[List[Any]] = None,
        tools: Optional[List[Callable]] = None,
        tool_map: Optional[Dict[str, Callable]] = None
    ) -> str:
        pass

    @abstractmethod
    async def unload(self) -> None:
        pass

    @abstractmethod
    async def warmup(self) -> None:
        pass
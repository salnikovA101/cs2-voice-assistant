from llm.base import BaseLLMProvider
from typing import Any, Dict, List, Optional

class OllamaProvider(BaseLLMProvider):
    def __init__(self) -> None:
        pass

    async def generate_response(self, user_text: str, image_bytes: Optional[bytes] = None, prompt: str = "", history: Optional[List[Any]] = None) -> str:
        return ""
    
    def unload(self) -> None:
        pass

    def warmup(self) -> None:
        pass
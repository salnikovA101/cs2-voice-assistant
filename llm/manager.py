from typing import Any, Dict, List, Optional
from llm.history_manager import HistoryManager
from llm.base import BaseLLMProvider
from llm.providers.gemini_provider import GeminiProvider
from llm.providers.ollama_provider import OllamaProvider
from llm.prompt_loader import PromptLoader

class LLMManager:
    def __init__(self, config: Dict[str, Any], name: str = "gemini", history_size: int = 4, prompt_folder: str = "prompts", base_prompt_name: str = "smart") -> None:
        self.config = config
        self.model_name = name
        self.current_prompt_name = base_prompt_name
        self.model: BaseLLMProvider = self._load(name)
        self.history_manager = HistoryManager(history_size)
        self.prompt_manager = PromptLoader(prompt_folder)
        

    async def generate_response(self, user_text: str, image_bytes: Optional[bytes] = None, model_name: Optional[str] = None) -> str:
        if model_name and model_name != self.model_name:
            self.model.unload()
            self.model_name = model_name
            self.model = self._load(model_name)
            self.model.warmup()
        prompt = self.prompt_manager.get_prompt(self.current_prompt_name)
        history = self._get_history(self.model_name)
        text = await self.model.generate_response(
            user_text=user_text,
            image_bytes=image_bytes,
            prompt=prompt,
            history=history
        )
        self.history_manager.add_entry(user_text, text)
        return text

    def _load(self, name: str) -> BaseLLMProvider:
        MODELS = {
            "gemini": GeminiProvider,
            "ollama": OllamaProvider
        }
        model_class = MODELS.get(name, GeminiProvider)
        model = model_class(self.config)
        model.warmup()
        return model
    
    def _get_history(self, name: str) -> List[Any]:
        HISTORY = {
            "gemini": self.history_manager.get_gemini,
            "ollama": self.history_manager.get_ollama
        }
        history_type = HISTORY.get(name, self.history_manager.get_gemini)
        return history_type()
    
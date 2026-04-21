from typing import Any, Dict, List, Optional
from llm.history_manager import HistoryManager
from llm.base import BaseLLMProvider
from llm.providers.gemini_provider import GeminiProvider
from llm.providers.ollama_provider import OllamaProvider
from llm.prompt_loader import PromptLoader
from utils.constants import LLMModelNames

class LLMManager:
    MODELS: Dict[LLMModelNames, Any] = {
        LLMModelNames.GEMINI: GeminiProvider,
        LLMModelNames.OLLAMA: OllamaProvider
    }

    HISTORY: Dict[LLMModelNames, Any] = {
        LLMModelNames.GEMINI: "get_gemini",
        LLMModelNames.OLLAMA: "get_ollama"
    }

    def __init__(self, config: Dict[str, Any], name: LLMModelNames = LLMModelNames.GEMINI, history_size: int = 4, prompt_folder: str = "prompts", base_prompt_name: str = "smart") -> None:
        self.config = config
        self.model_name = name
        self.current_prompt_name = base_prompt_name
        self.prompt_manager = PromptLoader(prompt_folder)
        self.history_manager = HistoryManager(history_size)
        self.model: BaseLLMProvider = self._load(name)

    async def generate_response(self, user_text: str, image_bytes: Optional[bytes] = None, model_name: Optional[LLMModelNames] = None) -> str:
        if model_name and model_name != self.model_name:
            await self.model.unload()
            self.model_name = model_name
            self.model = self._load(model_name)
            await self.model.warmup()
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

    def _load(self, name: LLMModelNames) -> BaseLLMProvider:
        model_class = self.MODELS.get(name, GeminiProvider)
        model = model_class(self.config)
        return model
    
    def _get_history(self, name: LLMModelNames) -> List[Any]:
        method_name = self.HISTORY.get(name, "get_gemini")
        history_type = getattr(self.history_manager, method_name)
        return history_type()
    
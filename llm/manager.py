from typing import Any, Dict, List, Optional
from llm.history_manager import HistoryManager
from llm.base import BaseLLMProvider
from llm.providers.gemini_provider import GeminiProvider
from llm.providers.ollama_provider import OllamaProvider
from llm.prompt_loader import PromptLoader
from utils.constants import LLMModelNames
from utils.config import LlmConfig

class LLMManager:
    MODELS: Dict[LLMModelNames, Any] = {
        LLMModelNames.GEMINI: GeminiProvider,
        LLMModelNames.OLLAMA: OllamaProvider
    }

    HISTORY: Dict[LLMModelNames, Any] = {
        LLMModelNames.GEMINI: "get_gemini",
        LLMModelNames.OLLAMA: "get_ollama"
    }

    def __init__(self, config: LlmConfig) -> None:
        self.config = config
        self.prompt_manager = PromptLoader(config.prompt_folder)
        self.history_manager = HistoryManager(config.history_len)
        self.model: BaseLLMProvider = self._load(config.current_model)

    async def generate_response(self, user_text: str, image_bytes: Optional[bytes] = None, model_name: Optional[LLMModelNames] = None) -> str:
        if model_name and model_name != self.config.current_model:
            await self.model.unload()
            self.config.current_model = model_name
            self.model = self._load(model_name)
            await self.model.warmup()
        prompt = self.prompt_manager.get_prompt(self.config.prompt_mode)
        history = self._get_history(self.config.current_model)
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
    
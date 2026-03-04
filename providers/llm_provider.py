from ollama import AsyncClient
import os
import time
from typing import Any, Dict, List

class LLMProvider:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config["llm"]
        self.client = AsyncClient(host=self.config["host"])
        self.system_prompt: str = self._load_prompt()
        print("LLM готов")

    def _load_prompt(self) -> str:
        mode = self.config["prompt_mode"]
        filename = "smart.txt" if mode == "smart" else "humor.txt"
        path = f"core/prompts/{filename}"
        
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return f.read().strip()
        return ""

    async def generate_response(self, user_text: str, screen_base64: str) -> str:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

        messages.append({
            "role": "user",
            "content": user_text,
            "images": [screen_base64]
        })

        try:
            start = time.perf_counter()
            response = await self.client.chat(
                model=self.config["model"],
                messages=messages,
                options={
                    "temperature": self.config["temperature"],
                    "num_predict": self.config["num_predict"],
                    "num_ctx": self.config["num_ctx"]
                }
            )
            print(f"LLM: {time.perf_counter() - start:.3f} сек")
            return response["message"]["content"].strip()
        
        except Exception as e:
            return f"Ошибка при обращении к LLM {e}."
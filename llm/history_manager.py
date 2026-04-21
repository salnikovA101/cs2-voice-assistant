from collections import deque
from typing import Deque, Dict, List, Any

class HistoryManager:
    def __init__(self, max_len: int) -> None:
        self.history: Deque[Dict[str, str]] = deque(maxlen=max_len)

    def add_entry(self, user_text: str, assistant_text: str) -> None:
        self.history.append({"user": user_text, "assistant": assistant_text})
    
    def get_gemini(self) -> List[Any]:
        contents: List[str] = []
        for entry in self.history:
            contents.append(f"User: {entry["user"]}")
            contents.append(f"Assistant: {entry["assistant"]}")
        return contents

    def get_ollama(self) -> List[Dict[str, str]]:
        contents: List[Dict[str, str]] = []
        for entry in self.history:
            contents.append({"role": "user", "content": entry["user"]})
            contents.append({"role": "assistant", "content": entry["assistant"]})
        return contents
    
    def clear_history(self) -> None:
        self.history.clear()
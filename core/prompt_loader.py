from pathlib import Path
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class PromptLoader:
    def __init__(self, folder_name: str) -> None:
        self.prompts: Dict[str, str] = {}
        self._load(folder_name)
    
    def _load(self, folder_name: str) -> None:
        path = Path(folder_name)
        if not path.is_dir():
            logger.error(f"Путь {folder_name} не существует или не является директорией.")
            return

        for file_path in path.glob("*.txt"):
            try:
                self.prompts[file_path.stem] = file_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.error(f"Ошибка при чтении файла {file_path.name}: {e}")
        
        logger.info(f"Теущие промпты: {sorted(self.prompts)}")

    def get_prompt(self, name: str) -> str:
        prompt = self.prompts.get(name, "")
        format = self.prompts.get("response_format", "")
        return prompt + "\n" + format

    def update_folder(self, folder_name: str) -> None:
        self.clear()
        self._load(folder_name)
    
    def clear(self) -> None:
        self.prompts.clear()
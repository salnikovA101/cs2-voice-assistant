from pathlib import Path
import logging
from utils.constants import TTSModes

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Класс для загрузки и управления текстовыми промптами.
    Загружает базовую логику (logic.txt) и формат вывода для TTS (output_{mode}.txt).
    """

    def __init__(self, folder_name: str, mode: TTSModes) -> None:
        """
        Инициализирует загрузчик и считывает файлы.
        """
        self.logic_text = ""
        self.output_text = ""
        self.mode = mode
        self._load(folder_name)

    def _load(self, folder_name: str) -> None:
        """
        Загружает тексты промптов из файлов logic.txt и output_{mode}.txt.

        Args:
            folder_name (str): Имя папки с промптами.
        """
        path = Path(folder_name)
        if not path.is_dir():
            logger.error(
                f"Путь {folder_name} не существует или не является директорией."
            )
            return

        logic_file = path / "logic.txt"
        output_file = path / f"output_{self.mode.value}.txt"

        try:
            if logic_file.exists():
                self.logic_text = logic_file.read_text(encoding="utf-8").strip()
                logger.info("Промпт logic.txt успешно загружен.")
            else:
                logger.warning("Файл logic.txt не найден.")

            if output_file.exists():
                self.output_text = output_file.read_text(encoding="utf-8").strip()
                logger.info(f"Промпт {output_file.name} успешно загружен.")
            else:
                logger.warning(f"Файл {output_file.name} не найден.")

        except Exception as e:
            logger.error(f"Ошибка при чтении файлов промптов: {e}")

    def get_system_prompt(self) -> str:
        """
        Возвращает объединенный текст промптов.
        """
        return f"{self.logic_text}\n\n{self.output_text}".strip()

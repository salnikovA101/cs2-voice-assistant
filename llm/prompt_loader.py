from pathlib import Path
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Класс для загрузки и управления текстовыми промптами из локальной директории.

    Автоматически считывает все файлы .txt в указанной папке и предоставляет
    удобный доступ к их содержимому по имени файла.
    """

    def __init__(self, folder_name: str) -> None:
        """
        Инициализирует загрузчик и запускает процесс чтения файлов.

        Args:
            folder_name (str): Путь к директории, содержащей текстовые файлы промптов.
        """
        self.prompts: Dict[str, str] = {}
        self._load(folder_name)

    def _load(self, folder_name: str) -> None:
        """
        Сканирует директорию и загружает содержимое всех .txt файлов в словарь.

        Имена файлов (без расширения) становятся ключами, а их содержимое — значениями.

        Args:
            folder_name (str): Путь к директории для сканирования.
        """
        path = Path(folder_name)
        if not path.is_dir():
            logger.error(
                f"Путь {folder_name} не существует или не является директорией."
            )
            return

        for file_path in path.glob("*.txt"):
            try:
                self.prompts[file_path.stem] = file_path.read_text(
                    encoding="utf-8"
                ).strip()
            except Exception as e:
                logger.error(f"Ошибка при чтении файла {file_path.name}: {e}")

        logger.info(f"Теущие промпты: {sorted(self.prompts)}")

    def get_prompt(self, name: str) -> str:
        """
        Возвращает текст конкретного промпта, объединяя его с общим форматом ответа.

        Метод ищет промпт по имени, а также пытается найти файл 'response_format'
        для добавления общих инструкций по форматированию в конец текста.

        Args:
            name (str): Имя файла промпта (без .txt).

        Returns:
            str: Сформированный текст промпта с инструкциями по форматированию.
        """
        prompt = self.prompts.get(name, "")
        format = self.prompts.get("response_format", "")
        return prompt + "\n\n" + format

    def update_folder(self, folder_name: str) -> None:
        """
        Полностью обновляет базу промптов, загружая файлы из новой директории.

        Args:
            folder_name (str): Путь к новой директории с промптами.
        """
        self.clear()
        self._load(folder_name)

    def clear(self) -> None:
        """
        Очищает текущий словарь загруженных промптов.
        """
        self.prompts.clear()

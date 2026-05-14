import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Callable, Dict, List, Any

from tools.base import TOOL_MARKER
import tools.implementations

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Автоматический реестр инструментов (Tools/Functions) для LLM-ассистента.

    Сканирует пакет `tools.implementations` и регистрирует все функции,
    помеченные декоратором `@tool`. Это позволяет добавлять новые инструменты
    простым созданием файла в папке `tools/implementations/` — без правки реестра.
    """

    def __init__(self, switch_model_callback: Callable) -> None:
        """
        Инициализирует реестр: устанавливает callback и сканирует инструменты.

        Args:
            switch_model_callback (Callable): Функция обратного вызова для переключения
                активной языковой модели (используется в Game Mode).
        """
        from tools.implementations import game_mode

        game_mode.set_switch_model_callback(switch_model_callback)

        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []
        self._discover()

    def _discover(self) -> None:
        """
        Автоматически обнаруживает все @tool-функции в пакете tools.implementations.
        """
        package_path = Path(tools.implementations.__file__).parent

        for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
            full_name = f"tools.implementations.{module_name}"
            try:
                module = importlib.import_module(full_name)
            except Exception as e:
                logger.warning(f"Не удалось импортировать модуль '{full_name}': {e}")
                continue

            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if getattr(obj, TOOL_MARKER, False):
                    self._tools[name] = obj
                    self._schemas.append(obj.openai_schema)
                    logger.debug(f"Зарегистрирован инструмент: {name}")

        logger.info(
            f"ToolRegistry: обнаружено {len(self._tools)} инструментов: "
            f"{list(self._tools.keys())}"
        )

    def get_tool_map(self) -> Dict[str, Callable]:
        """
        Возвращает словарь {имя_функции: callable} для вызова LLM.
        """
        return dict(self._tools)

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Возвращает список инструментов в формате OpenAI JSON Schema.
        """
        return list(self._schemas)

import webbrowser
import logging
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)

class Tools:
    def __init__(self, switch_model_callback: Callable):
        self.switch_model_callback = switch_model_callback

    def launch_cs2(self) -> str:
        """
        Launches Counter-Strike 2 via Steam. Use this when the user wants to play CS2 or Counter-Strike.
        """
        try:
            webbrowser.open("steam://rungameid/730")
            logger.debug("Инструмент выполнен: launch_cs2")
            return "Command sent to Steam to launch Counter-Strike 2."
        except Exception as e:
            logger.error(f"Ошибка запуска CS2: {e}")
            return f"Failed to launch CS2: {str(e)}"

    def enable_game_mode(self) -> str:
        """
        Switches the assistant to Game Mode. This changes the active LLM model to GEMINI. Use this when the user asks to enable game mode.
        """
        try:
            self.switch_model_callback()
            logger.debug("Инструмент выполнен: enable_game_mode")
            return "Successfully enabled Game Mode. The model is now switched to GEMINI."
        except Exception as e:
            logger.error(f"Ошибка переключения в Game Mode: {e}")
            return f"Failed to switch to Game Mode: {str(e)}"

    def get_tools_list(self) -> List[Callable]:
        return [self.launch_cs2, self.enable_game_mode]

    def get_tool_map(self) -> Dict[str, Callable]:
        return {func.__name__: func for func in self.get_tools_list()}
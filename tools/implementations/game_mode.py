import logging
from tools.base import tool

logger = logging.getLogger(__name__)

_switch_model_callback = None


def set_switch_model_callback(callback):
    """Устанавливает callback для переключения модели."""
    global _switch_model_callback
    _switch_model_callback = callback


@tool
def enable_game_mode() -> str:
    """
    Activates the specialized Game Mode and switches the underlying LLM to Gemini for better tactical analysis.
    Use this when the user wants to start a training session, enables 'game mode', or says 'включи игровой режим' / 'режим игры'.
    """
    try:
        if _switch_model_callback:
            _switch_model_callback()
        logger.info("Инструмент выполнен: enable_game_mode")
        return "Successfully enabled Game Mode. The model is now switched to GEMINI."
    except Exception as e:
        logger.error(f"Ошибка переключения в Game Mode: {e}")
        return f"Failed to switch to Game Mode: {str(e)}"

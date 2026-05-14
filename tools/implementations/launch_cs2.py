import webbrowser
import logging
from tools.base import tool

logger = logging.getLogger(__name__)


@tool
def launch_cs2() -> str:
    """
    Launches Counter-Strike 2.
    IMPORTANT: Before calling this, ALWAYS check system stats using 'get_system_stats'.
    If free VRAM is low (less than need to launch game), suggest 'enable_game_mode' to the user instead of launching.
    """
    try:
        webbrowser.open("steam://rungameid/730")
        logger.info("Инструмент выполнен: launch_cs2")
        return "Command sent to Steam to launch Counter-Strike 2."
    except Exception as e:
        logger.error(f"Ошибка запуска CS2: {e}")
        return f"Failed to launch CS2: {str(e)}"

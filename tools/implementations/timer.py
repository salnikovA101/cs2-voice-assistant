import threading
import logging
from tools.base import tool

logger = logging.getLogger(__name__)

_active_timers: dict[str, threading.Timer] = {}


def _timer_callback(name: str, seconds: int) -> None:
    """Callback при срабатывании таймера."""
    logger.info(f"Таймер '{name}' сработал! ({seconds} сек прошло)")
    _active_timers.pop(name, None)
    try:
        import winsound

        winsound.Beep(1000, 500)
    except Exception:
        pass


@tool
def set_timer(seconds: int, name: str) -> str:
    """
    Sets a countdown timer that will fire after the specified number of seconds.
    Use this when the user asks to set a timer, reminder, or alarm.
    Example triggers: 'поставь таймер на 5 минут', 'напомни через 30 секунд'.

    Args:
        seconds (int): Duration in seconds.
        name (str): A short label for this timer (e.g. 'break', 'tea', 'game').
    """
    try:
        if name in _active_timers:
            _active_timers[name].cancel()

        timer = threading.Timer(seconds, _timer_callback, args=(name, seconds))
        timer.daemon = True
        timer.start()
        _active_timers[name] = timer

        if seconds >= 60:
            minutes = seconds // 60
            remaining_secs = seconds % 60
            human_time = (
                f"{minutes} мин {remaining_secs} сек"
                if remaining_secs
                else f"{minutes} мин"
            )
        else:
            human_time = f"{seconds} сек"

        logger.info(f"Инструмент выполнен: set_timer, name='{name}', seconds={seconds}")
        return f"Timer '{name}' set for {human_time}."
    except Exception as e:
        logger.error(f"Ошибка set_timer: {e}")
        return f"Failed to set timer: {str(e)}"


@tool
def cancel_timer(name: str) -> str:
    """
    Cancels a previously set timer by its name.
    Use this when the user wants to cancel or stop a timer.

    Args:
        name (str): Name of the timer to cancel.
    """
    try:
        timer = _active_timers.pop(name, None)
        if timer:
            timer.cancel()
            logger.info(f"Инструмент выполнен: cancel_timer, name='{name}'")
            return f"Timer '{name}' cancelled."
        return f"No active timer named '{name}'."
    except Exception as e:
        logger.error(f"Ошибка cancel_timer: {e}")
        return f"Failed to cancel timer: {str(e)}"

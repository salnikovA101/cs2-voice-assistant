import webbrowser
import logging
from typing import Callable, Dict, List
import urllib.request
import urllib.parse
import re

logger = logging.getLogger(__name__)


class Tools:
    """
    Класс-регистратор инструментов (Tools/Functions) для LLM-ассистента.

    Содержит методы для взаимодействия с операционной системой, игровыми клиентами
    и веб-сервисами. Эти методы предназначены для вызова моделью через механизм
    Function Calling.
    """

    def __init__(self, switch_model_callback: Callable):
        """
        Инициализирует реестр инструментов.

        Args:
            switch_model_callback (Callable): Функция обратного вызова для переключения
                активной языковой модели (используется в Game Mode).
        """
        self.switch_model_callback = switch_model_callback

    def launch_cs2(self) -> str:
        """
        Launches Counter-Strike 2 (CS2) via Steam.
        Use this when the user expresses a desire to play, start, or open Counter-Strike, CS2, or "контра".
        Example triggers: "запусти кс", "давай играть в кс", "open cs2".
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
        Activates the specialized Game Mode and switches the underlying LLM to Gemini for better tactical analysis.
        Use this when the user wants to start a training session, enables 'game mode', or says 'включи игровой режим' / 'режим игры'.
        """
        try:
            self.switch_model_callback()
            logger.debug("Инструмент выполнен: enable_game_mode")
            return (
                "Successfully enabled Game Mode. The model is now switched to GEMINI."
            )
        except Exception as e:
            logger.error(f"Ошибка переключения в Game Mode: {e}")
            return f"Failed to switch to Game Mode: {str(e)}"

    def play_youtube_video(self, query: str) -> str:
        """
        Searches YouTube for a video and opens the first result.
        Use this when the user wants to watch a video, listen to music, or find gameplay.

        Args:
            query (str): The specific search term extracted from the user's request.
                        Exclude phrases like "открой на ютубе" or "найди видео".
                        Example: if user says "найди на ютубе музыку для катки", query should be "музыка для катки".
        """
        try:
            query_string = urllib.parse.urlencode({"search_query": query})
            url = f"https://www.youtube.com/results?{query_string}"
            html_content = urllib.request.urlopen(url)
            search_results = re.findall(
                r"watch\?v=(\S{11})", html_content.read().decode()
            )
            if search_results:
                video_url = f"https://www.youtube.com/watch?v={search_results[0]}"
                webbrowser.open(video_url)
                logger.debug(
                    f"Инструмент выполнен: play_youtube_video, запрос: '{query}'"
                )
                return f"Successfully opened the first YouTube video for: {query}"
            else:
                logger.warning(f"Видео не найдено по запросу: '{query}'")
                return f"No YouTube results found for: {query}"

        except Exception as e:
            logger.error(f"Ошибка поиска на YouTube: {e}")
            return f"Failed to play YouTube video: {str(e)}"

    def get_tools_list(self) -> List[Callable]:
        """
        Возвращает список всех доступных функций-инструментов.

        Returns:
            List[Callable]: Список методов класса Tools.
        """
        return [self.launch_cs2, self.enable_game_mode, self.play_youtube_video]

    def get_tool_map(self) -> Dict[str, Callable]:
        """
        Создает словарь соответствия имен функций их объектам.

        Используется для быстрого вызова нужного метода после того, как LLM
        вернет имя инструмента для выполнения.

        Returns:
            Dict[str, Callable]: Словарь вида {'название_функции': объект_функции}.
        """
        return {func.__name__: func for func in self.get_tools_list()}

import webbrowser
import logging
import yt_dlp
from tools.base import tool

logger = logging.getLogger(__name__)


@tool
def play_youtube_video(query: str) -> str:
    """
    Searches YouTube for a video and opens the first result in the browser.
    Use this when the user wants to watch a video, listen to music, or find gameplay.

    Args:
        query (str): The specific search term extracted from the user's request.
                    Exclude phrases like 'открой на ютубе' or 'найди видео'.
                    Example: if user says 'найди на ютубе музыку для катки', query should be 'музыка для катки'.
    """
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "playlist_items": "1",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)

        if not info or not info.get("entries"):
            logger.warning(f"Видео не найдено по запросу: '{query}'")
            return f"No YouTube results found for: {query}"

        entry = info["entries"][0]
        video_id = entry.get("id")
        title = entry.get("title", video_id)

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        webbrowser.open(video_url)

        logger.info(f"Инструмент выполнен: play_youtube_video, '{title}'")
        return f"Opened YouTube video: {title}"

    except Exception as e:
        logger.error(f"Ошибка поиска на YouTube: {e}")
        return f"Failed to play YouTube video: {str(e)}"

import asyncio
import logging

from core.assistant import Assistant
from utils.config import AppConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """
    Основная точка входа в приложение.

    Загружает конфигурацию, настраивает уровень логирования и запускает
    бесконечный цикл обработки голосовых команд.
    """
    config: AppConfig = AppConfig.load("config.yaml")
    if config.general.debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    assistant = Assistant(config)
    logger.info("Voice Assistant готов!")
    ptt_key = config.general.push_to_talk_key

    while True:
        logger.info(f"Удерживай '{ptt_key}' и говори")
        await assistant.run_pipeline()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bye")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")

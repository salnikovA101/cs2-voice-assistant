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
        quiet_loggers = [
            "httpx",
            "faster_whisper",
            "faster_qwen3_tts",
            "qwen_tts",
            "huggingface_hub"
        ]
        for log in quiet_loggers:
            logging.getLogger(log).setLevel(logging.ERROR)

    assistant = Assistant(config)
    logger.info("Voice Assistant готов!")
    ptt_key = config.general.push_to_talk_key

    try:
        while True:
            logger.info(f"Удерживай '{ptt_key}' и говори")
            try:
                await assistant.run_pipeline()
            except Exception as e:
                logger.error(f"Ошибка в pipeline: {e}", exc_info=True)
                await asyncio.sleep(1)
    finally:
        logger.info("Завершение работы...")
        await assistant.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bye")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)

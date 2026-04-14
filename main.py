import asyncio
import yaml
import os
import logging
from typing import Any, Dict
from dotenv import load_dotenv

from core.assistant import Assistant

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

async def main() -> None:
    with open("config.yaml", encoding="utf-8") as file:
        config: Dict[str, Any] = yaml.safe_load(file)
    
    config["llm"]["api_key"] = os.getenv("API_KEY")
    assistant = Assistant(config)
    
    logger.info("CS2 Voice Assistant готов!")
    ptt_key = config["general"]["push_to_talk_key"]

    while True:
        logger.info(f"Удерживай '{ptt_key}' и говори")
        await assistant.run_pipeline()
        await asyncio.sleep(0.01)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bye")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
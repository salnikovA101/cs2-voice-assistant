import asyncio
import yaml
import os
from typing import Any, Dict
from dotenv import load_dotenv

from core.assistant import Assistant

load_dotenv()

async def main() -> None:
    with open("config.yaml", encoding="utf-8") as file:
        config: Dict[str, Any] = yaml.safe_load(file)
    
    config["llm"]["api_key"] = os.getenv("API_KEY")
    assistant = Assistant(config)
    
    print("CS2 Voice Assistant готов!")
    ptt_key = config["general"]["push_to_talk_key"]

    while True:
        print(f"Удерживай '{ptt_key}' и говори")
        await assistant.run_pipeline()
        await asyncio.sleep(0.01)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bye")
    except Exception as e:
        print(f"Критическая ошибка: {e}")
import time
import logging
from typing import Any, Dict, List, Optional, Callable
from ollama import AsyncClient
from llm.base import BaseLLMProvider
from utils.config import LlmConfig

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Провайдер для взаимодействия с локальными моделями через Ollama API.

    Поддерживает мультимодальные запросы, механизм многократного вызова инструментов (Tool Calling loop).
    """

    def __init__(self, config: LlmConfig) -> None:
        """
        Инициализирует асинхронный клиент Ollama.

        Args:
            config (LlmConfig): Общий объект конфигурации LlmConfig, содержащий секцию .ollama.
        """
        self.config = config.ollama
        self.client = AsyncClient(host=self.config.host)
        logger.info(f"Ollama {self.config.model} готов")

    async def generate_response(
        self,
        user_text: str,
        image_bytes: Optional[bytes] = None,
        prompt: str = "",
        history: Optional[List[Any]] = None,
        tools: Optional[List[Callable]] = None,
        tool_map: Optional[Dict[str, Callable]] = None,
    ) -> str:
        """
        Генерирует ответ от локальной модели, обрабатывая вызовы инструментов при необходимости.

        Метод формирует стек сообщений, отправляет их в Ollama и, если модель
        решает вызвать функцию, исполняет её локально, возвращает результат модели
        и повторяет цикл до получения финального текстового ответа.

        Args:
            user_text (str): Текстовый запрос пользователя.
            image_bytes (Optional[bytes]): Данные изображения для мультимодальных моделей.
            prompt (str): Системный промпт.
            history (Optional[List[Any]]): История диалога.
            tools (Optional[List[Callable]]): Описания функций для модели.
            tool_map (Optional[Dict[str, Callable]]): Словарь с исполняемыми функциями.

        Returns:
            str: Финальный текстовый ответ модели.
        """
        try:
            start = time.perf_counter()
            messages: List[Any] = []
            if prompt:
                messages.append({"role": "system", "content": prompt})
            if history:
                messages.extend(history)
            user_msg: Dict[str, Any] = {"role": "user", "content": user_text}
            if image_bytes:
                user_msg["images"] = [image_bytes]
            messages.append(user_msg)
            response = await self.client.chat(
                model=self.config.model,
                messages=messages,
                think=self.config.think,
                tools=tools,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_output_tokens,
                    "num_ctx": self.config.num_ctx,
                },
            )
            max_turns = 5
            turns = 0
            logger.debug(response)
            while response.message.tool_calls and turns < max_turns:
                turns += 1
                logger.debug("Ollama использует инструменты...")
                messages.append(response.message)
                for call in response.message.tool_calls:
                    name = call.function.name
                    args = call.function.arguments
                    function_to_call = tool_map.get(name) if tool_map else None

                    if function_to_call:
                        logger.debug(f"Вызов функции {name}, аргументы: {args}")
                        result = function_to_call(**args)
                        messages.append({"role": "tool", "content": str(result)})
                    else:
                        logger.error(f"Ошибка функции {name} не существует")
                        messages.append(
                            {
                                "role": "tool",
                                "content": f"Error: function {name} not found.",
                            }
                        )
                response = await self.client.chat(
                    model=self.config.model,
                    messages=messages,
                    think=self.config.think,
                    tools=tools,
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_output_tokens,
                        "num_ctx": self.config.num_ctx,
                    },
                )
                logger.debug(response)
            text = response.message.content or "Ollama вернул пустой ответ."
            logger.debug(
                f"LLM: Ollama ответил за {time.perf_counter() - start:.2f} сек"
            )
            return text.strip()

        except Exception as e:
            logger.error(f"Ошибка Ollama: {e}")
            return f"Ошибка Ollama: {e}"

    async def unload(self) -> None:
        """
        Выгружает модель из памяти GPU/RAM, устанавливая keep_alive в 0.
        """
        try:
            await self.client.generate(model=self.config.model, prompt="", keep_alive=0)
            logger.debug(f"Ollama: модель {self.config.model} выгружена.")
        except Exception as e:
            logger.error(f"Ошибка выгрузки Ollama: {e}")

    async def warmup(self) -> None:
        """
        Предзагружает модель в память, устанавливая бесконечный keep_alive.
        """
        try:
            await self.client.generate(
                model=self.config.model, prompt="", keep_alive=-1
            )
            logger.debug(f"Ollama: модель {self.config.model} прогрета")
        except Exception as e:
            logger.error(f"Ошибка прогрева Ollama: {e}")

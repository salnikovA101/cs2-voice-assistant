import asyncio
import base64
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from openai import AsyncOpenAI
from utils.config import OpenAIProfile

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_INSTRUCT_TAG_RE = re.compile(
    r"<instruct>(.*?)</instruct>", flags=re.DOTALL | re.IGNORECASE
)


class BaseLLMProvider(ABC):
    """
    Базовый провайдер LLM на основе OpenAI-совместимого SDK.

    Содержит конкретную реализацию generate_response — общую для всех
    провайдеров (Ollama, LM Studio, OpenAI, Gemini и др.).

    Подклассы обязаны реализовать unload() и warmup().
    """

    def __init__(
        self, profile: OpenAIProfile, max_output_tokens: int, max_turns: int
    ) -> None:
        """
        Инициализирует базовый провайдер LLM.

        Args:
            profile (OpenAIProfile): Профиль настроек провайдера (URL, API ключ, модель).
            max_output_tokens (int): Лимит токенов на один ответ.
            max_turns (int): Максимальное количество итераций вызова инструментов.
        """
        self.profile = profile
        self.max_output_tokens = max_output_tokens
        self.max_turns = max_turns
        self.client = AsyncOpenAI(
            base_url=profile.base_url,
            api_key=profile.api_key or "api_key",
        )
        logger.info(
            f"[{self.__class__.__name__}] Инициализирован: "
            f"model={profile.model}, url={profile.base_url}"
        )

    async def generate_response(
        self,
        user_text: str,
        image_bytes: Optional[bytes] = None,
        prompt: str = "",
        history: Optional[List[Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_map: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Генерирует текстовый ответ на основе входных данных.

        Args:
            user_text: Текст запроса пользователя.
            image_bytes: Опциональное изображение.
            prompt: Системный промпт.
            history: История диалога в формате OpenAI messages.
            tools: Список инструментов в формате OpenAI tool schema.
            tool_map: Карта {имя_функции: callable} для вызова инструментов.

        Returns:
            Tuple[str, Optional[str]]: Кортеж из очищенного текста ответа и
                опциональной инструкции (эмоции) для TTS.
        """
        try:
            start = time.perf_counter()
            messages: List[Dict[str, Any]] = []

            if prompt:
                messages.append({"role": "system", "content": prompt})
            if history:
                messages.extend(history)

            content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
            if image_bytes:
                b64 = base64.b64encode(image_bytes).decode()
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )
            messages.append({"role": "user", "content": content})

            kwargs: Dict[str, Any] = {
                "model": self.profile.model,
                "messages": messages,
                "temperature": self.profile.temperature,
                "max_tokens": self.max_output_tokens,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = await self.client.chat.completions.create(**kwargs)
            logger.debug(response)
            message = response.choices[0].message

            turns = 0
            while message.tool_calls and turns < self.max_turns:
                turns += 1
                logger.debug(f"Tool loop turn {turns}/{self.max_turns}")
                messages.append(message.model_dump(exclude_none=True))

                for tc in message.tool_calls:
                    fn = tool_map.get(tc.function.name) if tool_map else None
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    if fn:
                        logger.debug(
                            f"Вызов инструмента '{tc.function.name}', args={args}"
                        )
                        result = await asyncio.to_thread(fn, **args)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": str(result),
                            }
                        )
                    else:
                        logger.error(
                            f"Инструмент '{tc.function.name}' не найден в tool_map"
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": f"Error: function '{tc.function.name}' not found.",
                            }
                        )

                kwargs["messages"] = messages
                response = await self.client.chat.completions.create(**kwargs)
                logger.debug(response)
                message = response.choices[0].message

            text = message.content or ""
            if not text and turns >= self.max_turns:
                logger.warning(
                    f"[{self.__class__.__name__}] max_turns исчерпан."
                    "Финальный запрос без tools."
                )
                fallback_messages = messages + [
                    {
                        "role": "user",
                        "content": "На основе найденной информации дай финальный текстовый ответ на вопрос пользователя.",
                    }
                ]
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("tools", None)
                fallback_kwargs.pop("tool_choice", None)
                fallback_kwargs["messages"] = fallback_messages
                fallback_response = await self.client.chat.completions.create(
                    **fallback_kwargs
                )
                logger.debug(fallback_response)
                text = fallback_response.choices[0].message.content or ""
            text = _THINK_TAG_RE.sub("", text).strip()

            instruct = None
            match = _INSTRUCT_TAG_RE.search(text)
            if match:
                instruct = match.group(1).strip()
                text = _INSTRUCT_TAG_RE.sub("", text).strip()

            text = re.sub(r"[*_#`~]", "", text)

            logger.debug(
                f"[{self.__class__.__name__}] Ответ за {time.perf_counter() - start:.2f}s"
            )
            return text.strip(), instruct

        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Ошибка generate_response: {e}")
            return f"Ошибка: {e}", None

    @abstractmethod
    async def unload(self) -> None:
        """Выгрузить модель из памяти (если поддерживается провайдером)."""

    @abstractmethod
    async def warmup(self) -> None:
        """Прогреть / загрузить модель (если поддерживается провайдером)."""

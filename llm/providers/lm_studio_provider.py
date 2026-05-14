import logging
import httpx

from llm.base import BaseLLMProvider
from utils.config import OpenAIProfile

logger = logging.getLogger(__name__)


class LMStudioProvider(BaseLLMProvider):
    """
    Lifecycle через /api/v1 REST API:
      - warmup: POST /api/v1/models/load → сохраняет instance_id
      - unload: POST /api/v1/models/unload с instance_id
    """

    def __init__(
        self, profile: OpenAIProfile, max_output_tokens: int, max_turns: int
    ) -> None:
        """Инициализирует провайдер LM Studio."""
        super().__init__(profile, max_output_tokens, max_turns)
        self._instance_id: str | None = None

    async def warmup(self) -> None:
        """
        Загружает модель в LM Studio через REST API.
        Модель будет загружена с параметрами из профиля (context_length).
        """
        url = f"{self.profile.host}/api/v1/models/load"
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    url,
                    json={
                        "model": self.profile.model,
                        "context_length": self.profile.context_length,
                        "echo_load_config": True,
                    },
                )
                if resp.status_code not in (200, 201):
                    logger.warning(
                        f"[LMStudioProvider] load → {resp.status_code}: {resp.text}"
                    )
                    return
                data = resp.json()
                self._instance_id = data.get("instance_id")

            logger.info(
                f"[LMStudioProvider] '{self.profile.model}' загружена "
                f"(instance_id={self._instance_id}, context_length={self.profile.context_length})"
            )
        except Exception as e:
            logger.warning(f"[LMStudioProvider] Не удалось загрузить модель: {e}")

    async def unload(self) -> None:
        """
        Выгружает модель из LM Studio, используя сохраненный instance_id.
        """
        try:
            if not self._instance_id:
                logger.debug(
                    f"[LMStudioProvider] Модель '{self.profile.model}' не была загружена ассистентом, "
                    "выгрузка пропущена"
                )
                return

            url = f"{self.profile.host}/api/v1/models/unload"
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json={"instance_id": self._instance_id})
                if resp.status_code not in (200, 204):
                    logger.warning(
                        f"[LMStudioProvider] unload → {resp.status_code}: {resp.text}"
                    )
                    return

            self._instance_id = None
            logger.info(f"[LMStudioProvider] '{self.profile.model}' выгружена")
        except Exception as e:
            logger.warning(f"[LMStudioProvider] Не удалось выгрузить модель: {e}")

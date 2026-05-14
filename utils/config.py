from urllib.parse import urlparse

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

from utils.constants import TTSModes, LLMProviderType


class OpenAIProfile(BaseModel):
    """Настройки профиля для OpenAI-совместимого провайдера."""

    provider: LLMProviderType = Field(
        default=LLMProviderType.OPENAI,
        description="Тип провайдера (openai, ollama, lm_studio)",
    )
    model: str = Field(default="", description="Название модели")
    base_url: str = Field(default="", description="URL адрес API")
    api_key: str = Field(default="", description="API ключ (если требуется)")
    temperature: float = Field(default=0.7, description="Температура генерации")
    context_length: Optional[int] = Field(
        default=None, description="Длина контекста (окно)"
    )

    @property
    def host(self) -> str:
        """Базовый адрес сервера (scheme + netloc), вычисляется из base_url."""
        parsed = urlparse(self.base_url)
        return f"{parsed.scheme}://{parsed.netloc}"


class LlmProfiles(BaseModel):
    gemini: OpenAIProfile = Field(default_factory=OpenAIProfile)
    ollama: OpenAIProfile = Field(default_factory=OpenAIProfile)
    lm_studio: OpenAIProfile = Field(default_factory=OpenAIProfile)
    other: OpenAIProfile = Field(default_factory=OpenAIProfile)


class LlmConfig(BaseModel):
    current_profile: str = "lm_studio"
    game_mode: str = "gemini"
    history_len: int = 6
    prompt_folder: str = "prompts"
    max_output_tokens: int = 4096
    max_turns: int = 5
    profiles: LlmProfiles = Field(default_factory=LlmProfiles)


class SttConfig(BaseModel):
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "int8_bfloat16"
    language: str = "ru"


class SpeedTtsConfig(BaseModel):
    silero_speaker: str = "baya"
    sample_rate: int = 24000
    language: str = "ru"
    speaker_type: str = "v5_ru"
    device: str = "cuda"
    speaker_name: str = "baya"


class CloneTtsConfig(BaseModel):
    model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    max_seq_len: int = 2048
    ref_voice: str = "voices/example.wav"
    ref_text: str = "Буду, получается, проходить сюжетку Пылью, Женщиной-пауком, ну или какими-нибудь другими супергероями. И, в общем-то, надеюсь, всё получится. Погнали!"
    language: str = "Russian"
    chunk_size: int = 4


class EmotionalTtsConfig(BaseModel):
    model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    max_seq_len: int = 2048
    language: str = "Russian"
    chunk_size: int = 4
    voice_id: str = "Aiden"


class TtsConfig(BaseModel):
    mode: TTSModes = TTSModes.CLONE
    speed: SpeedTtsConfig = Field(default_factory=SpeedTtsConfig)
    clone: CloneTtsConfig = Field(default_factory=CloneTtsConfig)
    emotional: EmotionalTtsConfig = Field(default_factory=EmotionalTtsConfig)


class GeneralConfig(BaseModel):
    push_to_talk_key: str = "right ctrl"
    image: bool = True
    debug_mode: bool = False
    enable_voice_output: bool = True
    enable_text_input: bool = True
    enable_voice_input: bool = True
    delay_between_questions: float = 1.0


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__"
    )
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)


def load_config() -> AppConfig:
    """
    Загружает конфигурацию из файла config.yaml и переменных окружения.
    Выполняет базовую валидацию наличия профилей и источников ввода.

    Returns:
        AppConfig: Объект конфигурации.
    """
    import yaml
    import logging

    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.getLogger(__name__).warning(
            "config.yaml не найден, используются значения по умолчанию"
        )
        data = {}

    config = AppConfig(**data)

    if not config.general.enable_text_input and not config.general.enable_voice_input:
        raise ValueError(
            "Оба источника ввода (текстовый и голосовой) выключены. Работа невозможна."
        )

    if not config.llm.profiles:
        raise ValueError(
            "В конфигурации не задано ни одного LLM профиля (llm.profiles)."
        )

    return config


config = load_config()

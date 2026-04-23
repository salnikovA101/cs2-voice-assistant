import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from google.genai.types import ThinkingConfig, ThinkingLevel

from utils.constants import TTSModes, LLMModelNames


class GeneralConfig(BaseModel):
    """
    Общие настройки приложения.

    Attributes:
        push_to_talk_key (str): Клавиша активации записи микрофона (PTT).
        debug_mode (bool): Флаг включения расширенного логирования для отладки.
    """

    model_config = ConfigDict(frozen=True)
    push_to_talk_key: str = "right ctrl"
    debug_mode: bool = False


class SttConfig(BaseModel):
    """
    Настройки модуля распознавания речи (Speech-to-Text).

    Attributes:
        model (str): Название используемой модели Faster Whisper.
        device (str): Устройство для вычислений ('cuda' или 'cpu').
        compute_type (str): Тип точности вычислений (например, 'int8_bfloat16').
        language (str): Код языка для распознавания по умолчанию.
    """

    model_config = ConfigDict(frozen=True)
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "int8_bfloat16"
    language: str = "ru"


class QualityTtsConfig(BaseModel):
    """
    Настройки высококачественного синтеза речи (Quality TTS).

    Attributes:
        model (str): Путь или идентификатор модели Qwen3-TTS.
        device (str): Устройство для инференса.
        attn_implementation (str): Реализация механизма внимания (например, 'sdpa').
        max_seq_len (int): Максимальная длина последовательности токенов.
        ref_voice (str): Путь к эталонному аудиофайлу для клонирования голоса.
        ref_text (str): Текст, произносимый в эталонном аудиофайле.
        language (str): Язык синтеза.
        chunk_size (int): Размер фрагмента при потоковой генерации.
    """

    model_config = ConfigDict(frozen=True)
    model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    max_seq_len: int = 1024
    ref_voice: str = "voices/example.wav"
    ref_text: str = ""
    language: str = "Russian"
    chunk_size: int = 6


class SpeedTtsConfig(BaseModel):
    """
    Настройки быстрого синтеза речи (Speed TTS) на базе Silero.

    Attributes:
        silero_speaker (str): Имя спикера для генерации.
        sample_rate (int): Частота дискретизации выходного аудио.
        language (str): Языковой пакет модели.
        speaker_type (str): Версия / тип модели Silero.
        device (str): Устройство для вычислений.
        speaker_name (str): Псевдоним или конкретное имя голоса.
    """

    model_config = ConfigDict(frozen=True)
    silero_speaker: str = "baya"
    sample_rate: int = 24000
    language: str = "ru"
    speaker_type: str = "v5_ru"
    device: str = "cuda"
    speaker_name: str = "baya"


class GeminiConfig(BaseModel):
    """
    Параметры интеграции с Google Gemini API.

    Attributes:
        model (str): Идентификатор используемой модели.
        api_key (Optional[str]): Ключ доступа к Google AI Studio.
        temperature (float): Степень случайности ответов.
        max_output_tokens (int): Лимит токенов на один ответ.
        thinking_config (ThinkingConfig): Настройки режима 'рассуждения' (Thinking).
    """

    model_config = ConfigDict(frozen=True)
    model: str = "gemma-4-26b-a4b-it"
    api_key: Optional[str] = None
    temperature: float = 1.0
    max_output_tokens: int = 4000
    thinking_config: ThinkingConfig = Field(
        default_factory=lambda: ThinkingConfig(thinking_level=ThinkingLevel.HIGH)
    )


class OllamaConfig(BaseModel):
    """
    Настройки для локального запуска моделей через Ollama.

    Attributes:
        model (str): Название локальной модели.
        host (str): Адрес API локального сервера Ollama.
        think (bool): Флаг включения процесса размышления модели.
        temperature (float): Степень случайности ответов.
        max_output_tokens (int): Лимит токенов на один ответ.
        num_ctx (int): Размер контекстного окна.
    """

    model_config = ConfigDict(frozen=True)
    model: str = "qwen3.5:0.8b"
    host: str = "http://127.0.0.1:11434"
    think: bool = True
    temperature: float = 0.6
    max_output_tokens: int = 4000
    num_ctx: int = 8192


class TtsConfig(BaseModel):
    """
    Группирующая конфигурация для модулей TTS.

    Attributes:
        mode (TTSModes): Текущий выбранный режим (SPEED или QUALITY).
        speed (SpeedTtsConfig): Настройки быстрого режима.
        quality (QualityTtsConfig): Настройки качественного режима.
    """

    mode: TTSModes = TTSModes.SPEED
    speed: SpeedTtsConfig = Field(default_factory=SpeedTtsConfig)
    quality: QualityTtsConfig = Field(default_factory=QualityTtsConfig)


class LlmConfig(BaseModel):
    """
    Конфигурация логики текстового ИИ (LLM).

    Attributes:
        current_model (LLMModelNames): Выбранный провайдер модели (Gemini или Ollama).
        history_len (int): Количество последних сообщений, хранимых в памяти диалога.
        prompt_mode (str): Выбранный режим системного промпта.
        prompt_folder (str): Путь к папке с текстовыми файлами промптов.
        gemini (GeminiConfig): Настройки для Gemini.
        ollama (OllamaConfig): Настройки для Ollama.
    """

    current_model: LLMModelNames = LLMModelNames.GEMINI
    history_len: int = 3
    prompt_mode: str = "humor"
    prompt_folder: str = "prompts"
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)


class AppConfig(BaseSettings):
    """
    Корневой класс конфигурации всего приложения.

    Использует pydantic-settings для автоматической загрузки параметров
    из YAML-файла и переменных окружения (.env).
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__"
    )
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)

    @classmethod
    def load(cls, yaml_path: str | Path = "config.yaml") -> "AppConfig":
        """
        Загружает конфигурацию из YAML-файла с приоритетом над значениями по умолчанию.

        Args:
            yaml_path (str | Path): Путь к файлу config.yaml. По умолчанию 'config.yaml'.

        Returns:
            AppConfig: Полностью инициализированный объект конфигурации.
        """
        yaml_data: Dict[str, Any] = {}
        if Path(yaml_path).exists():
            with open(yaml_path, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
        return cls(**yaml_data)

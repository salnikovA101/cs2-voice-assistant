import yaml
from pathlib import Path
from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from google.genai.types import ThinkingConfig, ThinkingLevel

from utils.constants import TTSModes, LLMModelNames

class GeneralConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    push_to_talk_key: str = "right ctrl"
    debug_mode: bool = False

class SttConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "int8_bfloat16"
    language: str = "ru"

class QualityTtsConfig(BaseModel):
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
    model_config = ConfigDict(frozen=True)
    silero_speaker: str = "baya"
    sample_rate: int = 24000
    language: str = "ru"
    speaker_type: str = "v5_ru"
    device: str = "cuda"
    speaker_name: str = "baya"

class GeminiConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str = "gemma-4-26b-a4b-it"
    api_key: Optional[str] = None
    temperature: float = 1.0
    max_output_tokens: int = 4000
    thinking_config: ThinkingConfig = Field(
        default_factory=lambda: ThinkingConfig(thinking_level=ThinkingLevel.HIGH)
    )

class OllamaConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str = "qwen3.5:0.8b" 
    host: str = "http://127.0.0.1:11434"
    think: bool = True
    temperature: float = 0.6
    max_output_tokens: int = 4000
    num_ctx: int = 8192

class TtsConfig(BaseModel):
    mode: TTSModes = TTSModes.SPEED
    speed: SpeedTtsConfig = Field(default_factory=SpeedTtsConfig)
    quality: QualityTtsConfig = Field(default_factory=QualityTtsConfig)

class LlmConfig(BaseModel):
    current_model: LLMModelNames = LLMModelNames.GEMINI
    history_len: int = 3
    prompt_mode: str = "humor"
    prompt_folder: str = "prompts"
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__"
    )
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)

    @classmethod
    def load(cls, yaml_path: str | Path = "config.yaml") -> "AppConfig":
        yaml_data = {}
        if Path(yaml_path).exists():
            with open(yaml_path, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
        return cls(**yaml_data)

if __name__ == "__main__":
    config = AppConfig.load()
    print(f"Загружен API Key: {config.llm.gemini.api_key}")
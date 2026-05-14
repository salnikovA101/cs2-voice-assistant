from enum import StrEnum


class LLMProviderType(StrEnum):
    """
    Тип провайдера для LLM.
    """

    OPENAI = "openai"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"


class TTSModes(StrEnum):
    """
    Режимы работы системы синтеза речи (TTS).
    """

    SPEED = "speed"
    CLONE = "clone"
    EMOTIONAL = "emotional"

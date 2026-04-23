from enum import StrEnum


class LLMModelNames(StrEnum):
    """
    Перечисление доступных провайдеров языковых моделей (LLM).
    """

    GEMINI = "gemini"
    OLLAMA = "ollama"


class TTSModes(StrEnum):
    """
    Режимы работы системы синтеза речи (TTS).
    """

    SPEED = "speed"
    QUALITY = "quality"

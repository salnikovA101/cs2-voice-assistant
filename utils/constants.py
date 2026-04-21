from enum import StrEnum

class LLMModelNames(StrEnum):
    GEMINI = "gemini"
    OLLAMA = "ollama"

class TTSModes(StrEnum):
    SPEED = "speed"
    QUALITY = "quality"
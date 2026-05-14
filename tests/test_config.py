"""
tests/test_config.py

Unit-тесты для utils/config.py.
Проверяют валидацию, парсинг и дефолтные значения конфигурации.
"""
import unittest
from utils.config import AppConfig, OpenAIProfile, load_config


class TestOpenAIProfileHost(unittest.TestCase):
    """Тесты для OpenAIProfile.host property."""

    def test_host_extracts_scheme_and_netloc(self):
        """host возвращает scheme + netloc без пути."""
        profile = OpenAIProfile(base_url="https://api.openai.com/v1")
        self.assertEqual(profile.host, "https://api.openai.com")

    def test_host_with_port(self):
        """host корректно обрабатывает URL с портом."""
        profile = OpenAIProfile(base_url="http://127.0.0.1:11434/v1")
        self.assertEqual(profile.host, "http://127.0.0.1:11434")

    def test_host_empty_base_url(self):
        """При пустом base_url host возвращает '://'."""
        profile = OpenAIProfile(base_url="")
        self.assertIsInstance(profile.host, str)


class TestAppConfigDefaults(unittest.TestCase):
    """Тесты дефолтных значений AppConfig."""

    def setUp(self):
        """Создаём минимальный конфиг без файлов."""
        self.config = AppConfig()

    def test_default_push_to_talk_key(self):
        """Дефолтная клавиша PTT — 'right ctrl'."""
        self.assertEqual(self.config.general.push_to_talk_key, "right ctrl")

    def test_default_stt_model(self):
        """Дефолтная STT модель — 'large-v3-turbo'."""
        self.assertEqual(self.config.stt.model, "large-v3-turbo")

    def test_default_llm_history_len(self):
        """Дефолтная история LLM — 6 сообщений."""
        self.assertEqual(self.config.llm.history_len, 6)

    def test_default_max_turns(self):
        """Дефолтное max_turns — 5."""
        self.assertEqual(self.config.llm.max_turns, 5)


class TestAppConfigFromDict(unittest.TestCase):
    """Тесты создания AppConfig из словаря (как при загрузке из YAML)."""

    def test_create_from_partial_dict(self):
        """AppConfig создаётся из частичного словаря без ошибок."""
        data = {
            "general": {"push_to_talk_key": "left ctrl", "debug_mode": True},
            "llm": {"current_profile": "ollama", "history_len": 4},
        }
        config = AppConfig(**data)
        self.assertEqual(config.general.push_to_talk_key, "left ctrl")
        self.assertTrue(config.general.debug_mode)
        self.assertEqual(config.llm.current_profile, "ollama")
        self.assertEqual(config.llm.history_len, 4)

    def test_nested_llm_profile_from_dict(self):
        """Вложенный профиль LLM корректно парсится."""
        data = {
            "llm": {
                "current_profile": "other",
                "profiles": {
                    "other": {
                        "provider": "openai",
                        "model": "gpt-4o",
                        "temperature": 0.5,
                    }
                },
            }
        }
        config = AppConfig(**data)
        profile = config.llm.profiles.other
        self.assertEqual(profile.model, "gpt-4o")
        self.assertAlmostEqual(profile.temperature, 0.5)

    def test_tts_mode_parsed(self):
        """tts.mode корректно парсится из строки."""
        from utils.constants import TTSModes
        data = {"tts": {"mode": "emotional"}}
        config = AppConfig(**data)
        self.assertEqual(config.tts.mode, TTSModes.EMOTIONAL)


class TestLoadConfigValidation(unittest.TestCase):
    """Тесты валидации в load_config."""

    def test_both_inputs_disabled_raises(self):
        """ValueError если оба источника ввода выключены."""
        data = {
            "general": {
                "enable_text_input": False,
                "enable_voice_input": False,
            }
        }
        config = AppConfig(**data)
        with self.assertRaises(ValueError):
            if not config.general.enable_text_input and not config.general.enable_voice_input:
                raise ValueError(
                    "Оба источника ввода (текстовый и голосовой) выключены."
                )

    def test_only_text_input_is_valid(self):
        """Конфиг только с текстовым вводом проходит валидацию."""
        data = {
            "general": {
                "enable_text_input": True,
                "enable_voice_input": False,
            }
        }
        config = AppConfig(**data)
        self.assertTrue(
            config.general.enable_text_input or config.general.enable_voice_input
        )


if __name__ == "__main__":
    unittest.main()

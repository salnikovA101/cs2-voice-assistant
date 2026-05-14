"""
tests/test_tool_registry.py

Unit-тесты для ToolRegistry из tools/registry.py.
Проверяют автоматическое обнаружение и регистрацию инструментов.
"""
import unittest
from unittest.mock import MagicMock
from tools.registry import ToolRegistry


class TestToolRegistry(unittest.TestCase):
    """Тесты для ToolRegistry."""

    def setUp(self):
        """Создаём ToolRegistry с mock-callback перед каждым тестом."""
        self.callback = MagicMock()
        self.registry = ToolRegistry(switch_model_callback=self.callback)

    def test_discovers_expected_tools(self):
        """Реестр обнаруживает все ожидаемые инструменты."""
        expected = {
            "enable_game_mode",
            "launch_cs2",
            "get_system_stats",
            "cancel_timer",
            "set_timer",
            "web_search",
            "play_youtube_video",
        }
        tool_map = self.registry.get_tool_map()
        self.assertEqual(set(tool_map.keys()), expected)

    def test_tool_count(self):
        """Обнаружено ровно 7 инструментов."""
        self.assertEqual(len(self.registry.get_tool_map()), 7)

    def test_get_tool_map_returns_callables(self):
        """get_tool_map возвращает только вызываемые объекты."""
        tool_map = self.registry.get_tool_map()
        for name, fn in tool_map.items():
            with self.subTest(tool=name):
                self.assertTrue(callable(fn), f"'{name}' не является callable")

    def test_get_tool_map_returns_copy(self):
        """get_tool_map возвращает копию — мутации не влияют на реестр."""
        tool_map = self.registry.get_tool_map()
        tool_map["fake_tool"] = MagicMock()
        self.assertNotIn("fake_tool", self.registry.get_tool_map())

    def test_openai_tools_count_matches_tool_map(self):
        """Количество OpenAI schemas совпадает с количеством инструментов."""
        self.assertEqual(
            len(self.registry.get_openai_tools()),
            len(self.registry.get_tool_map()),
        )

    def test_openai_tools_schema_structure(self):
        """Каждая schema содержит ключи 'type' и 'function'."""
        for schema in self.registry.get_openai_tools():
            with self.subTest(schema=schema.get("function", {}).get("name")):
                self.assertIn("type", schema)
                self.assertIn("function", schema)
                self.assertEqual(schema["type"], "function")

    def test_openai_tools_function_has_name(self):
        """Каждая schema содержит непустое поле 'name'."""
        for schema in self.registry.get_openai_tools():
            name = schema["function"].get("name", "")
            with self.subTest(schema_name=name):
                self.assertTrue(name, "Поле 'name' в schema пустое")

    def test_openai_tools_function_has_description(self):
        """Каждая schema содержит непустое поле 'description'."""
        for schema in self.registry.get_openai_tools():
            desc = schema["function"].get("description", "")
            name = schema["function"].get("name", "?")
            with self.subTest(tool=name):
                self.assertTrue(desc, f"Инструмент '{name}' не имеет description")

    def test_openai_tools_returns_copy(self):
        """get_openai_tools возвращает копию списка."""
        schemas = self.registry.get_openai_tools()
        schemas.append({"fake": True})
        self.assertNotIn({"fake": True}, self.registry.get_openai_tools())

    def test_callback_injected_without_call(self):
        """Callback передаётся в game_mode, но не вызывается при инициализации."""
        self.callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()

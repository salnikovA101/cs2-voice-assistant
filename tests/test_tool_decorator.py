"""
tests/test_tool_decorator.py

Unit-тесты для @tool декоратора из tools/base.py.
Проверяют корректность генерации OpenAI JSON Schema.
"""
import unittest
from tools.base import tool, TOOL_MARKER


class TestToolDecorator(unittest.TestCase):
    """Тесты декоратора @tool."""

    def _make_tool(self, func):
        """Вспомогательный метод: применяет @tool к переданной функции."""
        return tool(func)

    def test_schema_top_level_keys(self):
        """Schema содержит ключи 'type' и 'function'."""
        @tool
        def my_func(x: str) -> str:
            """My description."""
            return x

        schema = my_func.openai_schema
        self.assertIn("type", schema)
        self.assertIn("function", schema)
        self.assertEqual(schema["type"], "function")

    def test_schema_function_name(self):
        """Поле 'name' в schema совпадает с именем функции."""
        @tool
        def launch_rocket(target: str) -> str:
            """Launch a rocket."""
            return target

        self.assertEqual(my_func_name := launch_rocket.openai_schema["function"]["name"],
                         "launch_rocket")

    def test_schema_description_is_first_docstring_line(self):
        """Description берётся из первой строки docstring."""
        @tool
        def do_something(x: int) -> None:
            """First line of doc.

            Second line should be ignored.
            """

        desc = do_something.openai_schema["function"]["description"]
        self.assertEqual(desc, "First line of doc.")

    def test_schema_no_docstring(self):
        """Если docstring отсутствует, description — 'No description provided.'"""
        @tool
        def no_doc(x: str) -> str:
            return x

        desc = no_doc.openai_schema["function"]["description"]
        self.assertEqual(desc, "No description provided.")

    def test_type_mapping_str(self):
        """str маппится в 'string'."""
        @tool
        def f(name: str) -> str:
            """Test."""
            return name

        props = f.openai_schema["function"]["parameters"]["properties"]
        self.assertEqual(props["name"]["type"], "string")

    def test_type_mapping_int(self):
        """int маппится в 'integer'."""
        @tool
        def f(count: int) -> int:
            """Test."""
            return count

        props = f.openai_schema["function"]["parameters"]["properties"]
        self.assertEqual(props["count"]["type"], "integer")

    def test_type_mapping_bool(self):
        """bool маппится в 'boolean'."""
        @tool
        def f(flag: bool) -> bool:
            """Test."""
            return flag

        props = f.openai_schema["function"]["parameters"]["properties"]
        self.assertEqual(props["flag"]["type"], "boolean")

    def test_type_mapping_float(self):
        """float маппится в 'number'."""
        @tool
        def f(ratio: float) -> float:
            """Test."""
            return ratio

        props = f.openai_schema["function"]["parameters"]["properties"]
        self.assertEqual(props["ratio"]["type"], "number")

    def test_type_mapping_unknown_defaults_to_string(self):
        """Неизвестный тип аннотации маппится в 'string'."""
        class MyCustomType:
            pass

        @tool
        def f(val: MyCustomType) -> None:  # type: ignore[valid-type]
            """Test."""

        props = f.openai_schema["function"]["parameters"]["properties"]
        self.assertEqual(props["val"]["type"], "string")

    def test_required_param_without_default(self):
        """Параметр без default попадает в required."""
        @tool
        def f(x: str) -> str:
            """Test."""
            return x

        required = f.openai_schema["function"]["parameters"]["required"]
        self.assertIn("x", required)

    def test_optional_param_with_default_not_required(self):
        """Параметр с default значением не попадает в required."""
        @tool
        def f(x: str, limit: int = 10) -> str:
            """Test."""
            return x

        required = f.openai_schema["function"]["parameters"]["required"]
        self.assertIn("x", required)
        self.assertNotIn("limit", required)

    def test_all_params_required_when_no_defaults(self):
        """Все параметры без default попадают в required."""
        @tool
        def f(a: str, b: int, c: bool) -> str:
            """Test."""
            return a

        required = f.openai_schema["function"]["parameters"]["required"]
        self.assertCountEqual(required, ["a", "b", "c"])

    def test_tool_marker_is_set(self):
        """@tool выставляет атрибут TOOL_MARKER = True."""
        @tool
        def f(x: str) -> str:
            """Test."""
            return x

        self.assertTrue(getattr(f, TOOL_MARKER, False))

    def test_function_remains_callable(self):
        """Декорированная функция остаётся вызываемой и возвращает правильный результат."""
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        self.assertEqual(add(2, 3), 5)

    def test_function_without_params(self):
        """Функция без параметров имеет пустые properties и required."""
        @tool
        def no_params() -> str:
            """No params."""
            return "ok"

        params = no_params.openai_schema["function"]["parameters"]
        self.assertEqual(params["properties"], {})
        self.assertEqual(params["required"], [])


if __name__ == "__main__":
    unittest.main()

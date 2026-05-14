"""
tests/test_llm_base.py

Unit-тесты для регулярных выражений и вспомогательной логики llm/base.py.
Не требуют GPU, сети или инициализации LLM-клиента.
"""
import re
import unittest

from llm.base import _THINK_TAG_RE, _INSTRUCT_TAG_RE


class TestThinkTagRegex(unittest.TestCase):
    """Тесты удаления <think>...</think> тегов."""

    def test_removes_single_think_tag(self):
        """Одиночный <think> блок удаляется полностью."""
        text = "<think>внутренние размышления</think>Ответ пользователю."
        result = _THINK_TAG_RE.sub("", text).strip()
        self.assertEqual(result, "Ответ пользователю.")

    def test_removes_multiline_think_tag(self):
        """Многострочный <think> блок удаляется."""
        text = "<think>\nстрока 1\nстрока 2\n</think>Итоговый ответ."
        result = _THINK_TAG_RE.sub("", text).strip()
        self.assertEqual(result, "Итоговый ответ.")

    def test_removes_multiple_think_tags(self):
        """Несколько <think> блоков удаляются."""
        text = "<think>раз</think>Текст<think>два</think> продолжение."
        result = _THINK_TAG_RE.sub("", text).strip()
        self.assertEqual(result, "Текст продолжение.")

    def test_no_think_tag_unchanged(self):
        """Текст без <think> тегов не меняется."""
        text = "Обычный ответ без тегов."
        result = _THINK_TAG_RE.sub("", text).strip()
        self.assertEqual(result, text)

    def test_empty_think_tag(self):
        """Пустой <think></think> удаляется."""
        text = "<think></think>Ответ."
        result = _THINK_TAG_RE.sub("", text).strip()
        self.assertEqual(result, "Ответ.")


class TestInstructTagRegex(unittest.TestCase):
    """Тесты извлечения <instruct>...</instruct> тегов."""

    def test_extracts_instruct_content(self):
        """Содержимое <instruct> корректно извлекается."""
        text = "<instruct>excited</instruct>Привет!"
        match = _INSTRUCT_TAG_RE.search(text)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1).strip(), "excited")

    def test_removes_instruct_tag_from_text(self):
        """После извлечения <instruct> тег удаляется из текста."""
        text = "<instruct>happy</instruct>Сегодня отличный день!"
        cleaned = _INSTRUCT_TAG_RE.sub("", text).strip()
        self.assertEqual(cleaned, "Сегодня отличный день!")

    def test_case_insensitive(self):
        """Regex нечувствителен к регистру тега."""
        text = "<INSTRUCT>sad</INSTRUCT>Текст."
        match = _INSTRUCT_TAG_RE.search(text)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1).strip(), "sad")

    def test_no_instruct_tag_returns_none(self):
        """При отсутствии тега match возвращает None."""
        text = "Обычный текст без instruct."
        match = _INSTRUCT_TAG_RE.search(text)
        self.assertIsNone(match)

    def test_instruct_with_whitespace(self):
        """Пробелы внутри тега корректно обрезаются через .strip()."""
        text = "<instruct>  curious  </instruct>Текст."
        match = _INSTRUCT_TAG_RE.search(text)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1).strip(), "curious")

    def test_multiline_instruct(self):
        """Многострочное содержимое <instruct> поддерживается (DOTALL)."""
        text = "<instruct>\nexcited\n</instruct>Ответ."
        match = _INSTRUCT_TAG_RE.search(text)
        self.assertIsNotNone(match)
        self.assertIn("excited", match.group(1))


class TestMarkdownStripping(unittest.TestCase):
    """Тесты очистки Markdown-символов из ответа LLM."""

    _MARKDOWN_RE = re.compile(r"[*_#`~]")

    def _strip_md(self, text: str) -> str:
        return self._MARKDOWN_RE.sub("", text)

    def test_removes_asterisks(self):
        """Звёздочки (bold/italic) удаляются."""
        self.assertEqual(self._strip_md("**жирный** текст"), "жирный текст")

    def test_removes_hashes(self):
        """Решётки (заголовки) удаляются."""
        self.assertEqual(self._strip_md("## Заголовок"), " Заголовок")

    def test_removes_backticks(self):
        """Обратные кавычки (код) удаляются."""
        self.assertEqual(self._strip_md("`код`"), "код")

    def test_removes_underscores(self):
        """Подчёркивания удаляются."""
        self.assertEqual(self._strip_md("_курсив_"), "курсив")

    def test_plain_text_unchanged(self):
        """Обычный текст без Markdown не изменяется."""
        text = "Просто текст без форматирования."
        self.assertEqual(self._strip_md(text), text)


if __name__ == "__main__":
    unittest.main()

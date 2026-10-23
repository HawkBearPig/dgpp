"""Template reference generation uses restricted Jinja capabilities; no model needed."""
import json
from pathlib import Path
import sys
import unittest

try:
    import jinja2
except ModuleNotFoundError as error:
    if error.name != "jinja2":
        raise
    print("SKIP: chat-template generator tests require Jinja2 from requirements-tools.txt")
    raise SystemExit(2)

from jinja2.sandbox import SecurityError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import gen_chat_template_goldens as generator


class ChatTemplateGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.env = generator.env()

    def test_private_attribute_traversal_is_rejected(self):
        with self.assertRaises(SecurityError):
            self.env.from_string("{{ ''.__class__.__name__ }}").render()

    def test_helper_globals_are_not_available(self):
        self.assertEqual(
            self.env.from_string("{{ raise_exception.__globals__ is defined }}").render(),
            "False",
        )

    def test_input_collections_cannot_be_mutated(self):
        messages = [{"role": "user", "content": "hello"}]
        with self.assertRaises(SecurityError):
            self.env.from_string("{{ messages.clear() }}").render(messages=messages)
        self.assertEqual(messages, [{"role": "user", "content": "hello"}])

    def test_message_template_syntax_remains_literal(self):
        text = "{{ 1 + 1 }} {% include 'unused' %}"
        self.assertEqual(self.env.from_string("{{ message }}").render(message=text), text)

    def test_namespace_loop_controls_and_whitespace(self):
        source = (
            "{% set ns = namespace(total=0) %}\n"
            "{% macro show(n) %}[{{ n }}]{% endmacro %}\n"
            "  {% for n in [1, 2, 3] %}\n"
            "{% if n == 3 %}{% break %}{% endif %}\n"
            "{% set ns.total = ns.total + n %}\n"
            "{% endfor %}\n"
            "{{ show(ns.total) }}"
        )
        self.assertEqual(self.env.from_string(source).render(), "[3]")

    def test_tojson_keeps_unicode_and_explicit_ascii_option(self):
        value = {"text": "城市 <>&'", "enabled": True}
        self.assertEqual(
            self.env.from_string("{{ value|tojson }}").render(value=value),
            json.dumps(value, ensure_ascii=False),
        )
        self.assertEqual(
            self.env.from_string("{{ value|tojson(ensure_ascii=True) }}").render(value=value),
            json.dumps(value, ensure_ascii=True),
        )

    def test_template_validation_error_is_preserved(self):
        with self.assertRaisesRegex(jinja2.TemplateError, "unsupported role"):
            self.env.from_string("{{ raise_exception('unsupported role') }}").render()


if __name__ == "__main__":
    unittest.main()

"""Template reference generation uses restricted Jinja capabilities; no model needed."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

try:
    import jinja2
except ModuleNotFoundError as error:
    if error.name != "jinja2":
        raise
    print("SKIP: chat-template generator tests require Jinja2 from requirements-tools.txt")
    raise SystemExit(2)

if tuple(int(x) for x in jinja2.__version__.split(".")[:3]) < (3, 1, 6):
    # The sandbox's mutable-method list grew with 3.1.6 (list.clear among
    # them), and requirements-tools.txt pins that floor; an older system
    # Jinja2 would fail the mutation case for the wrong reason.
    print("SKIP: chat-template generator tests need Jinja2 >= 3.1.6 from requirements-tools.txt, found "
          + jinja2.__version__)
    sys.exit(2)
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

    def test_main_records_runtime_version_and_golden_data(self):
        # Keep the metadata/output regression independent of the optional
        # tokenizer package and checkpoint downloads.
        tokenizers = Mock()
        tokenizer = tokenizers.Tokenizer.from_file.return_value
        tokenizer.encode.return_value.ids = [42, 43]
        content = "城市 {{ 1 + 1 }}"
        kwargs = {"messages": [{"role": "user", "content": content}]}
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "chat_template.jinja"
            tokenizer_json = Path(directory) / "tokenizer.json"
            output = Path(directory) / "goldens.jsonl"
            template.write_text("{{ messages[0].content }}", encoding="utf-8")
            tokenizer_json.write_text("{}", encoding="utf-8")
            argv = ["gen_chat_template_goldens.py", "--model", "Qwen/fixture",
                    "--template", str(template), "--tokenizer-json", str(tokenizer_json),
                    "--out", str(output)]
            with patch.object(sys, "argv", argv), \
                    patch.dict(sys.modules, {"tokenizers": tokenizers}), \
                    patch.object(generator, "QWEN_CASES", [("literal_user", kwargs)]), \
                    redirect_stdout(io.StringIO()):
                generator.main()
            header, case = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(header["jinja2"], jinja2.__version__)
            self.assertEqual(header["model"], "Qwen/fixture")
            self.assertEqual(header["cases"], 1)
            self.assertEqual(header["template_hash"], f"{generator.fnv1a64(template.read_bytes()):016x}")
            self.assertEqual(header["tokenizer_revision"], f"{generator.fnv1a64(tokenizer_json.read_bytes()):016x}")
            self.assertEqual(case, {"name": "literal_user", "kwargs": kwargs,
                                    "render": content, "ids": [42, 43]})
            tokenizers.Tokenizer.from_file.assert_called_once_with(str(tokenizer_json))
            tokenizer.encode.assert_called_once_with(content)


if __name__ == "__main__":
    unittest.main()

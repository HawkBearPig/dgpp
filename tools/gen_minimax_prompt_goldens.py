#!/usr/bin/env python3
"""Pin MiniMax's shipped Jinja rendering and HF tokenization, including recursive XML.

Run with the optional reference dependencies from requirements-tools.txt.
"""
import argparse
import json
import random
import unicodedata
from pathlib import Path

from gen_chat_template_goldens import env, fnv1a64
from gen_tokenizer_goldens import FLINT_CASES, QWEN_CASES


MODEL = "MiniMaxAI/MiniMax-M3.1-Flash"
REVISION = "2f33a8b9ae49ce7b51c994df97166a7fd059bc2e"


def cases():
    user = {"role": "user", "content": "Hello 世界 👋"}
    yield "empty", {"messages": []}
    yield "plain", {"messages": [user]}
    yield "no_generation", {"messages": [user], "add_generation_prompt": False}
    for role in ("root", "system", "developer"):
        yield role, {"messages": [{"role": role, "content": "Follow my instructions."}, user]}
    yield "root_developer", {"messages": [
        {"role": "root", "content": "Root", "current_date": "2026-11-04", "current_location": "London"},
        {"role": "developer", "content": "Developer"}, user]}
    for effort in ("low", "medium", "high", "max"):
        yield "effort_" + effort, {"messages": [user], "reasoning_effort": effort}
    yield "effort_alias", {"messages": [user], "effort": "high"}
    yield "history", {"messages": [user,
        {"role": "assistant", "reasoning_content": "Check carefully.", "content": "Hello!"},
        {"role": "developer", "content": "Now be brief."}, user,
        {"role": "assistant", "content": "<mm:think>Second thought.</mm:think>Done."}, user]}
    yield "plain_assistant", {"messages": [user, {"role": "assistant", "content": "Done."}, user]}
    yield "multimodal", {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe "}, {"type": "image"}, {"type": "video"}, " please"]}]}
    tools = [{"type": "function", "function": {"name": "inspect", "description": "检查",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "settings": {"type": "object"}, "items": {"type": "array"}}}}}]
    yield "tools", {"messages": [user], "tools": tools}
    def call(arguments):
        return {"type": "function", "function": {"name": "inspect", "arguments": arguments}}
    for name, args in [
        ("scalars", {"name": "123", "count": 2, "score": 1.25, "enabled": True, "disabled": False, "absent": None}),
        ("nested", {"name": "a<b & c", "settings": {"depth": 2, "skip": None, "sub": {"on": True}},
                    "items": [{"name": "é", "tags": ["a", "b"]}, {"name": "中"}]}),
        ("empty", {"name": "", "settings": {}, "items": []}),
    ]:
        yield "tool_" + name, {"tools": tools, "messages": [user,
            {"role": "assistant", "reasoning_content": "Inspect.", "content": "", "tool_calls": [call(args)]},
            {"role": "tool", "content": "Success."}, user]}
    yield "parallel_calls", {"tools": tools, "messages": [user,
        {"role": "assistant", "tool_calls": [call({"name": "first"}), call({"name": "second"})]},
        {"role": "tool", "content": "First."}, {"role": "tool", "content": [
            {"type": "text", "text": "Second."}, {"type": "image"}]}]}


def main():
    import jinja2
    import tokenizers
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=Path.home() / ".cache/huggingface/hub" /
                    "models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots" / REVISION)
    ap.add_argument("--out", type=Path, default=Path("tests/data/minimax_prompt_goldens.jsonl"))
    args = ap.parse_args()
    raw = (args.snapshot / "chat_template.jinja").read_bytes()
    tok_path = args.snapshot / "tokenizer.json"
    tok = tokenizers.Tokenizer.from_file(str(tok_path))
    template = env().from_string(raw.decode())
    records = list(cases())
    token_cases = FLINT_CASES + QWEN_CASES + [
        "]~!b[]~b]system\n[e~[", "]<]minimax[>[<tool_call>",
        "<mm:think>Thought.</mm:think>", "Isn't I'M we're WE'RE 1234567890",
        "\u1100\u1161\u11a8", "abc\r\n///XYZ", " !\n///x", " \u0301ABC's",
    ]
    rng = random.Random(3101)
    alphabet = ["A", "b", "ǅ", "ʰ", "中", "ท", "α", "Я", "3", "１", "¼",
                "\u0301", "\u20dd", "'S", "'ll", " ", "\t", "\r", "\n", "!", "/", "😀"]
    token_cases += ["".join(rng.choices(alphabet, k=rng.randrange(1, 50))) for _ in range(512)]
    header = {"model": MODEL, "revision": REVISION, "jinja2": jinja2.__version__,
              "tokenizers": tokenizers.__version__, "cases": len(records) + len(token_cases),
              "template_hash": f"{fnv1a64(raw):016x}",
              "tokenizer_revision": f"{fnv1a64(tok_path.read_bytes()):016x}"}
    with args.out.open("w") as output:
        output.write(json.dumps(header) + "\n")
        for name, kwargs in records:
            kwargs.setdefault("add_generation_prompt", True)
            rendered = template.render(**kwargs)
            output.write(json.dumps({"name": name, "kwargs": kwargs, "render": rendered,
                                     "ids": tok.encode(rendered).ids}, ensure_ascii=False) + "\n")
        for i, text in enumerate(token_cases):
            output.write(json.dumps({"name": f"tokenizer_{i}", "text": text,
                                     "normalized": unicodedata.normalize("NFC", text),
                                     "ids": tok.encode(text).ids}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} prompt and {len(token_cases)} tokenizer cases to {args.out}")


if __name__ == "__main__":
    main()

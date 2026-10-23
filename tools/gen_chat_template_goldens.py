#!/usr/bin/env python3
"""Generates tests/data/glm_chat_template_goldens.jsonl — the differential
goldens for the GLM chat template (M6 Stage 3b, DESIGN §10).

The GOLDEN SOURCE is sandboxed Jinja2 configured like transformers'
chat-template environment (src/transformers/utils/chat_template_utils.py,
verified 2026-08-31):
  * trim_blocks=True, lstrip_blocks=True
  * extensions=[jinja2.ext.loopcontrols]  (the template uses {% break %};
    plain jinja2 rejects it)
  * the tojson OVERRIDE: json.dumps(ensure_ascii=..., indent=None,
    separators=None, sort_keys=False) — jinja's built-in tojson both
    escapes HTML (<, >, &, ') and rejects the ensure_ascii kwarg, so the
    override is REQUIRED to render this template at all.

The corpus is written with ensure_ascii=False (raw UTF-8) so minijson's
reader never meets surrogate pairs (it decodes single \\uXXXX only; the
tokenizer gate had to learn that lesson the hard way).

Each case carries the render globals as JSON ("kwargs"), the expected
render ("render"), and the token ids HF tokenizers 0.23.1 produces for
the render ("ids" — the template->tokenizer integration differential;
the C++ gate re-encodes with glm_tokenizer and compares).

Case selection (per DESIGN §10: English, Chinese, code, reasoning
blocks, tool calls — plus the structural edges the template branches
on): the plain/system/multimodal user paths; the reasoning-keep path
(trailing assistant) vs the reasoning-strip path (mid-conversation
assistant, and clear_thinking=false); reasoning_content as a field vs
think-markers inside content (the content.split path) vs both; the
close-without-open split edge; tool headers flat vs OpenAI-wrapped vs
defer_loading; aligned/sorted/unaligned tool-response blocks; string
vs JSON-serialized arguments; tool_reference entries direct and nested
in outputs; empty conversation and empty content.

Keyed by chat_template.jinja's FNV-1a-64 hash (header "template_hash"),
cross-referenced with the tokenizer revision ("tokenizer_revision").

Regenerating: python3
    tools/gen_chat_template_goldens.py
(the venv pins jinja2 3.1.2 + tokenizers 0.23.1).
"""
import glob
import json
import os
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cluster_doctor import cache_root, cached_snapshot
from site_env import cache_environment


MODEL = "unsloth/GLM-5.3-Flash-FP8"


def fnv1a64(data: bytes) -> int:
    # The house hash (glm_loader.cpp / glm_tokenizer.cpp), not the FNV
    # standard basis.
    h = 1469598103934665603
    for b in data:
        h = (h ^ b) * 1099511628211 & 0xFFFFFFFFFFFFFFFF
    return h


# The template's reasoning markers, built by concatenation (heredocs and
# raw literals mangle them in tool transcripts; see the record).
THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"


def env():
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    e = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    # transformers' tojson override, character for character.
    def tojson(x, ensure_ascii=False, indent=None, separators=None,
               sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)
    e.filters["tojson"] = tojson
    # transformers' raise_exception global (a TemplateError).
    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)
    e.globals["raise_exception"] = raise_exception
    return e


def weather_tool():
    return {
        "name": "get_weather",
        "description": "Get the current weather for a city (城市天气查询).",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer", "minimum": 0},
            },
            "required": ["city"],
        },
    }


def call(cid, name, arguments):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": arguments}}


CASES = [
    ("simple_user_gen", {
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "add_generation_prompt": True,
    }),
    ("system_user_gen", {
        "messages": [
            {"role": "system",
             "content": "You are a concise geography assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "add_generation_prompt": True,
    }),
    ("chinese_user_gen", {
        "messages": [{"role": "user", "content": "写一首关于秋天的诗。"}],
        "add_generation_prompt": True,
    }),
    ("code_user_gen", {
        "messages": [{"role": "user", "content":
                      "def add(a, b):\n    return a + b\nExplain this code."}],
        "add_generation_prompt": True,
    }),
    ("empty_conversation_gen", {
        "messages": [],
        "add_generation_prompt": True,
    }),
    ("empty_user_content_gen", {
        "messages": [{"role": "user", "content": ""}],
        "add_generation_prompt": True,
    }),
    ("effort_low", {
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": True,
        "reasoning_effort": "low",
    }),
    ("effort_high", {
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": True,
        "reasoning_effort": "high",
    }),
    ("multimodal_parts_gen", {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe these in order."},
            {"type": "image_url", "image_url": {"url": "https://x/cat.png"}},
            {"type": "video_url"},
            {"type": "input_audio"},
        ]}],
        "add_generation_prompt": True,
    }),
    ("multi_turn_mid_reasoning_stripped", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant", "reasoning_content": "Two plus two is four.",
             "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "add_generation_prompt": True,
    }),
    ("trailing_assistant_reasoning_content", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant", "reasoning_content": "Two plus two is four.",
             "content": "4"},
        ],
        "add_generation_prompt": True,
    }),
    ("trailing_assistant_think_markers", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant",
             "content": THINK_OPEN + " Two plus two is four." + THINK_CLOSE +
                        " The answer is 4."},
        ],
        "add_generation_prompt": True,
    }),
    ("think_close_without_open", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant",
             "content": " partial thinking" + THINK_CLOSE + " The answer is 4."},
        ],
        "add_generation_prompt": True,
    }),
    ("trailing_assistant_both_reasoning_forms", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant",
             "reasoning_content": "Field form wins over markers.",
             "content": THINK_OPEN + " marker form" + THINK_CLOSE +
                        " The answer is 4."},
        ],
        "add_generation_prompt": True,
    }),
    ("clear_thinking_false_mid_turn", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant", "reasoning_content": "Two plus two is four.",
             "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "add_generation_prompt": True,
        "clear_thinking": False,
    }),
    ("no_generation_prompt", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant", "content": "4"},
        ],
    }),
    ("tools_flat", {
        "messages": [{"role": "user",
                      "content": "What's the weather in Paris?"}],
        "tools": [weather_tool(), {
            "name": "search_web", "description": "Search the web."}],
        "add_generation_prompt": True,
    }),
    ("tools_function_wrap", {
        "messages": [{"role": "user",
                      "content": "What's the weather in Paris?"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Weather <queries> & conditions 天气",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}],
        "add_generation_prompt": True,
    }),
    ("tools_deferred", {
        "messages": [{"role": "user",
                      "content": "What's the weather in Paris?"}],
        "tools": [weather_tool(), {
            "name": "lazy_tool", "description": "Loaded on demand.",
            "defer_loading": True}],
        "add_generation_prompt": True,
    }),
    ("tool_call_aligned", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Tokyo?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather",
                     {"city": "Paris", "days": 3, "factor": 0.5}),
                call("call_2", "get_weather", {"city": "Tokyo"})]},
            {"role": "tool", "content": "18C, partly cloudy",
             "tool_call_id": "call_1"},
            {"role": "tool", "content": "22C, clear",
             "tool_call_id": "call_2"},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_call_unsorted", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Tokyo?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather", {"city": "Paris"}),
                call("call_2", "get_weather", {"city": "Tokyo"})]},
            {"role": "tool", "content": "22C, clear",
             "tool_call_id": "call_2"},
            {"role": "tool", "content": "18C, partly cloudy",
             "tool_call_id": "call_1"},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_call_str_args", {
        # NOTE: arguments MUST be a dict (a bare JSON string crashes the
        # template at `_args.items()`); the `v is not string` branch is
        # for string VALUES inside it, rendered verbatim.
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather",
                     {"city": "Paris", "note": "asked twice"})]},
            {"role": "tool", "content": "18C, partly cloudy",
             "tool_call_id": "call_1"},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_call_float_args", {
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather",
                     {"scale": 1.0, "ratio": 0.5, "count": 3})]},
            {"role": "tool", "content": "18C",
             "tool_call_id": "call_1"},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_call_outputs_and_references", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Tokyo?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather", {"city": "Paris"}),
                call("call_2", "get_weather", {"city": "Tokyo"})]},
            {"role": "tool", "content": [
                {"tool_call_id": "call_1", "output": "18C, partly cloudy"},
                {"tool_call_id": "call_2", "output": [
                    {"type": "tool_reference", "name": "get_weather",
                     "tool_call_id": "call_9"}]},
            ]},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_reference_direct", {
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather", {"city": "Paris"})]},
            {"role": "tool", "content": [
                {"type": "tool_reference", "name": "get_weather",
                 "tool_call_id": "call_1"},
            ]},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_no_prior_call", {
        "messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "tool", "content": "18C, partly cloudy",
             "tool_call_id": "call_0"},
            {"role": "user", "content": "Thanks!"},
        ],
        "add_generation_prompt": True,
    }),
]


# The Qwen3.8-Flash-Next template: the reasoning-effort system
# preamble, the tools block, <think> reasoning kept only after the last
# user query (preserve_thinking), the <function=...><parameter=...> tool
# calls, tool responses grouped into one user turn, enable_thinking false
# emitting an empty think block.
def qwen_call(cid, name, arguments):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": arguments}}


QWEN_CASES = [
    ("simple_user_gen", {
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "add_generation_prompt": True,
    }),
    ("system_user_gen", {
        "messages": [
            {"role": "system", "content": "You are a concise geography assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "add_generation_prompt": True,
    }),
    ("chinese_user_gen", {
        "messages": [{"role": "user", "content": "写一首关于秋天的诗。"}],
        "add_generation_prompt": True,
    }),
    ("code_user_gen", {
        "messages": [{"role": "user", "content": "def add(a, b):\n    return a + b\nExplain this code."}],
        "add_generation_prompt": True,
    }),
    ("effort_low", {
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": True, "reasoning_effort": "low",
    }),
    ("effort_medium_system", {
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        "add_generation_prompt": True, "reasoning_effort": "medium",
    }),
    ("thinking_disabled", {
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": True, "enable_thinking": False,
    }),
    ("thinking_disabled_with_system", {
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        "add_generation_prompt": True, "enable_thinking": False,
    }),
    ("multi_turn_reasoning_kept_after_last_query", {
        "messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "thinking about Q1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "thinking about Q2"},
        ],
        "add_generation_prompt": False,
    }),
    ("multi_turn_preserve_thinking_false", {
        "messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "thinking about Q1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "thinking about Q2"},
            {"role": "user", "content": "Q3"},
        ],
        "add_generation_prompt": True, "preserve_thinking": False,
    }),
    ("assistant_without_reasoning", {
        "messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ],
        "add_generation_prompt": True,
    }),
    ("content_parts_text", {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]}],
        "add_generation_prompt": True,
    }),
    ("empty_user_content_gen", {
        "messages": [{"role": "user", "content": ""}],
        "add_generation_prompt": True,
    }),
    ("tools_no_system", {
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [weather_tool()], "add_generation_prompt": True,
    }),
    ("tools_with_system", {
        "messages": [{"role": "system", "content": "You are helpful."},
                     {"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [weather_tool()], "add_generation_prompt": True, "reasoning_effort": "low",
    }),
    ("tool_call_and_response", {
        "messages": [
            {"role": "user", "content": "What's the weather in Paris for 3 days?"},
            {"role": "assistant", "content": "", "reasoning_content": "I should call the tool.",
             "tool_calls": [qwen_call("c1", "get_weather", {"city": "Paris", "days": 3})]},
            {"role": "tool", "content": "{\"temp\": 21, \"unit\": \"C\"}"},
        ],
        "tools": [weather_tool()], "add_generation_prompt": True,
    }),
    ("tool_call_with_content_and_two_calls", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Rome?"},
            {"role": "assistant", "content": "Let me check both.",
             "tool_calls": [qwen_call("c1", "get_weather", {"city": "Paris"}),
                            qwen_call("c2", "get_weather", {"city": "Rome", "days": 2})]},
            {"role": "tool", "content": "Paris: 21C"},
            {"role": "tool", "content": "Rome: 25C"},
            {"role": "assistant", "content": "Paris 21C, Rome 25C."},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()], "add_generation_prompt": True,
    }),
    ("tool_call_multiline_and_nested_args", {
        "messages": [
            {"role": "user", "content": "Run it."},
            {"role": "assistant", "content": "",
             "tool_calls": [qwen_call("c1", "run", {"code": "print(1)\nprint(2)\n", "opts": {"a": 1, "b": [1, 2]}, "flag": True})]},
            {"role": "tool", "content": "1\n2"},
        ],
        "tools": [{"name": "run", "description": "Run code", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}],
        "add_generation_prompt": True,
    }),
    ("tool_response_then_generation", {
        "messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "", "tool_calls": [qwen_call("c1", "get_weather", {"city": "Oslo"})]},
            {"role": "tool", "content": "cold"},
        ],
        "tools": [weather_tool()], "add_generation_prompt": True, "enable_thinking": False,
    }),
    ("no_generation_prompt", {
        "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
        "add_generation_prompt": False,
    }),
]


# The GLM-4.7 template: the GLM-5.3 one without the reasoning
# effort preamble, the deferred tools, the tool references and the media
# reminders; the reasoning split uses rstrip('\n') / lstrip('\n') and
# enable_thinking=false closes the think block in the generation prompt.
# The GLM-5.3 cases that render on it, plus the thinking switch.
GLM4_SKIP = {"effort_low", "effort_high", "multimodal_parts_gen", "tools_deferred",
             "tool_call_outputs_and_references", "tool_reference_direct"}
GLM4_CASES = [c for c in CASES if c[0] not in GLM4_SKIP] + [
    ("thinking_disabled", {
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": True, "enable_thinking": False,
    }),
    ("thinking_disabled_with_system", {
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        "add_generation_prompt": True, "enable_thinking": False,
    }),
    ("trailing_assistant_think_markers_newlines", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant",
             "content": THINK_OPEN + "\n\nTwo plus two is four.\n\n" + THINK_CLOSE +
                        "\n\nThe answer is 4.\n"},
        ],
        "add_generation_prompt": True,
    }),
    ("mid_turn_think_markers_stripped", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant",
             "content": THINK_OPEN + "\nreasoning\n" + THINK_CLOSE + "\n4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "add_generation_prompt": True,
    }),
    ("tool_call_two_turns", {
        "messages": [
            {"role": "user", "content": "Weather in Paris, then Tokyo."},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather", {"city": "Paris"})]},
            {"role": "tool", "content": "18C, partly cloudy",
             "tool_call_id": "call_1"},
            {"role": "assistant", "tool_calls": [
                call("call_2", "get_weather", {"city": "Tokyo", "days": 2})]},
            {"role": "tool", "content": "22C, clear",
             "tool_call_id": "call_2"},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_call_outputs_list", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Tokyo?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather", {"city": "Paris"}),
                call("call_2", "get_weather", {"city": "Tokyo"})]},
            {"role": "tool", "content": [
                {"tool_call_id": "call_1", "output": "18C, partly cloudy"},
                {"tool_call_id": "call_2", "output": "22C, clear"},
            ]},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
]


# The MiMo-V2.6-Flash template (2026-09-22): ChatML roles, the tools as
# one JSON line each inside <tools> in a leading system turn (or after a
# message's own `tools`), assistant turns always opening with the
# reasoning in think markers (empty when none), tool calls as
# <tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>
# with non-string values JSON-serialized, tool responses as a plain `tool`
# role turn, image / audio / video parts as their pad markers, and
# enable_thinking=false closing the think block in the generation prompt.
MIMO_SKIP = {"effort_low", "effort_medium_system", "multi_turn_preserve_thinking_false"}
MIMO_CASES = [c for c in QWEN_CASES if c[0] not in MIMO_SKIP] + [
    ("mimo_multimodal_parts", {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe "},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": " and "},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
            {"type": "video", "video": "x.mp4"},
            {"type": "text", "text": "."},
        ]}],
        "add_generation_prompt": True,
    }),
    ("mimo_message_tools", {
        "messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Weather in Oslo?", "tools": [weather_tool()]},
        ],
        "add_generation_prompt": True,
    }),
    ("mimo_message_tools_empty_body", {
        "messages": [{"role": "user", "content": "", "tools": [weather_tool()]}],
        "add_generation_prompt": True,
    }),
    ("mimo_string_arguments", {
        "messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\"}"}}]},
            {"role": "tool", "content": "cold"},
        ],
        "tools": [weather_tool()], "add_generation_prompt": True,
    }),
    ("mimo_custom_tool_input", {
        "messages": [
            {"role": "user", "content": "Run it."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "custom", "custom": {"name": "shell", "input": "ls -la\n"}}]},
            {"role": "tool", "content": "total 0"},
        ],
        "tools": [{"type": "custom", "custom": {"name": "shell", "description": "Run a shell command"}}],
        "add_generation_prompt": True,
    }),
    ("mimo_reasoning_then_tool_call", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Rome?"},
            {"role": "assistant", "content": "Checking.", "reasoning_content": "two cities",
             "tool_calls": [qwen_call("c1", "get_weather", {"city": "Paris"}),
                            qwen_call("c2", "get_weather", {"city": "Rome", "days": 2})]},
            {"role": "tool", "content": "Paris: 21C"},
            {"role": "tool", "content": "Rome: 25C"},
        ],
        "tools": [weather_tool()], "add_generation_prompt": True, "enable_thinking": False,
    }),
    ("mimo_chinese_tool_args", {
        "messages": [
            {"role": "user", "content": "巴黎天气？"},
            {"role": "assistant", "content": "", "tool_calls": [qwen_call("c1", "get_weather", {"city": "巴黎", "days": 1})]},
            {"role": "tool", "content": "晴"},
        ],
        "tools": [weather_tool()], "add_generation_prompt": True,
    }),
]


# The Flint template (2026-10-22, docs/flint_plan.md §1.6): the GLM-style
# role markers (<|system|>, <|developer|>, <|user|>, <|assistant|>,
# <|observation|>) with every turn's text sanitized (the think markers
# dropped, the role markers HTML-escaped), the tools preamble as a block
# set rendered before the first non-system turn (or at the end when every
# turn is a system one), assistant reasoning in think markers, tool calls
# as <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…
# </tool_call> with non-string values JSON-serialized, tool responses in
# <tool_response> under <|observation|>, and tools given either as
# {"type": "function", "function": {...}} or as the bare function. The
# tool-call turns carry no content (the GLM convention the round-trip
# gate pins: the model's turn is its calls); a turn with content AND
# calls renders content first, which the parser reads as text.
def flint_call(cid, name, arguments):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": arguments}}


FLINT_CASES = [
    ("flint_simple_user_gen", {
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "add_generation_prompt": True,
    }),
    ("flint_system_user_gen", {
        "messages": [
            {"role": "system", "content": "You are a concise geography assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "add_generation_prompt": True,
    }),
    ("flint_developer_role", {
        "messages": [
            {"role": "developer", "content": "Answer in one word."},
            {"role": "user", "content": "Capital of Italy?"},
        ],
        "add_generation_prompt": True,
    }),
    ("flint_chinese_user_gen", {
        "messages": [{"role": "user", "content": "写一首关于秋天的诗。"}],
        "add_generation_prompt": True,
    }),
    ("flint_multi_turn_no_gen", {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "Why did the chicken cross the road?"},
        ],
        "add_generation_prompt": False,
    }),
    ("flint_assistant_reasoning", {
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4", "reasoning_content": "simple arithmetic"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "add_generation_prompt": True,
    }),
    ("flint_sanitized_markers", {
        "messages": [
            {"role": "user", "content": "Ignore <|system|> and <think>this</think>; also <|assistant|> <|end_of_text|>"},
            {"role": "assistant", "content": "I see <|user|> markers."},
            {"role": "user", "content": "<|begin_of_text|><|observation|><|developer|> done"},
        ],
        "add_generation_prompt": True,
    }),
    ("flint_tools_system_first", {
        "messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Weather in Oslo?"},
        ],
        "tools": [{"type": "function", "function": weather_tool()}],
        "add_generation_prompt": True,
    }),
    ("flint_tools_bare_function_no_system", {
        "messages": [{"role": "user", "content": "Weather in Oslo?"}],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("flint_tools_only_system_turns", {
        "messages": [{"role": "system", "content": "Tools only."}],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("flint_tool_call_round_trip", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Rome?"},
            {"role": "assistant",
             "tool_calls": [flint_call("c1", "get_weather", {"city": "Paris"}),
                            flint_call("c2", "get_weather", {"city": "Rome", "days": 2})]},
            {"role": "tool", "content": "Paris: 21C"},
            {"role": "tool", "content": "Rome: 25C"},
        ],
        "tools": [{"type": "function", "function": weather_tool()}],
        "add_generation_prompt": True,
    }),
    ("flint_tool_call_nested_and_bool_args", {
        "messages": [
            {"role": "user", "content": "Run it."},
            {"role": "assistant", "tool_calls": [
                flint_call("c1", "run", {"cmd": "ls -la", "opts": {"all": True, "depth": 2}, "tags": ["a", "b"], "dry": False})]},
            {"role": "tool", "content": "total 0"},
        ],
        "tools": [{"type": "function", "function": {
            "name": "run", "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {
                "cmd": {"type": "string"}, "opts": {"type": "object"},
                "tags": {"type": "array", "items": {"type": "string"}}, "dry": {"type": "boolean"}},
                "required": ["cmd"]}}}],
        "add_generation_prompt": True,
    }),
    ("flint_tool_call_bare_call_shape", {
        "messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Oslo"}}]},
            {"role": "tool", "content": "cold"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("flint_chinese_tool_args", {
        "messages": [
            {"role": "user", "content": "巴黎天气？"},
            {"role": "assistant", "tool_calls": [flint_call("c1", "get_weather", {"city": "巴黎", "days": 1, "lang": "zh"})]},
            {"role": "tool", "content": "晴"},
        ],
        "tools": [{"type": "function", "function": weather_tool()}],
        "add_generation_prompt": True,
    }),
    ("flint_code_user_gen", {
        "messages": [{"role": "user", "content": "def add(a, b):\n    return a + b\n\nExplain."}],
        "add_generation_prompt": True,
    }),
    ("flint_tool_call_three_calls", {
        "messages": [
            {"role": "user", "content": "Weather in Oslo, Lima and Perth?"},
            {"role": "assistant", "tool_calls": [flint_call("c1", "get_weather", {"city": "Oslo"}),
                                                 flint_call("c2", "get_weather", {"city": "Lima", "days": 3}),
                                                 flint_call("c3", "get_weather", {"city": "Perth", "days": 0})]},
            {"role": "tool", "content": "Oslo: 4C"},
            {"role": "tool", "content": "Lima: 19C"},
            {"role": "tool", "content": "Perth: 27C"},
        ],
        "tools": [{"type": "function", "function": weather_tool()}],
        "add_generation_prompt": True,
    }),
    ("flint_tool_call_two_turns", {
        "messages": [
            {"role": "system", "content": "Use the tools."},
            {"role": "user", "content": "Weather in Oslo?"},
            {"role": "assistant", "tool_calls": [flint_call("c1", "get_weather", {"city": "Oslo"})]},
            {"role": "tool", "content": "Oslo: 4C"},
            {"role": "assistant", "content": "Oslo is at 4C."},
            {"role": "user", "content": "And Lima?"},
            {"role": "assistant", "tool_calls": [flint_call("c2", "get_weather", {"city": "Lima", "days": 1, "units": "C"})]},
            {"role": "tool", "content": "Lima: 19C"},
        ],
        "tools": [{"type": "function", "function": weather_tool()}],
        "add_generation_prompt": True,
    }),
    # Three turns carry keys the tool does not declare (units, lang, note /
    # strict): the grammar gate's explicit opt-out count.
    ("flint_tool_call_float_and_null_args", {
        "messages": [
            {"role": "user", "content": "Weather at the coordinates?"},
            {"role": "assistant", "tool_calls": [flint_call("c1", "get_weather", {"city": "Oslo", "days": 2, "note": None, "strict": True})]},
            {"role": "tool", "content": "4C"},
        ],
        "tools": [{"type": "function", "function": weather_tool()}],
        "add_generation_prompt": True,
    }),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--template", default=None)
    ap.add_argument("--tokenizer-json", default=None)
    ap.add_argument("--out", dest="out_opt", default=None)
    args = ap.parse_args()
    model = args.model
    is_qwen = "Qwen" in model
    is_glm4 = "GLM-4" in model
    is_mimo = "MiMo" in model
    is_flint = "flint" in model.lower()
    cases = FLINT_CASES if is_flint else MIMO_CASES if is_mimo else QWEN_CASES if is_qwen else GLM4_CASES if is_glm4 else CASES
    out_path = args.out_opt or args.out or (
        "tests/data/flint_chat_template_goldens.jsonl" if is_flint else
        "tests/data/mimo_chat_template_goldens.jsonl" if is_mimo else
        "tests/data/qwen_chat_template_goldens.jsonl" if is_qwen else
        "tests/data/glm4_chat_template_goldens.jsonl" if is_glm4 else "tests/data/glm_chat_template_goldens.jsonl")
    if args.template and args.tokenizer_json:
        tpl_path, tok_path = args.template, args.tokenizer_json
    else:
        snap = cached_snapshot(model, cache_root(cache_environment()))
        tpl_path = str(snap / "chat_template.jinja")
        tok_path = str(snap / "tokenizer.json")
    tpl_raw = pathlib.Path(tpl_path).read_bytes()
    tok_raw = pathlib.Path(tok_path).read_bytes()
    template_hash = f"{fnv1a64(tpl_raw):016x}"
    tok_hash = f"{fnv1a64(tok_raw):016x}"
    if not is_qwen and not is_glm4 and not is_mimo and not is_flint and tok_hash != "700b4469fc43f23b":
        sys.exit(f"unexpected tokenizer revision {tok_hash} — the tokenizer "
                 "goldens are keyed to 700b4469fc43f23b")

    jinja_env = env()
    template = jinja_env.from_string(tpl_raw.decode("utf-8"))
    import tokenizers
    tok = tokenizers.Tokenizer.from_file(tok_path)

    with open(out_path, "w", encoding="utf-8") as f:
        header = {
            "model": model,
            "jinja2": jinja_env.jinja2.__version__ if hasattr(
                jinja_env, "jinja2") else "3.1.2",
            "template_hash": template_hash,
            "tokenizer_revision": tok_hash,
            "cases": len(cases),
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for name, kwargs in cases:
            rendered = template.render(**kwargs)
            if not rendered:
                sys.exit(f"case {name} rendered empty — generator bug")
            ids = tok.encode(rendered).ids
            f.write(json.dumps({"name": name, "kwargs": kwargs,
                                "render": rendered, "ids": ids},
                               ensure_ascii=False) + "\n")
            print(f"{name:42s} {len(rendered):5d} chars  {len(ids):4d} ids")
    print(f"wrote {out_path}: {len(cases)} cases, template {template_hash}")


if __name__ == "__main__":
    main()

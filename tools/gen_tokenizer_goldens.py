#!/usr/bin/env python3
"""Generates tests/data/glm_tokenizer_goldens.jsonl — the differential
goldens for the GLM tokenizer (M6 Stage 3).

The GOLDEN SOURCE is HF tokenizers 0.23.1 (the pinned reference — the
venv configured with requirements-tools.txt) applied to this script's case list. The
C++ gate (tests/host/glm_tokenizer_test.cpp) loads the SAME corpus and
asserts byte-exact encode/decode parity, refusing to run against a
tokenizer.json whose FNV-1a-64 revision hash differs from the corpus
header (the revision key; the C++ and this script share the hash
definition).

Case selection: the corpus covers the pretokenizer's decision surface —
contractions in both cases, the whitespace alternatives (space-prefix,
trailing-space, newline runs, tabs before letters, nbsp), digit
clumping, Unicode letter/number classes, multi-script text, emoji
(ZWJ sequences, skin tones), code fences, and the added tokens
standalone and inline. Deliberately avoids codepoints added to Unicode
after 13.0 so table-version skew between this generator (Python 3.12 /
Unicode 15.0) and the C++ tables cannot flake the gate (see
tools/gen_unicode_tables.py).

Regenerating: python3 tools/gen_tokenizer_goldens.py
  [--model ORG/NAME] [--tokenizer-json PATH] [--out FILE]
(the corpus and the default output follow the model's name: GLM, Qwen,
MiMo, DeepSeek-V4.1-*, DeepSeek-V4-* — e.g.
  .venv/bin/python tools/gen_tokenizer_goldens.py --model deepseek-ai/DeepSeek-V4-Flash-0731
writes tests/data/dsv4_tokenizer_goldens.jsonl)

The Qwen3.8-Flash-Next corpus (2026-09-09) adds the NFC and mark cases its
tokenizer.json needs — decomposed and precomposed accents, singleton
decompositions, Hangul jamo, scripts with combining marks, digit runs (one
number per pretoken), a mark at the start of text — and its own added
tokens; generated on the 5090 box (tokenizers 0.22.2):
  python tools/gen_tokenizer_goldens.py --model Qwen/Qwen3.8-Flash-Next-FP8
      --tokenizer-json /tmp/qwen38_tokenizer.json --out tests/data/qwen_tokenizer_goldens.jsonl
"""
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cluster_doctor import cache_root, cached_snapshot
from site_env import cache_environment


MODEL = "unsloth/GLM-5.3-Flash-FP8"


def fnv1a64(data: bytes) -> int:
    # Matches the C++ (glm_tokenizer.cpp + glm_loader.cpp's house hash):
    # the project's established basis, not the FNV standard basis.
    h = 1469598103934665603
    for b in data:
        h = (h ^ b) * 1099511628211 & 0xFFFFFFFFFFFFFFFF
    return h


THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"
USER = "<|" + "user|>"
ASSIST = "<|" + "assistant|>"
EOS = "<|" + "end" + "of" + "text|>"

CASES = [
    # The fabric-run anchors (the M6 Stage 1d/2 prompts and generations).
    "The capital of France is",
    " Paris",
    ".",
    " In",
    " French,",
    " is",
    " spelled",
    # Contractions: every form, both cases, embedded.
    "I don't think it's CAN'T 'LL 'Re we're I'm 'll 'Ve 'd 'm 'T 'S",
    "can't won't shouldn't they're we've y'all",
    # Whitespace alternatives.
    "  5", " a", "   abc", "\tabc", "\n\n  x", "!!!\n", "a \n b",
    "abc   ", " \n", "\r\n\r\n", "hello\n\nworld\n", "x\u00a0y",
    # Digit clumping and numeric forms.
    "1234567", "0000", "0.5", "1,000,000.50", "123456789012345",
    # Unicode letter/number classes (Nl, No included — not just ASCII
    # digits).
    "½ Ⅻ ² ⅓",
    # Multi-script text.
    "你好，世界",
    "Привет мир",
    "Γεια σου",
    "สวัสดีครับ",
    "שלום עולם",
    "مرحبا بالعالم",
    "naïve café Straße",
    # Emoji (pictographic, skin tone, ZWJ sequence).
    "emoji 👍🏽 test",
    "family 👨‍👩‍👧‍👦 walk",
    # Code and punctuation runs.
    "code ```python\nprint(1)\n``` end",
    "!!!???***", "a-b_c+d=e",
    # Added tokens: standalone, inline, and a template-shaped run.
    THINK_OPEN,
    THINK_CLOSE,
    USER,
    ASSIST,
    EOS,
    "a" + THINK_OPEN + "b",
    "x" + THINK_OPEN + " r" + THINK_CLOSE + "y" + USER + "hi" + ASSIST,
    EOS + THINK_OPEN + " e" + THINK_CLOSE + USER + "u" + ASSIST,
    "before " + USER + " after",
    # Degenerate inputs.
    "", " ", "  ", "\n", "x",
    # Realistic mixed sample.
    "def add(a, b):\n    return a + b  # inline comment\n\nprint(add(2, 3))\n",
    "The quick brown fox jumps over the lazy dog. 0123456789 !@#$%^&*()",
]


IM_START = "<|" + "im_start|>"
IM_END = "<|" + "im_end|>"
TOOL_CALL = "<" + "tool_call>"
TOOL_CALL_END = "</" + "tool_call>"
TOOL_RESP = "<" + "tool_response>"

QWEN_CASES = [c for c in CASES if USER not in c and ASSIST not in c] + [
    # Qwen's added tokens standalone, inline, template-shaped.
    IM_START, IM_END, TOOL_CALL, TOOL_CALL_END, TOOL_RESP,
    IM_START + "user\nhi" + IM_END + "\n" + IM_START + "assistant\n",
    "x" + THINK_OPEN + "\nr\n" + THINK_CLOSE + "\n\ny" + IM_END,
    EOS + IM_START + "system\nYou are helpful." + IM_END,
    TOOL_CALL + "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n" + TOOL_CALL_END,
    # NFC: decomposed vs precomposed, singletons, reordering, Hangul jamo.
    "e\u0301", "\u00e9", "Cafe\u0301 au lait", "caf\u00e9",
    "\u212b \u2126 \u00c5 \u03a9",
    "\ufb01ne \ufb02ow",
    "\u1e9b\u0323", "s\u0323\u0307", "\u0073\u0307\u0323",
    "\u1100\u1161\u11a8", "\u1100\u1161", "\ud55c\uad6d\uc5b4",
    "x\u0301\u0302y", "\u0301abc", " \u0301x", "1\u0301", "a\u0301 b",
    "\u0327\u0301e", "q\u0307\u0323",
    # Marks inside letter runs: Devanagari, Thai, Arabic with harakat, Vietnamese.
    "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e04\u0e23\u0e31\u0e1a",
    "\u0645\u064f\u062d\u064e\u0645\u0651\u064e\u062f",
    "Ti\u1ebfng Vi\u1ec7t", "Tie\u0302\u0301ng Vie\u0323\u0302t",
    # Digits: one per pretoken; mixed with marks and punctuation.
    "12345", "3.14159", "2024-09-09", "v2.1.0", "1st 2nd 3rd", "\u00bd\u00bc",
    "phone: +1 (555) 010-9999",
    # Emoji with modifiers (Sk) and ZWJ (Cf): both outside the mark class.
    "\U0001f44d\U0001f3fd", "\U0001f468\u200d\U0001f469\u200d\U0001f467",
    # A combining enclosing mark and a spacing mark after punctuation.
    "a\u20dd", "(\u0301)",
]


# The DeepSeek-V4.1 corpus (2026-09-13): its three-stage pre-tokenizer —
# number runs cut in threes (every \p{N} script), CJK runs isolated (the
# three literal ranges; Korean and halfwidth kana are letters), the
# punctuation+ASCII-letters alternative (".foo", "'t"), the \p{P}/\p{S}
# run class (format and control characters fall between matches), and its
# own added tokens (the DSML markers, the role tokens, the placeholders).
DS_BOS = "<｜begin▁of▁sentence｜>"
DS_EOS = "<｜end▁of▁sentence｜>"
DS_USER = "<｜User｜>"
DS_ASSIST = "<｜Assistant｜>"
DS_SYSTEM = "<｜System｜>"
DS_DSML = "｜DSML｜"
DSV41_CASES = [c for c in CASES if USER not in c and ASSIST not in c and EOS not in c] + [
    DS_BOS, DS_EOS, DS_USER, DS_ASSIST, DS_SYSTEM, DS_DSML, "<｜latest_reminder｜>", "<｜tool▁calls▁begin｜>",
    "<｜place▁holder▁no▁7｜>", "<｜deepseek_image｜>",
    DS_BOS + DS_SYSTEM + "You are a helpful assistant." + DS_USER + "Hello" + DS_ASSIST + THINK_CLOSE + "Hi!" + DS_EOS,
    DS_BOS + DS_SYSTEM + "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    + "You are a helpful assistant." + DS_USER + "What is 2+2?" + DS_ASSIST + THINK_OPEN,
    "Simple arithmetic." + THINK_CLOSE + "2 + 2 = 4." + DS_EOS,
    "\n\n<" + DS_DSML + " calls>\n<" + DS_DSML + ' invoke name="get_weather">\n<' + DS_DSML
    + ' parameter name="city" string="true">Paris</' + DS_DSML + " parameter>\n<" + DS_DSML
    + ' parameter name="days" string="false">3</' + DS_DSML + " parameter>\n</" + DS_DSML + " invoke>\n</" + DS_DSML
    + " calls>" + DS_EOS,
    DS_USER + "<tool_result>{\"temp\": 21}</tool_result>" + DS_ASSIST + THINK_OPEN,
    "x" + DS_DSML + "y", "before " + DS_USER + " after",
    # CJK isolation and the scripts around it.
    "你好，世界", "日本語のテキスト", "你好abc123世界", "半角ｶﾅ ①②③", "한국어 텍스트", "中文 English 混合 text",
    "こんにちは世界！", "東京タワーは333メートル",
    # Number runs in threes, every script.
    "1234567", "x1234y", "a1b22c333d4444", "١٢٣٤", "１２３４５", "3.14159", "2024-09-13", "  5 a", "v2.1.0",
    # Punctuation + ASCII letters, punctuation/symbol runs, symbols.
    ".foo bar-baz", "$abc(def)", "I don't", "(x)", "ab.cd", "@user #tag", "a+b=c", "x<y>z", "~/.bashrc", "C++ & C#",
    "€100 £5 ¥3", "→ ← ↑", "a·b", "«quoted»", "…", "?!x", " ?!x", "!!!\n\nx",
    # Format and control characters: gaps between matches.
    "a\u200db", "x\u0000y", "\ufeffbom", "tab\tx", "x\u00a0y", "a\u0301 b", "e\u0301", " \u0301x",
    "\U0001f44d\U0001f3fd", "\U0001f468\u200d\U0001f469\u200d\U0001f467", "a\u200d",
    "über naïve", "Straße", "Ελληνικά", "Кириллица", "עברית", "العربية",
]


# The DeepSeek-V4 corpus (DeepSeek-V4-Flash-0731, 2026-10-01): the V4.1
# vocabulary, merges and pre-tokenizer with nine added tokens changed — no
# <｜System｜> (id 128799 is a placeholder here, so the text is ordinary BPE
# pieces; likewise <｜deepseek_image｜>), and <｜image｜>, <｜image2｜> and the
# six table-markup tokens where V4.1 has placeholders. The V4.1 cases all
# stand (as text where the token is gone); added: the changed tokens, the
# quick-instruction task tokens, and the shapes of this model's encoder
# (encoding_dsv4.py) — the effort preambles, the bare system text, and the
# DSML block in its spelling (no space after the tag, "tool_calls").
DSV4_EFFORT_HIGH = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, "
    "rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered "
    "alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n")
DSV4_EFFORT_MAX = (
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose "
    "the problem into its most fundamental components, trace every causal chain to its root, and resolve the "
    "underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain "
    "that no assumption remains unchecked and no error remains undiscovered.\n\n")
DSV4_CASES = DSV41_CASES + [
    "<｜place▁holder▁no▁799｜>", "<｜image｜>", "<｜image2｜>", "<｜table｜>", "<｜/table>｜", "<｜tr｜>", "<｜/tr｜>",
    "<｜td｜>", "<｜/td｜>", "<｜table｜><｜tr｜><｜td｜>cell 1<｜/td｜><｜td｜>2<｜/td｜><｜/tr｜><｜/table>｜",
    "<｜action｜>", "<｜query｜>", "<｜authority｜>", "<｜domain｜>", "<｜title｜>", "<｜extracted_url｜>", "<｜read_url｜>",
    DS_BOS + "You are a helpful assistant." + DS_USER + "Hello" + DS_ASSIST + THINK_CLOSE + "Hi!" + DS_EOS,
    DS_BOS + DSV4_EFFORT_HIGH + "You are a helpful assistant." + DS_USER + "What is 2+2?" + DS_ASSIST + THINK_OPEN,
    DS_BOS + DSV4_EFFORT_MAX + DS_USER + "What is 2+2?" + DS_ASSIST + THINK_OPEN,
    DS_BOS + "System text.<｜latest_reminder｜>2026-07-31,Friday,Paris,App,English" + DS_USER + "Which day?" + DS_ASSIST
    + THINK_CLOSE + "<｜action｜>",
    "\n\n<" + DS_DSML + "tool_calls>\n<" + DS_DSML + 'invoke name="get_weather">\n<' + DS_DSML
    + 'parameter name="city" string="true">Paris</' + DS_DSML + "parameter>\n<" + DS_DSML
    + 'parameter name="days" string="false">3</' + DS_DSML + "parameter>\n</" + DS_DSML + "invoke>\n</" + DS_DSML
    + "tool_calls>" + DS_EOS,
    "<" + DS_DSML + 'invoke name="now">\n\n</' + DS_DSML + "invoke>\n<" + DS_DSML + 'invoke name="now">\n</' + DS_DSML
    + "invoke>\n",
    'You can invoke tools by writing a "<' + DS_DSML + 'tool_calls>" block like the following:\n\n<' + DS_DSML
    + "tool_calls>\n<" + DS_DSML + 'invoke name="$TOOL_NAME">\n<' + DS_DSML
    + 'parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</' + DS_DSML + "parameter>\n...\n</"
    + DS_DSML + "invoke>\n</" + DS_DSML + "tool_calls>\n",
    DS_EOS + DS_USER + "<tool_result>found\n\n[Unsupported audio]</tool_result>\n\nAnd now?" + DS_ASSIST + THINK_OPEN,
    "该助手为DeepSeek，由深度求索公司创造。<｜latest_reminder｜>2026-02-21,星期六,广州,App,中文",
]


# The MiMo-V2.6-Flash corpus (2026-09-22): the Qwen2 regex under NFC —
# GLM's letter/punctuation classes (marks are punctuation, not letters)
# with a single \p{N} per pretoken — and its own added tokens (Qwen's plus
# the audio / video markers); the tool-call block in the MiMo template's
# newline-free form.
MIMO_CASES = QWEN_CASES + [
    "<|mimo_audio_start|>", "<|mimo_audio_end|>", "<|mimo_video_start|>", "<|mimo_video_end|>",
    "<|audio_pad|>", "<|mimo_audio_eod|>", "<|vision_start|><|image_pad|><|vision_end|>",
    IM_START + "assistant\n" + THINK_OPEN + "plan" + THINK_CLOSE + "Sure." + IM_END,
    IM_START + "assistant\n" + THINK_OPEN + THINK_CLOSE + "Sure." + IM_END,
    TOOL_CALL + "<function=get_weather><parameter=city>Paris</parameter></function>" + TOOL_CALL_END,
    IM_START + "user\n" + TOOL_RESP + "{\"temp\": 21}" + "</" + "tool_response>" + IM_END,
    # Marks are punctuation under this regex: a mark run after a letter run
    # splits, a leading mark is its own piece.
    "a\u0301\u0302b", "x\u20dd y", "\u0301abc",
]


# The Flint corpus (2026-10-22, docs/flint_plan.md §1.6): the DeepSeek
# three-stage shape (number runs of at most three, the CJK ranges) with
# the o200k main pattern over the case classes — upper runs (Lu, Lt, Lm,
# Lo, M) before lower runs (Ll, Lm, Lo, M) with the contraction suffix —
# and Flint's own added tokens (the GLM-style role and tool markers).
FLINT_EOS = "<|" + "end_of_text|>"
FLINT_CASES = [c for c in CASES if c != EOS and EOS not in c] + [
    FLINT_EOS, "<|" + "begin_of_text|>", "<|" + "system|>", "<|" + "developer|>", "<|" + "observation|>",
    FLINT_EOS + THINK_OPEN + " e" + THINK_CLOSE + USER + "u" + ASSIST,
    "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
    "<|" + "observation|><tool_response>{\"temp\": 21}</tool_response>",
    "<tools>\n{\"name\": \"f\"}\n</tools>",
    # The case classes: capitalized words, all-caps, camelCase, mixed
    # runs, titlecase and Lm / Lo scripts, marks inside words, the
    # contraction after an upper run.
    "HELLO World hello WORLD's ISN'T MacDonald's", "ABCdef abcDEF AbCdEf", "ǅungla ǈubav",
    "ﬁne ﬂow", "x\u02b0y ʰ ʳ", "สวัสดี ครับ", "مرحباً بكم", "a\u0301b\u0302c", "\u0301abc", "x\u20dd y",
    "ÀÉÎÕÜ àéîõü Ñandú", "Straße STRASSE", "中文English混合Text日本語テキスト", "abc123def 12345678 x9y",
    "I'M 'LL 'rE 'VE we'D", "ΑΒΓ αβγ Αβγ", "ПРИВЕТ Привет привет",
    # The backtracking seams: a shared-class run (Thai, Lo) before capitals
    # cuts after the last shared character; capitals before it join; a
    # mark before capitals stands alone; a mark before lowercase joins.
    "\u0e17ABC", "A\u0e17B", "ABC\u0e17", "\u0e17\u0e17AB\u0e17", "\u0301ABC", "\u0301abc", "\u0301\u0302x", "\u0301!",
    "ISN'T aBc ÀBc ǅx xǅ ǅ", " ÀB", " \u0e17A", "x\u0e17y", "ABC'S abc's",
    "a.b,c;d:e", "x->y => z", "(a) [b] {c}", "#tag @user $5 50%",
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tokenizer-json", default=None)
    ap.add_argument("--out", dest="out_opt", default=None)
    args = ap.parse_args()
    model = args.model
    is_mimo = "MiMo" in model
    is_flint = "flint" in model.lower()
    is_qwen = "Qwen" in model or is_mimo  # NFC tokenizers
    is_dsv4 = "DeepSeek-V4-" in model  # DeepSeek-V4-Flash-0731 (model_type deepseek_v4), not V4.1
    is_dsv41 = "DeepSeek-V4" in model and not is_dsv4
    cases = (FLINT_CASES if is_flint else DSV4_CASES if is_dsv4 else DSV41_CASES if is_dsv41 else MIMO_CASES if is_mimo
             else QWEN_CASES if is_qwen else CASES)
    out_path = args.out_opt or args.out or (
        "tests/data/flint_tokenizer_goldens.jsonl" if is_flint else
        "tests/data/dsv4_tokenizer_goldens.jsonl" if is_dsv4 else
        "tests/data/dsv41_tokenizer_goldens.jsonl" if is_dsv41 else
        "tests/data/mimo_tokenizer_goldens.jsonl" if is_mimo else
        "tests/data/qwen_tokenizer_goldens.jsonl" if is_qwen else "tests/data/glm_tokenizer_goldens.jsonl")
    if args.tokenizer_json:
        tok_path = args.tokenizer_json
    else:
        tok_path = str(cached_snapshot(model, cache_root(cache_environment())) / "tokenizer.json")
    with open(tok_path, "rb") as f:
        raw = f.read()
    import tokenizers
    tok = tokenizers.Tokenizer.from_file(tok_path)

    with open(out_path, "w", encoding="utf-8") as f:
        header = {
            "model": model,
            "tokenizers": tokenizers.__version__,
            "revision_hash": f"{fnv1a64(raw):016x}",
            "cases": len(cases),
        }
        f.write(json.dumps(header) + "\n")
        for text in cases:
            ids = tok.encode(text).ids
            # HF decode() defaults to skip_special_tokens=True — special
            # added tokens (the EOS here) decode to nothing. The gate's
            # round trip pins the VERBATIM semantics explicitly.
            decoded = tok.decode(ids, skip_special_tokens=False)
            # An NFC tokenizer round-trips to the NFC form of the input.
            import unicodedata
            expect = unicodedata.normalize("NFC", text) if is_qwen else text
            if decoded != expect:
                sys.exit(f"HF verbatim round-trip failed for {text!r} -> {decoded!r}")
            f.write(json.dumps({"text": text, "ids": ids}) + "\n")
    print(f"wrote {out_path}: {len(cases)} cases, revision "
          f"{header['revision_hash']} (tokenizers {tokenizers.__version__})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generates src/text/unicode_tables.hpp — the codepoint tables behind
the tokenizer's pretokenizer scanners and its NFC normalizer (M6 Stage 3;
the Qwen additions 2026-09-09).

The GLM scanner implements this exact regex (the checkpoint's
tokenizer.json Split pattern, verified at load and refused otherwise):

  (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\\r\\n\\p{L}\\p{N}]?\\p{L}+
  | \\p{N}{1,3} | ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*
  | \\s*[\\r\\n]+ | \\s+(?!\\S) | \\s+

and the Qwen3.8 scanner its sibling with [\\p{L}\\p{M}]+, a single
\\p{N}, and \\p{M} excluded from the punctuation class, which together
need four Unicode predicates on codepoints:

  is_letter      Unicode General_Category L* (Lu, Ll, Lt, Lm, Lo)
  is_lower       Ll; is_upper_or_title: Lu, Lt (the Flint pattern's case classes)
  is_mark        Unicode General_Category M* (Mn, Mc, Me)
  is_number      Unicode General_Category N* (Nd, Nl, No)
  is_upper       Unicode General_Category Lu and Lt (the Nemotron/Dory
                 CamelCase scanner's upper class; U = (L and not Ll) or M)
  is_lower       Unicode General_Category Ll (the same scanner's lower
                 class; L2 = (L and not (Lu or Lt)) or M)
  is_white_space Unicode White_Space property (NOT category Zs alone:
                 0x09-0x0D and 0x85 are White_Space but Cc/Cf)

The NFC normalizer (Qwen's tokenizer.json normalizer) needs the canonical
decomposition mappings (one level, 1-2 codepoints; Hangul is algorithmic),
the nonzero canonical combining classes, the primary composites (the
composition exclusions applied: a length-2 canonical decomposition whose
NFC recomposes to the composite), and the quick-check set (codepoints
whose NFC differs from themselves, or that may combine with a preceding
one: nonzero ccc, the second element of a primary composite, the Hangul
V and T jamo) — a string without any of those is already NFC.

The tables are generated from Python's unicodedata at authoring time and
COMMITTED (reproducible via this script). Unicode version skew caveat:
tokenizers 0.23.1's regex engine may carry a newer Unicode revision than
this generator's; the differential goldens (tests/data/
glm_tokenizer_goldens.jsonl) avoid codepoints added after Unicode 13.0, so
a skew cannot flake the gate — but re-run this script + regen the goldens
together when bumping either side.

Usage: tools/gen_unicode_tables.py [output_path]
"""
import sys
import unicodedata

MAX_CODEPOINT = 0x110000

# Unicode White_Space property (stable set; verified against the standard).
WHITE_SPACE = (
    list(range(0x09, 0x0E)) + [0x20, 0x85, 0xA0, 0x1680]
    + list(range(0x2000, 0x200B)) + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
)


def ranges_of(predicate):
    out = []
    start = None
    for cp in range(MAX_CODEPOINT):
        if predicate(cp):
            if start is None:
                start = cp
        elif start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, MAX_CODEPOINT - 1))
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else \
        "src/text/unicode_tables.hpp"
    letters = ranges_of(
        lambda cp: unicodedata.category(chr(cp)) in ("Lu", "Ll", "Lt", "Lm", "Lo"))
    # The CamelCase case tables (the Nemotron/Dory scanner's U and L2, the
    # Flint pattern's case classes): Lu+Lt versus Ll — the same ranges both
    # patterns use.
    lowers = ranges_of(lambda cp: unicodedata.category(chr(cp)) == "Ll")
    uppers = ranges_of(lambda cp: unicodedata.category(chr(cp)) in ("Lu", "Lt"))
    marks = ranges_of(
        lambda cp: unicodedata.category(chr(cp)) in ("Mn", "Mc", "Me"))
    numbers = ranges_of(
        lambda cp: unicodedata.category(chr(cp)) in ("Nd", "Nl", "No"))
    puncts = ranges_of(
        lambda cp: unicodedata.category(chr(cp)) in ("Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"))
    symbols = ranges_of(
        lambda cp: unicodedata.category(chr(cp)) in ("Sm", "Sc", "Sk", "So"))
    ws_set = set(WHITE_SPACE)
    spaces = ranges_of(lambda cp: cp in ws_set)

    # ---- NFC data -------------------------------------------------------
    decomp = {}  # cp -> (a, b) or (a,)
    for cp in range(MAX_CODEPOINT):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        d = unicodedata.decomposition(chr(cp))
        if not d or d.startswith("<"):
            continue
        parts = tuple(int(x, 16) for x in d.split())
        assert 1 <= len(parts) <= 2, (cp, parts)
        decomp[cp] = parts
    ccc_runs = []
    start = None
    prev = 0
    for cp in range(MAX_CODEPOINT + 1):
        c = unicodedata.combining(chr(cp)) if cp < MAX_CODEPOINT else 0
        if c != prev:
            if prev != 0:
                ccc_runs.append((start, cp - 1, prev))
            start = cp
            prev = c
    compose = {}
    for cp, parts in decomp.items():
        if len(parts) == 2 and unicodedata.normalize("NFC", chr(parts[0]) + chr(parts[1])) == chr(cp):
            compose[parts] = cp
    seconds = {b for (_, b) in compose}
    def quick_no_or_maybe(cp):
        if 0xD800 <= cp <= 0xDFFF:
            return False
        ch = chr(cp)
        if unicodedata.normalize("NFC", ch) != ch:
            return True
        if unicodedata.combining(ch) != 0 or cp in seconds:
            return True
        return 0x1161 <= cp <= 0x1175 or 0x11A8 <= cp <= 0x11C2
    quick = ranges_of(quick_no_or_maybe)

    def emit(name, table, note):
        lines = [
            f"inline constexpr uint32_t k{name}Ranges[][2] = {{",
        ]
        row = "    "
        for lo, hi in table:
            cell = f"{{{hex(lo)}, {hex(hi)}}}, "
            # 8 pairs per row keeps the file readable
            if row.count("{") == 8:
                lines.append(row.rstrip())
                row = "    "
            row += cell
        if row.strip():
            lines.append(row.rstrip().rstrip(","))
        lines.append("};")
        body = "\n".join(lines)
        return f"// {note} — {len(table)} ranges\n{body}"

    def emit_rows(name, rows, width, note, elem="uint32_t"):
        lines = [f"inline constexpr {elem} k{name}[][{width}] = {{"]
        row = "    "
        for r in rows:
            cell = "{" + ", ".join(hex(v) for v in r) + "}, "
            if row.count("{") == 6:
                lines.append(row.rstrip())
                row = "    "
            row += cell
        if row.strip():
            lines.append(row.rstrip().rstrip(","))
        lines.append("};")
        return f"// {note} — {len(rows)} entries\n" + "\n".join(lines)

    decomp_rows = [(cp, parts[0], parts[1] if len(parts) == 2 else 0) for cp, parts in sorted(decomp.items())]
    compose_rows = [(a, b, c) for (a, b), c in sorted(compose.items())]

    text = f"""#pragma once
// GENERATED by tools/gen_unicode_tables.py — DO NOT EDIT BY HAND.
// Unicode {unicodedata.unidata_version} (Python {sys.version.split()[0]}).
// Codepoint tables for the tokenizer's pretokenizer scanners and its NFC
// normalizer; binary-search lookup. See the generator's docstring for the
// exact regexes these serve and the version-skew caveat.
#include <cstddef>
#include <cstdint>
#include <iterator>

namespace dgpp::text::unicode {{

{emit("Letter", letters, "General_Category L* (Lu, Ll, Lt, Lm, Lo)")}

{emit("Lower", lowers, "General_Category Ll — the case classes' lower range (the Flint pattern's, the Nemotron/Dory scanner's L2)")}

{emit("Upper", uppers, "General_Category Lu, Lt — the case classes' upper range (the Flint pattern's, the Nemotron/Dory scanner's U)")}

{emit("Mark", marks, "General_Category M* (Mn, Mc, Me)")}

{emit("Number", numbers, "General_Category N* (Nd, Nl, No)")}

{emit("WhiteSpace", spaces, "Unicode White_Space property (not category Zs alone)")}

{emit("Punct", puncts, "General_Category P* (Pc, Pd, Ps, Pe, Pi, Pf, Po) — the DeepSeek pattern's \\p{{P}}")}

{emit("Symbol", symbols, "General_Category S* (Sm, Sc, Sk, So) — the DeepSeek pattern's \\p{{S}}")}

{emit("NfcQuick", quick, "NFC quick check: codepoints that are not QC=Yes (No or Maybe)")}

{emit_rows("Ccc", ccc_runs, 3, "Nonzero canonical combining classes as runs {{lo, hi, ccc}}")}

{emit_rows("Decomp", decomp_rows, 3, "Canonical decompositions {{cp, first, second-or-0}}, sorted by cp")}

{emit_rows("Compose", compose_rows, 3, "Primary composites {{first, second, composite}}, sorted by (first, second)")}

inline bool in_ranges(const uint32_t (*ranges)[2], size_t n, uint32_t cp) {{
  size_t lo = 0, hi = n;
  while (lo < hi) {{
    const size_t mid = (lo + hi) / 2;
    if (cp < ranges[mid][0]) hi = mid;
    else if (cp > ranges[mid][1]) lo = mid + 1;
    else return true;
  }}
  return false;
}}

inline bool is_letter(uint32_t cp) {{
  return in_ranges(kLetterRanges, std::size(kLetterRanges), cp);
}}
inline bool is_number(uint32_t cp) {{
  return in_ranges(kNumberRanges, std::size(kNumberRanges), cp);
}}
inline bool is_white_space(uint32_t cp) {{
  return in_ranges(kWhiteSpaceRanges, std::size(kWhiteSpaceRanges), cp);
}}
inline bool is_mark(uint32_t cp) {{
  return in_ranges(kMarkRanges, std::size(kMarkRanges), cp);
}}
inline bool is_lower(uint32_t cp) {{
  return in_ranges(kLowerRanges, std::size(kLowerRanges), cp);
}}
inline bool is_upper_or_title(uint32_t cp) {{
  return in_ranges(kUpperRanges, std::size(kUpperRanges), cp);
}}
inline bool is_upper(uint32_t cp) {{
  return in_ranges(kUpperRanges, std::size(kUpperRanges), cp);
}}
inline bool is_punctuation(uint32_t cp) {{
  return in_ranges(kPunctRanges, std::size(kPunctRanges), cp);
}}
inline bool is_symbol(uint32_t cp) {{
  return in_ranges(kSymbolRanges, std::size(kSymbolRanges), cp);
}}
inline bool nfc_quick_no_or_maybe(uint32_t cp) {{
  return in_ranges(kNfcQuickRanges, std::size(kNfcQuickRanges), cp);
}}

}}  // namespace dgpp::text::unicode
"""
    with open(out_path, "w") as f:
        f.write(text)
    print(f"wrote {out_path}: {len(letters)} letter, {len(marks)} mark, {len(numbers)} number, "
          f"{len(puncts)} punctuation, {len(symbols)} symbol, "
          f"{len(spaces)} white-space, {len(quick)} nfc-quick ranges; {len(ccc_runs)} ccc runs, "
          f"{len(decomp_rows)} decompositions, {len(compose_rows)} composites (Unicode "
          f"{unicodedata.unidata_version})")


if __name__ == "__main__":
    main()

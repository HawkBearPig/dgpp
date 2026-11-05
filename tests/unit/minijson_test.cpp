#include "loaders/minijson.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "common/test.hpp"

using dgpp::minijson::parse;
using dgpp::minijson::Value;

namespace {
void rejects_json(std::string_view input, bool document = false) {
  try {
    if (document)
      dgpp::minijson::parse_document(input);
    else
      parse(input);
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error("accepted malformed JSON: " + std::string(input));
}
}  // namespace

DGPP_TEST(minijson_number_grammar) {
  for (const char* token :
       {"1e",     "1e+",      "1e-",       "1E",    "1E+",    "1E-",
        "1.2.3",  "1-2",      "1+2",       "1e2e3", "1e++2",  "1e--2",
        "01",     "00",       "-01",       "00.1",  "01e2",   "1.",
        "-1.",    "1.e2",     "-",         "--1",   "+1",     ".1",
        "NaN",    "Infinity", "-Infinity", "1e309", "-1e309", "1.7976931348623159e308",
        "1e-400", "-1e-400"}) {
    rejects_json(token);
    rejects_json("[" + std::string(token) + "]");
    rejects_json("{\"n\":" + std::string(token) + "}");
  }
}

DGPP_TEST(minijson_number_representations) {
  using K = Value::Kind;
  const std::pair<const char*, int64_t> integers[] = {{"0", 0},
                                                      {"-0", 0},
                                                      {"1", 1},
                                                      {"-1", -1},
                                                      {"9007199254740993", 9007199254740993LL},
                                                      {"9223372036854775807", INT64_MAX},
                                                      {"-9223372036854775808", INT64_MIN}};
  for (const auto& [text, expected] : integers) {
    const auto result = parse(text);
    if (result.root.kind() != K::Int || result.root.as_int() != expected)
      throw std::runtime_error("integer representation: " + std::string(text));
  }
  const std::pair<const char*, double> doubles[] = {
      {"0.0", 0.0},
      {"-0.0", -0.0},
      {"-0e0", -0.0},
      {"1.0", 1.0},
      {"1e0", 1.0},
      {"1E+2", 100.0},
      {"1e-2", .01},
      {"0.125", .125},
      {"9223372036854775808", 0x1p63},
      {"-9223372036854775809", -0x1p63},
      {"18446744073709551615", 0x1p64},
      {"1.7976931348623157e308", std::numeric_limits<double>::max()},
      {"2.2250738585072014e-308", std::numeric_limits<double>::min()},
      {"4.9406564584124654e-324", std::numeric_limits<double>::denorm_min()}};
  for (const auto& [text, expected] : doubles) {
    const auto result = parse(text);
    if (result.root.kind() != K::Double || result.root.as_double() != expected ||
        std::signbit(result.root.as_double()) != std::signbit(expected))
      throw std::runtime_error("double representation: " + std::string(text));
  }
  const auto nested = parse("{\"n\":[1,1.0,1e0 \t\r\n],\"end\":-2}");
  if (nested.root.at("n").items()[0].kind() != K::Int ||
      nested.root.at("n").items()[1].kind() != K::Double ||
      nested.root.at("n").items()[2].kind() != K::Double || nested.root.at("end").as_int() != -2)
    throw std::runtime_error("number delimiters or representations changed");
}

DGPP_TEST(minijson_document_requires_only_json_whitespace_after_value) {
  for (const char* root : {"0", "-0.0", "true", "null", "\"x\"", "[]", "{}"}) {
    const std::string padded = " \t\r\n" + std::string(root) + " \t\r\n";
    const auto result = dgpp::minijson::parse_document(padded);
    if (result.consumed != 4 + std::strlen(root))
      throw std::runtime_error("document consumed count includes trailing whitespace");
    for (const std::string& suffix :
         {std::string("junk"), std::string("{}"), std::string("[]"), std::string("true"),
          std::string("0"), std::string(1, '\0'), std::string("\v"), std::string("\f"),
          std::string("//comment"), std::string("/*comment*/")})
      rejects_json(std::string(root) + " \n" + suffix, true);
  }
  for (const char* input : {"", " \n\t", "0x10", "1foo", "{}suffix", "{}{}"})
    rejects_json(input, true);
  for (const auto& [input, count] :
       std::initializer_list<std::pair<const char*, size_t>>{{"{}suffix", 2},
                                                             {"1 rest", 1},
                                                             {"1,2", 1},
                                                             {"truefalse", 4},
                                                             {"0x10", 1},
                                                             {"  [1, 2, 3]   \n\t ", 11}})
    if (parse(input).consumed != count) throw std::runtime_error("prefix API changed");
}

DGPP_TEST(minijson_object_scalars) {
  const char* json = R"({"a":1,"b":-2,"c":3.5,"d":true,"e":false,"f":null})";
  auto r = parse(json);
  const Value& o = r.root;
  if (!o.is_object()) throw std::runtime_error("not object");
  if (o.at("a").as_int() != 1) throw std::runtime_error("a");
  if (o.at("b").as_int() != -2) throw std::runtime_error("b");
  if (o.at("c").as_double() != 3.5) throw std::runtime_error("c");
  if (!o.at("d").as_bool()) throw std::runtime_error("d");
  if (o.at("e").as_bool(true)) throw std::runtime_error("e");
  if (!o.at("f").is_null()) throw std::runtime_error("f");
}

DGPP_TEST(minijson_nested_arrays_and_views) {
  // Shapes come from safetensors headers: array of ints under string keys.
  const char* json =
      R"({"t":{"dtype":"BF16","shape":[2,3,4],"data_offsets":[0,48]}})";
  auto r = parse(json);
  const Value& shape = r.root.at("t").at("shape");
  if (!shape.is_array() || shape.items().size() != 3)
    throw std::runtime_error("shape");
  if (shape.items()[2].as_int() != 4) throw std::runtime_error("elem");
  if (std::string_view(r.root.at("t").at("dtype").as_string()) != "BF16")
    throw std::runtime_error("dtype view");
}

DGPP_TEST(minijson_escape_decode_produces_owned_storage) {
  const char* json = "{\"k\\n\\u0041\" : \"v\\t\\\"q\\\"\"}";
  auto r = parse(json);
  if (r.root.at(std::string_view("k\nA")).as_string() !=
      std::string_view("v\t\"q\""))
    throw std::runtime_error("escape decode mismatch");
}

DGPP_TEST(minijson_unicode_escaped_and_raw_strings_agree) {
  const std::pair<const char*, const char*> cases[] = {
      {R"(\u0041)", "A"}, {R"(\u00e9)", "\xC3\xA9"},
      {R"(\u0800)", "\xE0\xA0\x80"}, {R"(\ud7ff)", "\xED\x9F\xBF"},
      {R"(\ue000)", "\xEE\x80\x80"}, {R"(\uffff)", "\xEF\xBF\xBF"},
      {R"(\ud83d\udced)", "📭"}, {R"(\uD83C\uDFE2)", "🏢"},
      {R"(\ud840\udc00)", "𠀀"},
      {R"(\ud800\udc00)", "\xF0\x90\x80\x80"},
      {R"(\udbff\udfff)", "\xF4\x8F\xBF\xBF"},
      {R"(\ud83d\udced\ud83c\udfe2)", "📭🏢"},
  };
  for (const auto& [escaped, utf8] : cases) {
    const std::string input = "\"" + std::string(escaped) + "\"";
    const std::string raw = "\"" + std::string(utf8) + "\"";
    const auto decoded = parse(input);
    if (decoded.root.as_string() != utf8 ||
        decoded.root.as_string() != parse(raw).root.as_string())
      throw std::runtime_error("escaped/raw mismatch: " + input);
    const std::string object = "{\"" + std::string(escaped) + "\":\"ok\"}";
    if (parse(object).root.at(utf8).as_string() != "ok")
      throw std::runtime_error("escaped object key mismatch");
  }
}

DGPP_TEST(minijson_unicode_unpaired_surrogates_preserve_following_text) {
  const std::pair<const char*, const char*> cases[] = {
      {R"("\ud800")", "�"}, {R"("\udbff")", "�"},
      {R"("\udc00")", "�"}, {R"("\udfff")", "�"},
      {R"("\ud800x")", "�x"}, {R"("\ud800\u0041")", "�A"},
      {R"("\ud800\n")", "�\n"}, {R"("\ud800\ud800")", "��"},
      {R"("\udced\ud83d")", "��"},
      {R"("\ud800\ud83d\udced")", "�📭"},
      {R"("\udc00\ud83d\udced")", "�📭"},
      {R"("\\ud83d\\udced")", R"(\ud83d\udced)"},
  };
  for (const auto& [input, expected] : cases)
    if (parse(input).root.as_string() != expected)
      throw std::runtime_error(std::string("unpaired surrogate mismatch: ") + input);
}

DGPP_TEST(minijson_unicode_requires_four_hex_digits) {
  for (const char* input : {R"("\u")", R"("\u123")", R"("\u12xz")",
                            R"("\u000g")", R"("\ug000")", R"("\u+123")",
                            R"("\u-123")", R"("\u 123")", R"("\u0x41")",
                            R"("\ud800\u")", R"("\ud800\udc0")",
                            R"("\ud800\udc0x")", R"("\ud800\u12xz")"}) {
    bool threw = false;
    try {
      (void)parse(input);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    if (!threw) throw std::runtime_error(std::string("accepted malformed escape: ") + input);
  }
}

DGPP_TEST(minijson_trailing_whitespace_and_consumed_count) {
  auto r = parse("  [1, 2, 3]   \n\t ");
  if (!r.root.is_array() || r.root.items().size() != 3)
    throw std::runtime_error("array");
  if (r.consumed != std::strlen("  [1, 2, 3]"))
    throw std::runtime_error("consumed count");
}

// The nesting bound (2026-09-05, the malformed-HTTP fuzzer's find under
// ASan): a document deeper than 256 containers is refused as malformed
// instead of running the parser's stack out; 200 levels still parse.
DGPP_TEST(minijson_nesting_depth_is_bounded) {
  std::string ok(200, '[');
  ok += std::string(200, ']');
  const dgpp::minijson::ParseResult r = dgpp::minijson::parse(ok);
  if (!r.root.is_array()) throw std::runtime_error("200 levels parse");
  for (const size_t depth : {size_t{257}, size_t{100000}}) {
    std::string deep(depth, '[');
    deep += std::string(depth, ']');
    bool threw = false;
    std::string msg;
    try {
      (void)dgpp::minijson::parse(deep);
    } catch (const std::exception& e) {
      threw = true;
      msg = e.what();
    }
    if (!threw || msg.find("nesting deeper") == std::string::npos)
      throw std::runtime_error("depth " + std::to_string(depth) +
                               " refused loudly (got: " + msg + ")");
    std::string obj;
    for (size_t i = 0; i < depth; ++i) obj += "{\"a\":";
    obj += "1";
    obj += std::string(depth, '}');
    threw = false;
    try {
      (void)dgpp::minijson::parse(obj);
    } catch (const std::exception&) {
      threw = true;
    }
    if (!threw) throw std::runtime_error("object depth " + std::to_string(depth) + " refused");
  }
}

DGPP_TEST(minijson_malformed_throws) {
  bool threw = false;
  try {
    parse("{\"a\":}");
  } catch (const std::exception&) {
    threw = true;
  }
  if (!threw) throw std::runtime_error("should throw on malformed");
}

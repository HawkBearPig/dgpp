#include "text/chat_template.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"

namespace {
using dgpp::text::ChatTemplate;
using dgpp::text::Value;
constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

void equal(const std::string& actual, const std::string& expected) {
  if (actual != expected)
    throw std::runtime_error("render mismatch: expected " + expected + ", got " + actual);
}

std::string render(const std::string& source, Value::Members globals = {}) {
  return ChatTemplate::compile(source).render(Value::map_value(std::move(globals)));
}

Value::Members numbers(int64_t a, int64_t b, int64_t step = 1) {
  return {{"a", Value::integer(a)}, {"b", Value::integer(b)}, {"step", Value::integer(step)}};
}

void rejects(const std::string& source, Value::Members globals, const char* message) {
  try {
    (void)render(source, std::move(globals));
  } catch (const std::runtime_error& error) {
    const std::string text = error.what();
    if (text.find("chat-template: line 2:") != std::string::npos &&
        text.find(message) != std::string::npos)
      return;
    throw;
  }
  throw std::runtime_error("expected template error");
}
}  // namespace

DGPP_TEST(chat_template_replace_empty_pattern) {
  struct Case {
    std::string input, replacement, expected;
  };
  const std::vector<Case> cases = {
      {"", "", ""},
      {"", "-", "-"},
      {"abc", "", "abc"},
      {"abc", "-", "-a-b-c-"},
      {"é中😀", "|", "|é|中|😀|"},
      {"e\xCC\x81", "_",
       "_e_\xCC\x81"
       "_"},
      {std::string("a\0b", 3), "|", std::string("|a|\0|b|", 7)},
  };
  for (const auto& c : cases)
    equal(render("{{ text|replace('', replacement) }}",
                 {{"text", Value::string_value(c.input)},
                  {"replacement", Value::string_value(c.replacement)}}),
          c.expected);
}

DGPP_TEST(chat_template_replace_nonempty_pattern) {
  equal(render("{{ 'aaaa'|replace('aa', 'b') }}"), "bb");
  equal(render("{{ 'abc'|replace('b', '') }}"), "ac");
  equal(render("{{ 'abc'|replace('z', 'x') }}"), "abc");
}

DGPP_TEST(chat_template_integer_index_boundaries) {
  for (int64_t i : {kMin, kMax, int64_t(-4), int64_t(3)}) {
    equal(render("{{ [10,20,30][a] is undefined }}", numbers(i, 0)), "True");
    equal(render("{{ [][a] is undefined }}", numbers(i, 0)), "True");
    equal(render("{{ 'é中😀'[a] is undefined }}", numbers(i, 0)), "True");
  }
  equal(render("{{ [10,20,30][-3] }}"), "10");
  equal(render("{{ [10,20,30][-1] }}"), "30");
  equal(render("{{ [10,20,30][0] }}"), "10");
  equal(render("{{ 'é中😀'[-1] }}"), "😀");
}

DGPP_TEST(chat_template_range_boundary_steps) {
  struct Case {
    int64_t start, stop, step;
    const char* expected;
  };
  const Case cases[] = {
      {kMax - 1, kMax, 2, "[9223372036854775806]"},
      {kMin + 1, kMin, -2, "[-9223372036854775807]"},
      {kMin, kMax, kMax, "[-9223372036854775808, -1, 9223372036854775806]"},
      {kMax, kMin, kMin, "[9223372036854775807, -1]"},
      {kMin, kMin + 2, 1, "[-9223372036854775808, -9223372036854775807]"},
      {kMax, kMax - 2, -1, "[9223372036854775807, 9223372036854775806]"},
      {kMin, kMax, -1, "[]"},
      {kMax, kMin, 1, "[]"},
      {0, 0, kMin, "[]"},
      {1, 6, 2, "[1, 3, 5]"},
      {5, 0, -2, "[5, 3, 1]"},
  };
  for (const auto& c : cases)
    equal(render("{{ range(a,b,step)|tojson }}", numbers(c.start, c.stop, c.step)), c.expected);
  equal(render("{{ range(3)|tojson }}"), "[0, 1, 2]");
  equal(render("{{ range(2,4)|tojson }}"), "[2, 3]");
  rejects("\n{{ range(a,b,step) }}", numbers(0, 1, 0), "step must not be zero");
}

DGPP_TEST(chat_template_slice_boundary_steps) {
  equal(render("{{ [0,1,2][1:3:step]|tojson }}", numbers(0, 0, kMax)), "[1]");
  equal(render("{{ [0,1,2][::step]|tojson }}", numbers(0, 0, kMin)), "[2]");
  equal(render("{{ [0,1,2][a:b]|tojson }}", numbers(kMin, kMax)), "[0, 1, 2]");
  equal(render("{{ [0,1,2][a:b:-1]|tojson }}", numbers(kMax, kMin)), "[2, 1, 0]");
  equal(render("{{ [][::step]|tojson }}", numbers(0, 0, kMin)), "[]");
  equal(render("{{ 'é中😀'[1:3:step] }}", numbers(0, 0, kMax)), "中");
  equal(render("{{ 'é中😀'[::step] }}", numbers(0, 0, kMin)), "😀");
  equal(render("{{ 'abc'[::-1] }}"), "cba");
  rejects("\n{{ [0,1,2][::step] }}", numbers(0, 0, 0), "step cannot be zero");
}

DGPP_TEST(chat_template_integer_arithmetic_boundaries) {
  struct Case {
    int64_t a, b, sum;
  };
  const Case sums[] = {{kMin, 0, kMin}, {kMax, 0, kMax},     {kMin, kMax, -1},
                       {0, kMin, kMin}, {kMin, 1, kMin + 1}, {kMax, -1, kMax - 1}};
  for (const auto& c : sums) equal(render("{{ a + b }}", numbers(c.a, c.b)), std::to_string(c.sum));
  const Case differences[] = {{kMin, kMin, 0}, {kMax, kMax, 0},  {kMin, 0, kMin},
                              {kMax, 0, kMax}, {-1, kMin, kMax}, {kMin, -1, kMin + 1}};
  for (const auto& c : differences)
    equal(render("{{ a - b }}", numbers(c.a, c.b)), std::to_string(c.sum));
  equal(render("{{ a + b }}", {{"a", Value::number(1.5)}, {"b", Value::integer(2)}}), "3.5");
  equal(render("{{ a - b }}", {{"a", Value::number(1.5)}, {"b", Value::integer(2)}}), "-0.5");
  equal(render("{{ 'hello' + ' world' }}"), "hello world");
}

DGPP_TEST(chat_template_integer_arithmetic_overflow_refuses) {
  rejects("\n{{ a + b }}", numbers(kMax, 1), "integer addition overflow");
  rejects("\n{{ a + b }}", numbers(kMin, -1), "integer addition overflow");
  rejects("\n{{ a + b }}", numbers(kMin, kMin), "integer addition overflow");
  rejects("\n{{ a - b }}", numbers(kMin, 1), "integer subtraction overflow");
  rejects("\n{{ a - b }}", numbers(kMax, -1), "integer subtraction overflow");
  rejects("\n{{ a - b }}", numbers(0, kMin), "integer subtraction overflow");
}
